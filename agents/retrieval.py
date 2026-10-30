"""Retrieval agent: runs a field's queries over the bid (plus filtered passes), merges and ranks the hits,
and rewrites queries when the validator rejected the previous evidence."""
from __future__ import annotations

import threading

from agents.base import Agent
from agents.fields import SPECS
from agents.llm import LLMError, get_llm
from agents.state import RetrievalRequest, RetrievalResult
from core.models import SearchFilters, SearchHit
from search.engine import search

_SEARCH_LOCK = threading.Lock()  # embedded Qdrant + ONNX sessions are shared; LLM calls stay parallel

REWRITE_SYSTEM = """You write search queries for a hybrid (keyword + semantic) search engine over procurement
bid documents. Given a field to find and why the previous evidence was rejected, return 3 short, different
queries using the vocabulary such documents use (e.g. "Agency POC", "Solicitation Due", "net 30").
Reply as JSON: {"queries": ["...", "...", "..."]}"""


class RetrievalAgent(Agent):
    name = "retrieval"

    def request_for(self, field: str, queries: list[str] | None = None) -> RetrievalRequest:
        spec = SPECS[field]
        passes: list = [None]
        if spec.doc_types:
            passes.append(list(spec.doc_types))
        if spec.addendum_sensitive and self.state.has_addenda:
            passes.append(["addendum"])
        return RetrievalRequest(field=field, bid_id=self.state.bid_id, queries=queries or list(spec.queries),
                                doc_type_passes=passes)

    def run(self, req: RetrievalRequest) -> RetrievalResult:
        with self.step("retrieve", field=req.field, queries=req.queries,
                       passes=[p or "all" for p in req.doc_type_passes]) as out:
            # fuse all (query, pass) result lists with RRF: chunks several queries agree on rise to the top
            fused: dict[str, list] = {}
            for doc_types in req.doc_type_passes:
                k = 2 if doc_types else req.k  # filtered passes add a couple of targeted hits
                for q in req.queries:
                    with _SEARCH_LOCK:
                        hits = search(q, SearchFilters(bid_id=req.bid_id, doc_types=doc_types), k=k)
                    for h in hits:
                        entry = fused.setdefault(h.chunk.chunk_id, [0.0, h])
                        entry[0] += 1 / (10 + h.rank)
            hits = [h for _, h in sorted(fused.values(), key=lambda x: x[0], reverse=True)][:6]
            out["hits"] = [f"{h.chunk.file_name} p.{h.chunk.page_start}" for h in hits]
        self.state.set_evidence(req.field, hits, req.queries)
        return RetrievalResult(field=req.field, queries=req.queries, hits=hits)

    def rewrite(self, field: str, reason: str) -> list[str]:
        spec = SPECS[field]
        with self.step("rewrite_queries", field=field, reason=reason) as out:
            data = get_llm().chat_json(REWRITE_SYSTEM, (
                f"Field: {spec.label} - {spec.description}\nPrevious queries: {self.state.queries.get(field)}\n"
                f"Rejection reason: {reason}"), fast=True)
            queries = [q for q in data.get("queries", []) if isinstance(q, str) and q.strip()][:3]
            out["queries"] = queries
        return queries or list(spec.queries)
