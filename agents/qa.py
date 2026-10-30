"""Grounded Q&A over search results: answers only from retrieved chunks, with citations."""
from __future__ import annotations

import re

from pydantic import BaseModel, Field

from agents.base import Agent
from agents.llm import LLMError, get_llm
from agents.state import AgentStep
from core.models import Citation, SearchFilters, SearchHit
from search import store
from search.engine import search

NOT_FOUND = "Not found in documents"
_MARKERS = re.compile(r"\s*【[^】]*】")  # inline citation markers some models add; citations are returned separately

SYSTEM = f"""You answer questions about public procurement bid documents.
Use ONLY the numbered sources given. Never use outside knowledge or guess.
Addenda override the original RFP: when sources conflict, prefer the latest addendum and say so.
When comparing bids, name each bid and cite sources for each.
If the sources do not contain the answer, set "answer" to "{NOT_FOUND}" and "sources" to [].
Do not put citation markers in the answer text; list source numbers in "sources" only.
Reply as JSON: {{"answer": "<concise answer>", "sources": [<source numbers used>]}}"""


class Answer(BaseModel):
    question: str
    answer: str
    found: bool
    citations: list[Citation] = Field(default_factory=list)
    quotes: list[str] = Field(default_factory=list)
    trace: list[AgentStep] = Field(default_factory=list)


def format_sources(hits: list[SearchHit], max_chars: int = 1200) -> str:
    blocks = []
    for i, h in enumerate(hits, 1):
        c = h.chunk
        label = c.doc_type + (f" {c.addendum_number}" if c.addendum_number else "")
        blocks.append(f"[{i}] {c.bid_id} | {label} | {c.file_name} p.{c.page_start}\n{c.text[:max_chars]}")
    return "\n\n".join(blocks)


def retrieve_for_question(question: str, filters: SearchFilters | None, k: int) -> list[SearchHit]:
    """One bid: top-k plus its addenda (so 'after all addendums' questions see the changes).
    No bid given: the same per indexed bid, so comparisons get evidence from every bid."""
    bids = [filters.bid_id] if filters and filters.bid_id else store.list_bids()
    per_bid = k if len(bids) == 1 else max(3, k // 2)
    hits: dict[str, SearchHit] = {}
    for bid in bids:
        base = (filters or SearchFilters()).model_copy(update={"bid_id": bid})
        for h in search(question, base, k=per_bid) + search(
                question, base.model_copy(update={"doc_types": ["addendum"]}), k=2):
            hits.setdefault(h.chunk.chunk_id, h)
    return list(hits.values())


class QAAgent(Agent):
    """Q&A / report agent in Q&A mode: retrieve (per bid + addenda), then answer only from the sources."""
    name = "qa"

    def run(self, question: str, filters: SearchFilters | None = None, k: int = 6) -> Answer:
        with self.step("retrieve", question=question, filters=filters.model_dump() if filters else None) as out:
            hits = retrieve_for_question(question, filters, k)
            out["hits"] = [f"{h.chunk.file_name} p.{h.chunk.page_start}" for h in hits]
        if not hits:
            return Answer(question=question, answer=NOT_FOUND, found=False)
        with self.step("answer", sources=len(hits)) as out:
            try:
                data = get_llm().chat_json(SYSTEM, f"Question: {question}\n\nSources:\n{format_sources(hits)}")
            except LLMError as exc:
                out.update(status="error", error=str(exc)[:200])
                return Answer(question=question, answer=f"Answer unavailable: {str(exc)[:150]}", found=False)
            text = _MARKERS.sub("", str(data.get("answer") or NOT_FOUND)).strip()
            used = [hits[n - 1] for n in data.get("sources") or [] if isinstance(n, int) and 1 <= n <= len(hits)]
            found = bool(used) and not text.lower().startswith(NOT_FOUND.lower())
            out.update(found=found, cited=[f"{h.chunk.file_name} p.{h.chunk.page_start}" for h in used])
        return Answer(question=question, answer=text if found else NOT_FOUND, found=found,
                      citations=[h.citation for h in used] if found else [],
                      quotes=[h.chunk.text[:300] for h in used] if found else [])


def answer_question(question: str, filters: SearchFilters | None = None, k: int = 6) -> Answer:
    """Q&A mode entry point: routed through the orchestrator so every step lands in the answer's trace."""
    from agents.orchestrator import Orchestrator
    return Orchestrator.ask(question, filters, k)
