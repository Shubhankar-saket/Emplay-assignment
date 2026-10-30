"""Hybrid search: dense + BM25 prefetch, RRF fusion in Qdrant, cross-encoder rerank, query expansion."""
from __future__ import annotations

import re

from qdrant_client import models

from core.config import get_settings
from core.logging import get_logger, log_event
from core.models import Chunk, SearchFilters, SearchHit, SearchMode
from search import store
from search.models import bm25_text, dense_model, reranker, sparse_model

log = get_logger("search.engine")

# Domain synonyms for keyword recall: "deadline" also searches "due date", "closing date", ...
SYNONYMS: dict[str, list[str]] = {
    "deadline": ["due date", "closing date", "submission date", "solicitation due"],
    "due date": ["deadline", "closing date", "solicitation due"],
    "pre-bid": ["pre-proposal", "prebid conference", "pre-bid meeting"],
    "warranty": ["guarantee", "extended warranty", "support"],
    "payment": ["invoice", "net 30", "payment terms"],
    "delivery": ["ship", "deliver within", "delivery date"],
    "contact": ["buyer", "procurement officer", "email", "phone"],
    "bond": ["bid bond", "bid security", "surety", "performance bond"],
    "insurance": ["liability", "coverage", "certificate of insurance"],
    "cooperative": ["piggyback", "interlocal", "co-op", "cooperative contract"],
    "specs": ["specification", "configuration", "processor", "memory"],
    "price": ["pricing", "cost", "unit price", "price schedule"],
    "term": ["contract term", "contract period", "renewal"],
}


def expand_query(query: str) -> str:
    q = query.lower()
    extra = [s for k, syns in SYNONYMS.items() if re.search(rf"\b{re.escape(k)}\b", q) for s in syns]
    return query + (" " + " ".join(dict.fromkeys(extra)) if extra else "")


def _sparse_query(text: str) -> models.SparseVector:
    s = next(iter(sparse_model().query_embed(bm25_text(text))))
    return models.SparseVector(indices=s.indices.tolist(), values=s.values.tolist())


def search(query: str, filters: SearchFilters | None = None, k: int = 5,
           mode: SearchMode = "hybrid_rerank", expand: bool = True) -> list[SearchHit]:
    s, client = get_settings(), store.get_client()
    if not client.collection_exists(s.collection):
        return []
    flt = store.build_filter(filters)
    kw_query = expand_query(query) if expand else query
    limit = s.rerank_candidates if mode == "hybrid_rerank" else k

    if mode == "dense":
        res = client.query_points(s.collection, query=next(iter(dense_model().query_embed(query))).tolist(),
                                  using=store.DENSE, query_filter=flt, limit=limit, with_payload=True)
    elif mode == "bm25":
        res = client.query_points(s.collection, query=_sparse_query(kw_query), using=store.SPARSE,
                                  query_filter=flt, limit=limit, with_payload=True)
    else:
        dense_vec = next(iter(dense_model().query_embed(query))).tolist()
        res = client.query_points(
            s.collection,
            prefetch=[models.Prefetch(query=dense_vec, using=store.DENSE, filter=flt, limit=s.prefetch_limit),
                      models.Prefetch(query=_sparse_query(kw_query), using=store.SPARSE, filter=flt,
                                      limit=s.prefetch_limit)],
            query=models.FusionQuery(fusion=models.Fusion.RRF), limit=limit, with_payload=True)

    chunks = [Chunk.model_validate(p.payload) for p in res.points]
    scores = [p.score for p in res.points]
    if mode == "hybrid_rerank" and chunks:
        # Blend cross-encoder rank with hybrid rank (RRF, k=60) instead of replacing it, so the
        # small reranker's vocabulary gaps ("POC" vs "procurement officer") can't bury a good hit.
        ce = list(reranker().rerank(query, [c.embedding_text() for c in chunks]))
        ce_rank = {i: r for r, i in enumerate(sorted(range(len(chunks)), key=lambda i: ce[i], reverse=True))}
        scores = [1 / (60 + i) + 1 / (60 + ce_rank[i]) for i in range(len(chunks))]
        order = sorted(range(len(chunks)), key=lambda i: scores[i], reverse=True)
        chunks, scores = [chunks[i] for i in order], [scores[i] for i in order]

    hits = [SearchHit(chunk=c, score=float(sc), rank=i + 1) for i, (c, sc) in enumerate(zip(chunks[:k], scores[:k]))]
    log_event(log, "search", query=query, mode=mode, filters=filters.model_dump() if filters else None,
              hits=[(h.chunk.file_name, h.chunk.page_start) for h in hits])
    return hits
