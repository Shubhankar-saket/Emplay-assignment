"""Lazy-loaded local ONNX models (fastembed): dense embedder, BM25 sparse embedder, cross-encoder."""
from __future__ import annotations

import re
from functools import lru_cache

from core.config import get_settings

# IDs like JA-207652 / 210-BLYZ / WD22TB4: BM25 would split them on '-', so a joined form is appended.
_ID_RE = re.compile(r"\b[A-Za-z0-9]+(?:[-/][A-Za-z0-9]+)+\b")


def bm25_text(text: str) -> str:
    joined = {m.replace("-", "").replace("/", "").lower() for m in _ID_RE.findall(text) if any(c.isdigit() for c in m)}
    return text + ("\n" + " ".join(sorted(joined)) if joined else "")


@lru_cache
def dense_model():
    from fastembed import TextEmbedding
    return TextEmbedding(get_settings().dense_model)


@lru_cache
def sparse_model():
    from fastembed import SparseTextEmbedding
    s = get_settings()
    return SparseTextEmbedding(s.sparse_model, avg_len=s.bm25_avg_len)


@lru_cache
def reranker():
    from fastembed.rerank.cross_encoder import TextCrossEncoder
    return TextCrossEncoder(get_settings().rerank_model)
