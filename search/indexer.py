"""Incremental indexing: only files whose content hash is not yet in the collection are embedded."""
from __future__ import annotations

from pathlib import Path

from qdrant_client import models

from core.config import Settings, get_settings
from core.logging import get_logger, log_event
from ingestion.pipeline import file_sha256, list_bid_files, parse_file
from search import store
from search.chunking import chunk_document
from search.models import bm25_text, dense_model, sparse_model

log = get_logger("search.indexer")
_BATCH = 64


def _upsert(chunks) -> None:
    dense = list(dense_model().embed([c.embedding_text() for c in chunks], batch_size=_BATCH))
    sparse = list(sparse_model().embed([bm25_text(c.embedding_text()) for c in chunks], batch_size=_BATCH))
    store.ensure_collection(len(dense[0]))
    points = [
        models.PointStruct(
            id=c.chunk_id,
            vector={store.DENSE: d.tolist(),
                    store.SPARSE: models.SparseVector(indices=s.indices.tolist(), values=s.values.tolist())},
            payload=c.model_dump(),
        )
        for c, d, s in zip(chunks, dense, sparse)
    ]
    for i in range(0, len(points), _BATCH):
        store.get_client().upsert(get_settings().collection, points=points[i:i + _BATCH])


def index_bid(bid_dir: Path, settings: Settings | None = None, force: bool = False) -> dict:
    """Index one bid folder. Unchanged files are skipped; changed files replace their old chunks."""
    settings = settings or get_settings()
    bid_id = bid_dir.name
    stats = {"bid_id": bid_id, "indexed": [], "skipped": [], "failed": [], "chunks": 0}
    for path in list_bid_files(bid_dir):
        file_hash = file_sha256(path)
        if not force and store.count(store.build_filter(None, bid_id=bid_id, file_hash=file_hash)):
            stats["skipped"].append(path.name)
            continue
        doc = parse_file(path, bid_id, settings, file_hash=file_hash)
        if doc is None:
            stats["failed"].append(path.name)
            continue
        store.delete(store.build_filter(None, bid_id=bid_id, file_name=path.name))  # old version, if any
        chunks = chunk_document(doc, settings.chunk_max_chars, settings.chunk_overlap_chars)
        if chunks:
            _upsert(chunks)
        stats["indexed"].append(path.name)
        stats["chunks"] += len(chunks)
    log_event(log, "bid_indexed", **stats)
    return stats


def delete_bid(bid_id: str) -> None:
    store.delete(store.build_filter(None, bid_id=bid_id))
    log_event(log, "bid_deleted", bid_id=bid_id)
