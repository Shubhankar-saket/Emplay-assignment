"""Qdrant access: embedded on disk when QDRANT_URL is unset, server/cloud otherwise."""
from __future__ import annotations

import atexit
from functools import lru_cache

from qdrant_client import QdrantClient, models

from core.config import get_settings
from core.models import SearchFilters

DENSE, SPARSE = "dense", "bm25"
_INDEXED_FIELDS = {"bid_id": "keyword", "doc_type": "keyword", "file_name": "keyword",
                   "file_hash": "keyword", "addendum_number": "integer"}


@lru_cache
def get_client() -> QdrantClient:
    s = get_settings()
    if s.qdrant_url:
        client = QdrantClient(url=s.qdrant_url, api_key=s.qdrant_api_key, timeout=60)
    else:
        s.qdrant_path.mkdir(parents=True, exist_ok=True)
        client = QdrantClient(path=str(s.qdrant_path))
    atexit.register(client.close)  # embedded mode errors on interpreter exit otherwise
    return client


def ensure_collection(dim: int) -> None:
    client, name = get_client(), get_settings().collection
    if client.collection_exists(name):
        return
    client.create_collection(
        name,
        vectors_config={DENSE: models.VectorParams(size=dim, distance=models.Distance.COSINE)},
        sparse_vectors_config={SPARSE: models.SparseVectorParams(modifier=models.Modifier.IDF)},
    )
    for field, schema in _INDEXED_FIELDS.items():
        client.create_payload_index(name, field, field_schema=schema)


def build_filter(f: SearchFilters | None, **extra) -> models.Filter | None:
    must = []
    if f:
        if f.bid_id:
            must.append(models.FieldCondition(key="bid_id", match=models.MatchValue(value=f.bid_id)))
        if f.doc_types:
            must.append(models.FieldCondition(key="doc_type", match=models.MatchAny(any=list(f.doc_types))))
        if f.addendum_number is not None:
            must.append(models.FieldCondition(key="addendum_number", match=models.MatchValue(value=f.addendum_number)))
        if f.file_name:
            must.append(models.FieldCondition(key="file_name", match=models.MatchValue(value=f.file_name)))
    must += [models.FieldCondition(key=k, match=models.MatchValue(value=v)) for k, v in extra.items()]
    return models.Filter(must=must) if must else None


def count(flt: models.Filter | None) -> int:
    client, name = get_client(), get_settings().collection
    if not client.collection_exists(name):
        return 0
    return client.count(name, count_filter=flt, exact=True).count


def list_bids() -> list[str]:
    """Distinct bid_ids in the collection."""
    client, name = get_client(), get_settings().collection
    if not client.collection_exists(name):
        return []
    bids, offset = set(), None
    while True:
        points, offset = client.scroll(name, limit=1000, offset=offset, with_payload=["bid_id"], with_vectors=False)
        bids.update(p.payload["bid_id"] for p in points)
        if offset is None:
            return sorted(bids)


def delete(flt: models.Filter) -> None:
    client, name = get_client(), get_settings().collection
    if client.collection_exists(name):
        client.delete(name, points_selector=models.FilterSelector(filter=flt))
