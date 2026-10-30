"""Ingestion agent: parses the bid folder and triggers (incremental) indexing."""
from __future__ import annotations

from pathlib import Path

from agents.base import Agent
from search import store
from search.indexer import index_bid


class IngestionAgent(Agent):
    name = "ingestion"

    def run(self, force: bool = False) -> dict:
        with self.step("index_bid", bid_dir=self.state.bid_dir) as out:
            stats = index_bid(Path(self.state.bid_dir), force=force)
            self.state.has_addenda = store.count(
                store.build_filter(None, bid_id=self.state.bid_id, doc_type="addendum")) > 0
            out.update(indexed=stats["indexed"], skipped=stats["skipped"], failed=stats["failed"],
                       chunks=stats["chunks"], has_addenda=self.state.has_addenda)
        return stats
