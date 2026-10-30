"""Agent base class: every unit of work runs inside step(), which times it and records it in the trace."""
from __future__ import annotations

import time
from contextlib import contextmanager
from typing import Any, Iterator

from agents.state import AgentStep, BidState, now
from core.logging import get_logger, log_event
from core.models import SearchHit


class Agent:
    name = "agent"

    def __init__(self, state: BidState) -> None:
        self.state = state
        self.log = get_logger(f"agents.{self.name}")

    @contextmanager
    def step(self, action: str, **detail: Any) -> Iterator[dict[str, Any]]:
        """Yields a dict the caller fills with outputs; status/duration are recorded even on errors."""
        out: dict[str, Any] = {}
        t0, status = time.time(), "ok"
        try:
            yield out
        except Exception as exc:
            status, out["error"] = "error", repr(exc)[:300]
            raise
        finally:
            rec = AgentStep(ts=now(), agent=self.name, action=action, status=out.pop("status", status),
                            duration_ms=int((time.time() - t0) * 1000), detail={**detail, **out})
            self.state.add_step(rec)
            log_event(self.log, action, bid_id=self.state.bid_id, **rec.model_dump(exclude={"ts"}))


def number_sources(hits: list[SearchHit], max_chars: int = 900) -> tuple[str, list[SearchHit]]:
    """De-duplicate hits by chunk and render them as numbered [n] source blocks for a prompt."""
    unique: dict[str, SearchHit] = {}
    for h in hits:
        unique.setdefault(h.chunk.chunk_id, h)
    ordered = list(unique.values())
    blocks = []
    for i, h in enumerate(ordered, 1):
        c = h.chunk
        label = c.doc_type + (f" {c.addendum_number}" if c.addendum_number else "")
        blocks.append(f"[{i}] {label} | {c.file_name} p.{c.page_start}\n{c.text[:max_chars]}")
    return "\n\n".join(blocks), ordered
