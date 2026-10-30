"""Shared state and the Pydantic messages agents exchange. Every agent step is appended to the trace."""
from __future__ import annotations

import threading
import time
from typing import Any, Literal

from pydantic import BaseModel, Field, PrivateAttr

from core.models import Citation, DocType, SearchHit

NOT_FOUND = "Not found in documents"
FieldStatus = Literal["pending", "accepted", "rejected", "not_found"]


class FieldValue(BaseModel):
    value: Any = NOT_FOUND
    sources: list[Citation] = Field(default_factory=list)
    confidence: float = 0.0
    notes: str = ""
    normalized: str | None = None  # e.g. ISO-8601 datetime with UTC offset for date fields
    original_value: Any = None  # pre-addendum value when an addendum overrode the original documents
    status: FieldStatus = "pending"
    attempts: int = 0

    @property
    def is_not_found(self) -> bool:
        return isinstance(self.value, str) and self.value.strip().lower().startswith(NOT_FOUND.lower())


class AddendumChange(BaseModel):
    field: str
    old_value: Any
    new_value: Any
    addendum_number: int | None
    source: Citation
    note: str = ""


class ValidationCounts(BaseModel):
    accepted: int = 0
    rejected: int = 0
    not_found: int = 0
    retried: int = 0


class AgentStep(BaseModel):
    ts: str
    agent: str
    action: str
    status: Literal["ok", "error", "skipped"] = "ok"
    duration_ms: int | None = None
    detail: dict[str, Any] = Field(default_factory=dict)


# ---- messages ---------------------------------------------------------------------------

class RetrievalRequest(BaseModel):
    field: str
    bid_id: str
    queries: list[str]
    doc_type_passes: list[list[DocType] | None] = Field(default_factory=lambda: [None])
    k: int = 4


class RetrievalResult(BaseModel):
    field: str
    queries: list[str]
    hits: list[SearchHit]


class ExtractionRequest(BaseModel):
    fields: list[str]
    evidence: dict[str, list[SearchHit]]
    feedback: dict[str, str] = Field(default_factory=dict)  # validator reasons on retries


class ExtractionResult(BaseModel):
    values: dict[str, FieldValue]


class ValidationVerdict(BaseModel):
    field: str
    accepted: bool
    reason: str


# ---- state ------------------------------------------------------------------------------

class BidState(BaseModel):
    bid_id: str
    bid_dir: str
    fields: dict[str, FieldValue] = Field(default_factory=dict)
    evidence: dict[str, list[SearchHit]] = Field(default_factory=dict)
    queries: dict[str, list[str]] = Field(default_factory=dict)
    changes: list[AddendumChange] = Field(default_factory=list)
    counts: ValidationCounts = Field(default_factory=ValidationCounts)
    has_addenda: bool = False
    trace: list[AgentStep] = Field(default_factory=list)
    _lock: threading.Lock = PrivateAttr(default_factory=threading.Lock)

    def add_step(self, step: AgentStep) -> None:
        with self._lock:
            self.trace.append(step)

    def set_field(self, key: str, value: FieldValue) -> None:
        with self._lock:
            self.fields[key] = value

    def set_evidence(self, key: str, hits: list[SearchHit], queries: list[str]) -> None:
        with self._lock:
            self.evidence[key] = hits
            self.queries[key] = queries


def now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S")
