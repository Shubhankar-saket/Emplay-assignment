"""Orchestrator: plans the run and routes work between agents.

Extraction mode:
  ingestion -> [retrieval -> extraction] x 3 groups in parallel -> addendum reconciliation
  -> validation (parallel) -> rejected fields go back to retrieval (rewritten queries) + extraction,
     at most AGENT_MAX_RETRIES times -> report
"""
from __future__ import annotations

from typing import TYPE_CHECKING

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from agents.base import Agent
from agents.extraction import ExtractionAgent
from agents.fields import GROUPS, SPECS
from agents.ingestion_agent import IngestionAgent
from agents.reconciliation import ReconciliationAgent
from agents.report import ReportAgent
from agents.retrieval import RetrievalAgent
from agents.state import BidState, ExtractionRequest, FieldValue
from agents.validator import ValidatorAgent
from core.config import get_settings
from core.models import SearchFilters
from search import store

if TYPE_CHECKING:
    from agents.qa import Answer


class Orchestrator(Agent):
    name = "orchestrator"

    def __init__(self, bid_dir: Path | None = None, out_dir: Path = Path("outputs"), bid_id: str | None = None) -> None:
        super().__init__(BidState(bid_id=bid_id or (bid_dir.name if bid_dir else "all"), bid_dir=str(bid_dir or "")))
        self.out_dir = out_dir
        self.settings = get_settings()
        self.pool = ThreadPoolExecutor(max_workers=self.settings.llm_concurrency)

    @classmethod
    def ask(cls, question: str, filters: SearchFilters | None = None, k: int = 6) -> "Answer":
        """Q&A mode: plan (which bids), route to the Q&A agent, attach the trace to the answer."""
        from agents.qa import QAAgent
        orch = cls(bid_id=filters.bid_id if filters and filters.bid_id else None)
        orch.pool.shutdown()
        with orch.step("plan", mode="qa") as out:
            bids = [filters.bid_id] if filters and filters.bid_id else store.list_bids()
            out["plan"] = [f"retrieve:{b}(+addenda)" for b in bids] + ["answer_with_citations"]
        answer = QAAgent(orch.state).run(question, filters, k)
        answer.trace = orch.state.trace
        return answer

    def run_extraction(self, force_index: bool = False) -> dict:
        IngestionAgent(self.state).run(force=force_index)
        with self.step("plan") as out:
            out["plan"] = (["retrieve+extract:" + g for g in GROUPS]
                           + (["reconcile_addenda"] if self.state.has_addenda else ["normalize_dates"])
                           + ["validate", f"retry_rejected(max={self.settings.agent_max_retries})", "report"])
        list(self.pool.map(self._retrieve_and_extract, GROUPS))
        ReconciliationAgent(self.state).run()
        self._validate_with_retries()
        record = ReportAgent(self.state).run(self.out_dir)
        self.pool.shutdown()
        return record

    def _retrieve_and_extract(self, group: str, keys: list[str] | None = None,
                              feedback: dict[str, str] | None = None) -> None:
        keys = keys or GROUPS[group]
        retrieval = RetrievalAgent(self.state)
        for k in keys:
            queries = retrieval.rewrite(k, feedback[k]) if feedback and k in feedback else None
            retrieval.run(retrieval.request_for(k, queries))
        ExtractionAgent(self.state, group).run(ExtractionRequest(
            fields=keys, evidence={k: self.state.evidence[k] for k in keys}, feedback=feedback or {}))

    def _validate_with_retries(self) -> None:
        pending = [k for keys in GROUPS.values() for k in keys]
        for attempt in range(self.settings.agent_max_retries + 1):
            batches = [[k for k in keys if k in pending] for keys in GROUPS.values()]
            verdicts = [v for vs in self.pool.map(lambda b: ValidatorAgent(self.state).run(b) if b else [], batches)
                        for v in vs]
            rejected = {v.field: v.reason for v in verdicts if not v.accepted}
            for v in verdicts:
                if v.accepted:
                    self.state.fields[v.field].status = "not_found" if self.state.fields[v.field].is_not_found \
                        else "accepted"
            if not rejected or attempt == self.settings.agent_max_retries:
                break
            with self.step("route_rejected", attempt=attempt + 1, fields=rejected):
                self.state.counts.retried += len(rejected)
                by_group: dict[str, list[str]] = {}
                for k in rejected:
                    by_group.setdefault(SPECS[k].group, []).append(k)
                list(self.pool.map(lambda g: self._retrieve_and_extract(g, by_group[g], rejected), by_group))
                for k in rejected:  # extraction resets attempts to 1; keep the running count
                    self.state.fields[k].attempts = attempt + 2
            pending = list(rejected)
        for k in pending:
            val = self.state.fields[k]
            if val.status == "pending":
                val.status = "rejected"
                val.confidence = min(val.confidence, 0.3)
                val.notes = (val.notes + " | failed validation after retries").strip(" |")
        c = self.state.counts
        statuses = [self.state.fields.get(k, FieldValue()).status for keys in GROUPS.values() for k in keys]
        c.accepted, c.not_found, c.rejected = (statuses.count("accepted"), statuses.count("not_found"),
                                               statuses.count("rejected"))
