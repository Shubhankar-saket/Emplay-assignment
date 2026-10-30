"""Addendum reconciliation agent: applies the latest addendum changes to extracted fields, keeps a change
log, and normalises date fields (time zones) in code."""
from __future__ import annotations

import json

from agents.base import Agent, number_sources
from agents.fields import DATE_FIELDS, SPECS
from agents.llm import LLMError, get_llm
from agents.state import AddendumChange, FieldValue
from agents.timeutil import parse_datetime
from core.config import get_settings
from core.models import Chunk, SearchFilters, SearchHit
from search import store

SYSTEM = """You reconcile procurement bid addenda with values extracted from the original documents.
Given the current field values and the full text of every addendum (numbered sources), list each field
whose value an addendum changes, replaces or adds to, versus the ORIGINAL documents. List a change even
when the current value already reflects the addendum, and give the original value it replaced.
Ignore addendum content that does not change a field.
When several addenda change the same field, the highest addendum number wins.
Reply as JSON: {"changes": [{"field": "<field key>", "old_value": <original value or null>, "new_value": ...,
"source": <n>, "note": "<what changed>"}]}
Reply {"changes": []} when nothing changes."""


class ReconciliationAgent(Agent):
    name = "reconciliation"

    def run(self) -> list[AddendumChange]:
        if self.state.has_addenda:
            self._apply_addenda()
        else:
            with self.step("apply_addenda") as out:
                out.update(status="skipped", reason="bid has no addenda")
        self._normalize_dates()
        return self.state.changes

    def _addendum_hits(self) -> list[SearchHit]:
        s = get_settings()
        flt = store.build_filter(SearchFilters(bid_id=self.state.bid_id, doc_types=["addendum"]))
        points, _ = store.get_client().scroll(s.collection, scroll_filter=flt, limit=60, with_payload=True)
        chunks = sorted((Chunk.model_validate(p.payload) for p in points),
                        key=lambda c: (-(c.addendum_number or 0), c.page_start, c.chunk_id))  # latest first: it wins, and survives the cap
        return [SearchHit(chunk=c, score=0.0, rank=i + 1) for i, c in enumerate(chunks)]

    def _apply_addenda(self) -> None:
        hits = self._addendum_hits()
        current = {k: v.value for k, v in self.state.fields.items()}
        with self.step("apply_addenda", addendum_chunks=len(hits)) as out:
            sources, ordered = number_sources(hits[:12], max_chars=800)
            fields_desc = "\n".join(f'- "{k}": {SPECS[k].description}' for k in current)
            data = get_llm().chat_json(SYSTEM, (
                f"Fields:\n{fields_desc}\n\nCurrent values:\n{json.dumps(current, ensure_ascii=False)}\n\n"
                f"Addenda:\n{sources}"))
            applied = []
            for ch in data.get("changes") or []:
                key, n = ch.get("field"), ch.get("source")
                if key not in self.state.fields or not isinstance(n, int) or not 1 <= n <= len(ordered):
                    continue
                src = ordered[n - 1]
                old = self.state.fields[key]
                already = _same(old.value, ch.get("new_value")) or src.citation in old.sources
                self.state.changes.append(AddendumChange(
                    field=key, old_value=(ch.get("old_value") or old.original_value) if already else old.value,
                    new_value=ch.get("new_value"),
                    addendum_number=src.chunk.addendum_number, source=src.citation, note=str(ch.get("note") or "")))
                if already:  # extraction already applied this addendum; only the change log is updated
                    applied.append(key)
                    continue
                self.state.set_field(key, FieldValue(
                    value=ch.get("new_value"), sources=[src.citation] + old.sources,
                    confidence=max(old.confidence, 0.85), attempts=old.attempts,
                    notes=f"Updated by Addendum {src.chunk.addendum_number}: {ch.get('note', '')} "
                          f"(original: {json.dumps(old.value, ensure_ascii=False)[:150]})"))
                applied.append(key)
            out["changed_fields"] = applied

    def _normalize_dates(self) -> None:
        with self.step("normalize_dates") as out:
            for key in DATE_FIELDS:
                val = self.state.fields.get(key)
                if not val or val.is_not_found:
                    continue
                iso, note = parse_datetime(val.value if isinstance(val.value, str) else json.dumps(val.value))
                if iso:
                    val.normalized = iso
                    if note:
                        val.notes = (val.notes + " | " + note).strip(" |")
                    out[key] = iso


def _same(a, b) -> bool:
    norm = lambda v: " ".join(json.dumps(v, ensure_ascii=False).lower().split()) if not isinstance(v, str)         else " ".join(v.lower().split())
    return norm(a) == norm(b)
