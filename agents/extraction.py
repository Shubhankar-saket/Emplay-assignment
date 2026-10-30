"""Extraction agents (dates / commercial / product): fill fields from retrieved evidence only."""
from __future__ import annotations

import json

from agents.base import Agent, number_sources
from agents.fields import SPECS
from agents.llm import LLMError, get_llm
from agents.state import NOT_FOUND, ExtractionRequest, ExtractionResult, FieldValue

BATCH = 4  # fields per LLM call: keeps prompts small for free-tier token limits
HITS_PER_FIELD = 5
SOURCE_CHARS = 700

SYSTEM = f"""You extract fields from public procurement bid documents.
Rules:
- Use ONLY the numbered sources. Never guess or use outside knowledge.
- Cite the source numbers that state each value.
- If the sources do not state a field, the value is exactly "{NOT_FOUND}" with no sources and confidence 0.
- Addenda override the original RFP: when they conflict, use the latest addendum, mention the change in notes and
  put the value it replaced in "original_value".
- Keep values concise. Keep dates exactly as written, including time and time zone. Lists stay JSON lists.
- If a source contains an obvious error (typo, mislabelled section), copy the value faithfully and flag it in notes.
- confidence: 0.9+ when stated explicitly, 0.5-0.8 when inferred or partial.
Reply as JSON: {{"fields": {{"<field key>": {{"value": ..., "sources": [<n>], "confidence": <0-1>, "notes": "...",
"original_value": <only when an addendum replaced it>}}}}}}"""


class ExtractionAgent(Agent):
    def __init__(self, state, group: str) -> None:
        self.name = f"extraction.{group}"
        super().__init__(state)
        self.group = group

    def run(self, req: ExtractionRequest) -> ExtractionResult:
        values: dict[str, FieldValue] = {}
        for i in range(0, len(req.fields), BATCH):
            values.update(self._extract(req.fields[i:i + BATCH], req))
        for key, val in values.items():
            self.state.set_field(key, val)
        return ExtractionResult(values=values)

    def _extract(self, keys: list[str], req: ExtractionRequest) -> dict[str, FieldValue]:
        sources, ordered = number_sources([h for k in keys for h in req.evidence.get(k, [])[:HITS_PER_FIELD]],
                                          max_chars=SOURCE_CHARS)
        lines = []
        for k in keys:
            line = f'- "{k}" ({SPECS[k].label}): {SPECS[k].description}'
            if k in req.feedback:
                line += f"\n  A reviewer rejected the previous answer: {req.feedback[k]}"
            lines.append(line)
        user = f"Bid: {self.state.bid_id}\nFields:\n" + "\n".join(lines) + f"\n\nSources:\n{sources}"
        with self.step("extract", fields=keys, sources=len(ordered), retry=bool(req.feedback)) as out:
            try:
                data = get_llm().chat_json(SYSTEM, user).get("fields") or {}
            except LLMError as exc:  # fields stay "not found" with a note; the validator retry loop can recover
                out.update(status="error", error=str(exc)[:200])
                data = {k: {"value": NOT_FOUND, "notes": f"extraction failed: {str(exc)[:120]}"} for k in keys}
            values = {k: self._to_value(data.get(k), ordered) for k in keys}
            out["values"] = {k: _short(v.value) for k, v in values.items()}
        return values

    @staticmethod
    def _to_value(raw, ordered) -> FieldValue:
        if not isinstance(raw, dict):
            return FieldValue(notes="extractor returned no value for this field")
        nums = [n for n in raw.get("sources") or [] if isinstance(n, int) and 1 <= n <= len(ordered)]
        try:
            conf = max(0.0, min(1.0, float(raw.get("confidence") or 0)))
        except (TypeError, ValueError):
            conf = 0.5
        val = FieldValue(value=raw.get("value", NOT_FOUND), sources=[ordered[n - 1].citation for n in nums],
                         confidence=conf, notes=str(raw.get("notes") or ""), original_value=raw.get("original_value"))
        if val.is_not_found:
            val.value, val.sources, val.confidence = NOT_FOUND, [], 0.0
        elif not val.sources:  # an uncited value is not trusted
            val.confidence = min(val.confidence, 0.3)
            val.notes = (val.notes + " | no source cited").strip(" |")
        val.attempts = 1
        return val


def _short(v, n: int = 120) -> str:
    s = v if isinstance(v, str) else json.dumps(v, ensure_ascii=False)
    return s[:n]
