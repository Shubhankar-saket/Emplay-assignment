"""Validator / critic agent: checks each field against the text it cites (code checks first, then an LLM)."""
from __future__ import annotations

import json

from agents.base import Agent
from agents.fields import SPECS
from agents.llm import LLMError, get_llm
from agents.state import ValidationVerdict

SYSTEM = """You are a strict reviewer of fields extracted from procurement bid documents.
For each field you get the value, the source text it cites, and other retrieved text.
Accept a value only if the cited text supports it and it answers the field's definition.
Accept partially complete values when what is given is correct (e.g. a contact without a name); say what
is missing in the reason. Accept "Not found in documents" only if none of the text shown contains the information.
Values that faithfully copy an error in the source (typo, mislabelled heading) are acceptable.
Reject with a specific reason (what is wrong or missing, where to look) so it can be retried.
Reply as JSON: {"verdicts": {"<field key>": {"accepted": true|false, "reason": "..."}}}"""
BATCH = 4  # fields per call (free-tier token-per-minute limits)
NO_VERDICT = "no verdict returned"


class ValidatorAgent(Agent):
    name = "validator"

    def run(self, keys: list[str]) -> list[ValidationVerdict]:
        verdicts: dict[str, ValidationVerdict] = {}
        to_llm = []
        for k in keys:
            v = self.state.fields[k]
            if not v.is_not_found and not v.sources:
                verdicts[k] = ValidationVerdict(field=k, accepted=False, reason="value has no cited source")
            else:
                to_llm.append(k)
        for i in range(0, len(to_llm), BATCH):
            verdicts.update(self._llm_verdicts(to_llm[i:i + BATCH]))
        for k in [k for k in to_llm if verdicts[k].reason == NO_VERDICT]:  # dropped from a batch reply
            verdicts.update(self._llm_verdicts([k]))
        return [verdicts[k] for k in keys]

    def _llm_verdicts(self, keys: list[str]) -> dict[str, ValidationVerdict]:
        with self.step("validate", fields=keys) as out:
            try:
                data = get_llm().chat_json(SYSTEM, self._prompt(keys), fast=True).get("verdicts") or {}
            except LLMError as exc:  # leave fields rejected; the retry loop / final status handles them
                out.update(status="error", error=str(exc)[:200])
                data = {}
            verdicts = {}
            for k in keys:
                raw = data.get(k) if isinstance(data.get(k), dict) else {}
                verdicts[k] = ValidationVerdict(field=k, accepted=bool(raw.get("accepted")),
                                                reason=str(raw.get("reason") or NO_VERDICT))
            out["verdicts"] = {k: v.accepted for k, v in verdicts.items()}
        return verdicts

    def _prompt(self, keys: list[str]) -> str:
        parts = []
        for k in keys:
            v = self.state.fields[k]
            cited = {(c.file, c.page) for c in v.sources}
            texts = []
            for h in self.state.evidence.get(k, []):
                c = h.chunk
                tag = "CITED" if (c.file_name, c.page_start) in cited else "other"
                texts.append(f"  ({tag}) {c.file_name} p.{c.page_start}: {c.text[:700]}")
            parts.append(f'### "{k}" ({SPECS[k].label}): {SPECS[k].description}\n'
                         f"Value: {json.dumps(v.value, ensure_ascii=False)}\nNotes: {v.notes}\n"
                         f"Text:\n" + "\n".join(sorted(texts, key=lambda s: "(CITED)" not in s)[:4]))
        return "\n\n".join(parts)
