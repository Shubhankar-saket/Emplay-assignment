"""Report agent: writes the summary field and the final outputs (JSON record, Markdown summary, trace)."""
from __future__ import annotations

import json
import time
from pathlib import Path

from agents.base import Agent
from agents.fields import FIELDS
from agents.llm import LLMError, get_llm
from agents.state import FieldValue

SYSTEM = """Write a 3-5 sentence factual summary of a procurement solicitation from the extracted fields
given. Use only those values; skip any field marked "Not found in documents". Mention addendum changes.
Reply as JSON: {"summary": "..."}"""


class ReportAgent(Agent):
    name = "report"

    def run(self, out_dir: Path = Path("outputs")) -> dict:
        self._summary()
        record = self.record()
        out_dir.mkdir(parents=True, exist_ok=True)
        bid = self.state.bid_id
        with self.step("write_outputs") as out:
            (out_dir / f"{bid}.json").write_text(json.dumps(record, indent=2, ensure_ascii=False), encoding="utf-8")
            (out_dir / f"{bid}_summary.md").write_text(self.markdown(record), encoding="utf-8")
            out["files"] = [f"{bid}.json", f"{bid}_summary.md", f"{bid}_trace.json"]
        (out_dir / f"{bid}_trace.json").write_text(
            json.dumps([s.model_dump() for s in self.state.trace], indent=1, ensure_ascii=False), encoding="utf-8")
        return record

    def _summary(self) -> None:
        facts = {f.label: v.value for f in FIELDS if (v := self.state.fields.get(f.key)) and not v.is_not_found}
        changes = [c.model_dump(include={"field", "new_value", "addendum_number"}) for c in self.state.changes]
        with self.step("summarize") as out:
            try:
                data = get_llm().chat_json(SYSTEM, json.dumps({"fields": facts, "addendum_changes": changes},
                                                              ensure_ascii=False)[:12000])
            except LLMError as exc:
                out.update(status="error", error=str(exc)[:200])
                data = {}
            cited = []
            for key in ("title", "due_date", "product", "company"):
                for c in (self.state.fields.get(key) or FieldValue()).sources[:1]:
                    if c not in cited:
                        cited.append(c)
            self.state.set_field("summary", FieldValue(
                value=data.get("summary") or "", sources=cited, confidence=0.7, status="accepted",
                notes="generated from the other extracted fields"))
            out["chars"] = len(data.get("summary") or "")

    def record(self) -> dict:
        fields = {}
        for f in FIELDS:
            v = self.state.fields.get(f.key) or FieldValue()
            fields[f.label] = {
                "value": v.value, "sources": [c.model_dump(exclude_none=True) for c in v.sources],
                "confidence": round(v.confidence, 2), "notes": v.notes, "status": v.status,
                **({"normalized": v.normalized} if v.normalized else {}),
            }
        return {"bid_id": self.state.bid_id, "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
                "fields": fields, "addendum_changes": [c.model_dump() for c in self.state.changes],
                "validation": self.state.counts.model_dump(), "llm_usage": get_llm().usage}

    @staticmethod
    def markdown(record: dict) -> str:
        v = record["validation"]
        lines = [f"# {record['bid_id']}: extraction summary", "", record["fields"]["Summary"]["value"], "",
                 f"Validation: {v['accepted']} accepted, {v['rejected']} rejected, {v['not_found']} not found, "
                 f"{v['retried']} retries.", "", "| Field | Value | Source | Confidence |", "|---|---|---|---|"]
        for label, f in record["fields"].items():
            if label == "Summary":
                continue
            val = f["value"] if isinstance(f["value"], str) else json.dumps(f["value"], ensure_ascii=False)
            src = "; ".join(f"{s['file']} p.{s['page']}" for s in f["sources"][:2])
            lines.append(f"| {label} | {val[:200].replace('|', '/').replace(chr(10), ' ')} | {src} | {f['confidence']} |")
        if record["addendum_changes"]:
            lines += ["", "## Addendum changes", ""]
            lines += [f"- {c['field']}: {c['old_value']!r} -> {c['new_value']!r} (Addendum {c['addendum_number']}, "
                      f"{c['source']['file']} p.{c['source']['page']})" for c in record["addendum_changes"]]
        return "\n".join(lines) + "\n"
