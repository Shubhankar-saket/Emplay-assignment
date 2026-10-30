"""Decision agents built on the validated extraction records (outputs/<bid>.json):

- GoNoGoAgent: GO / NO-GO / REVIEW for a bidder profile, with reasons tied to cited fields; the deadline
  status is computed in code from the normalised due date.
- ComparisonAgent: field-by-field comparison of two or more bids.

Both work from extracted values only (each already has file + page citations), never from raw documents.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field

from agents.base import Agent
from agents.llm import LLMError, get_llm
from agents.state import NOT_FOUND, BidState
from core.config import get_settings

VALUE_CHARS = 400  # per field value in prompts (free-tier token limits)


class Reason(BaseModel):
    field: str
    assessment: Literal["pro", "con", "risk"]
    detail: str


class GoNoGo(BaseModel):
    bid_id: str
    verdict: Literal["GO", "NO-GO", "REVIEW"]
    score: int = Field(ge=0, le=100)
    deadline: dict[str, Any]
    reasons: list[Reason] = Field(default_factory=list)
    missing_info: list[str] = Field(default_factory=list)
    profile: str = ""


class FieldComparison(BaseModel):
    field: str
    differs: bool
    comparison: str


class Comparison(BaseModel):
    bids: list[str]
    overall: str
    fields: list[FieldComparison] = Field(default_factory=list)


GO_SYSTEM = """You advise a bidder whether to respond to a public procurement solicitation.
You get the bidder's profile and fields extracted from the solicitation (validated, with citations).
Judge fit only from these: requirements the bidder meets (pro), cannot meet (con) or that are uncertain or
costly (risk). "Not found in documents" is missing information, not a requirement.
Verdict: GO if no blocking con, NO-GO if any requirement clearly cannot be met, otherwise REVIEW.
Score 0-100 = overall fit. Ignore whether the deadline has passed; that is handled separately.
Reply as JSON: {"verdict": "GO|NO-GO|REVIEW", "score": <int>, "reasons": [{"field": "<field label>",
"assessment": "pro|con|risk", "detail": "<one sentence>"}], "missing_info": ["<what to ask the buyer>"]}"""

COMPARE_SYSTEM = """You compare procurement solicitations field by field using only the extracted values given
(each bid's fields were validated against its documents). For each field say concisely how the bids differ or
that they match; "Not found in documents" means that bid does not state it. Then give a 2-3 sentence overall
comparison (scope, timing, commercial terms, product). Reply as JSON:
{"overall": "...", "fields": {"<field label>": {"differs": true|false, "comparison": "..."}}}"""


def load_record(bid_id: str, out_dir: Path = Path("outputs")) -> dict:
    path = out_dir / f"{bid_id}.json"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found: run `python main.py --bid <folder>` first")
    return json.loads(path.read_text(encoding="utf-8"))


def load_profile(path: Path | None = None) -> dict:
    path = path or get_settings().bidder_profile_path
    return {k: v for k, v in json.loads(Path(path).read_text(encoding="utf-8")).items() if not k.startswith("_")}


def _compact(record: dict) -> dict[str, Any]:
    out = {}
    for label, f in record["fields"].items():
        v = f["value"] if isinstance(f["value"], str) else json.dumps(f["value"], ensure_ascii=False)
        out[label] = v[:VALUE_CHARS]
    return out


def deadline_status(record: dict, as_of: datetime | None = None) -> dict[str, Any]:
    due = record["fields"].get("Due Date", {})
    iso = due.get("normalized")
    if not iso:
        return {"value": due.get("value", NOT_FOUND), "status": "unknown"}
    dt = datetime.fromisoformat(iso)
    now = as_of or datetime.now(timezone.utc)
    if dt.tzinfo is None:  # no zone in the documents: compare dates only
        dt, now = dt.replace(tzinfo=timezone.utc), now.replace(tzinfo=timezone.utc)
    return {"value": due.get("value"), "normalized": iso, "status": "closed" if dt < now else "open",
            "days_left": (dt - now).days}


class GoNoGoAgent(Agent):
    name = "go_no_go"

    def run(self, record: dict, profile: dict | None = None) -> GoNoGo:
        profile = profile or load_profile()
        deadline = deadline_status(record)
        with self.step("assess", bid_id=record["bid_id"], deadline=deadline["status"]) as out:
            try:
                data = get_llm().chat_json(GO_SYSTEM, json.dumps(
                    {"bidder_profile": profile, "solicitation": _compact(record),
                     "addendum_changes": [c["field"] for c in record.get("addendum_changes", [])]},
                    ensure_ascii=False))
            except LLMError as exc:
                out.update(status="error", error=str(exc)[:200])
                data = {"verdict": "REVIEW", "score": 0, "missing_info": [f"assessment failed: {str(exc)[:120]}"]}
            reasons = [Reason(**r) for r in data.get("reasons") or []
                       if isinstance(r, dict) and r.get("assessment") in ("pro", "con", "risk")]
            verdict = data.get("verdict") if data.get("verdict") in ("GO", "NO-GO", "REVIEW") else "REVIEW"
            result = GoNoGo(bid_id=record["bid_id"], verdict=verdict,
                            score=max(0, min(100, int(data.get("score") or 0))), deadline=deadline,
                            reasons=reasons, missing_info=[str(m) for m in data.get("missing_info") or []],
                            profile=profile.get("company", ""))
            out.update(verdict=result.verdict, score=result.score)
        return result


class ComparisonAgent(Agent):
    name = "comparison"

    def run(self, records: list[dict]) -> Comparison:
        bids = [r["bid_id"] for r in records]
        with self.step("compare", bids=bids) as out:
            payload = {r["bid_id"]: {k: v for k, v in _compact(r).items() if k != "Summary"} for r in records}
            try:
                data = get_llm().chat_json(COMPARE_SYSTEM, json.dumps(payload, ensure_ascii=False))
            except LLMError as exc:
                out.update(status="error", error=str(exc)[:200])
                data = {"overall": f"Comparison unavailable: {str(exc)[:150]}", "fields": {}}
            fields = [FieldComparison(field=k, differs=bool(v.get("differs")), comparison=str(v.get("comparison", "")))
                      for k, v in (data.get("fields") or {}).items() if isinstance(v, dict)]
            out["differing_fields"] = [f.field for f in fields if f.differs]
        return Comparison(bids=bids, overall=str(data.get("overall") or ""), fields=fields)


# ---- standalone entry points (work from saved outputs) -----------------------------------

def go_no_go_markdown(g: GoNoGo) -> str:
    d = g.deadline
    lines = [f"## Go / no-go: {g.verdict} (fit {g.score}/100)", "",
             f"Bidder profile: {g.profile}. Deadline: {d.get('value')} ({d['status']}"
             + (f", {d['days_left']} days" if "days_left" in d else "") + ").", ""]
    lines += [f"- **{r.assessment.upper()}** ({r.field}): {r.detail}" for r in g.reasons]
    if g.missing_info:
        lines += ["", "Ask the buyer:"] + [f"- {m}" for m in g.missing_info]
    return "\n".join(lines) + "\n"


def run_go_no_go(bid_id: str, out_dir: Path = Path("outputs")) -> GoNoGo:
    """Assess a saved record and write the verdict back into outputs/<bid>.json and its summary."""
    record = load_record(bid_id, out_dir)
    state = BidState(bid_id=bid_id, bid_dir="")
    result = GoNoGoAgent(state).run(record)
    record["go_no_go"] = result.model_dump()
    (out_dir / f"{bid_id}.json").write_text(json.dumps(record, indent=2, ensure_ascii=False), encoding="utf-8")
    summary = out_dir / f"{bid_id}_summary.md"
    if summary.exists():
        text = summary.read_text(encoding="utf-8").split("\n## Go / no-go")[0].rstrip()
        summary.write_text(text + "\n\n" + go_no_go_markdown(result), encoding="utf-8")
    return result


def run_comparison(bid_ids: list[str], out_dir: Path = Path("outputs")) -> Comparison:
    records = [load_record(b, out_dir) for b in bid_ids]
    result = ComparisonAgent(BidState(bid_id="+".join(bid_ids), bid_dir="")).run(records)
    stem = out_dir / f"comparison_{'_vs_'.join(bid_ids)}"
    stem.with_suffix(".json").write_text(json.dumps(result.model_dump(), indent=2, ensure_ascii=False),
                                         encoding="utf-8")
    lines = [f"# Comparison: {' vs '.join(bid_ids)}", "", result.overall, "", "| Field | Differs | Comparison |",
             "|---|---|---|"]
    lines += [f"| {f.field} | {'yes' if f.differs else 'no'} | {f.comparison.replace('|', '/')} |" for f in result.fields]
    stem.with_suffix(".md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return result
