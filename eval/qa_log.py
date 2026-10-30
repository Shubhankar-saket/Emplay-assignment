"""Q&A log deliverable: runs a fixed question set through Q&A mode -> outputs/qa_log.md (+ .json).

Run: python -m eval.qa_log
"""
from __future__ import annotations

import json
from pathlib import Path

from agents.qa import answer_question
from core.logging import setup_logging
from core.models import SearchFilters

QUESTIONS: list[tuple[str, str | None]] = [  # (question, bid filter or None for all bids)
    ("What is the deadline after all addendums?", "Bid1"),
    ("When and how is the pre-proposal meeting held?", "Bid1"),
    ("Must devices be etched or asset tagged, and did any addendum change that?", "Bid1"),
    ("Can other school districts purchase from this contract?", "Bid1"),
    ("How long is the contract term?", "Bid1"),
    ("Is a bid bond required?", "Bid2"),
    ("How soon after award must the laptops be delivered?", "Bid2"),
    ("What processor, memory and storage does the laptop have?", "Bid2"),
    ("Who is the procurement contact and what is their email?", "Bid2"),
    ("How must proposals be submitted?", "Bid2"),
    ("Compare the warranties of both bids.", None),
    ("Which bid closes first, and on what date?", None),
]


def main() -> None:
    setup_logging()
    rows = []
    for q, bid in QUESTIONS:
        ans = answer_question(q, SearchFilters(bid_id=bid) if bid else None)
        rows.append({"bid": bid or "all", **ans.model_dump(exclude={"trace"}),
                     "agent_steps": [f"{s.agent}.{s.action}" for s in ans.trace]})
        print(f"[{'found' if ans.found else 'not found'}] {q}")
    out = Path("outputs")
    out.mkdir(exist_ok=True)
    (out / "qa_log.json").write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")
    lines = ["# Q&A log", "", f"{len(rows)} questions run through Q&A mode (`python -m eval.qa_log`). "
             "Each answer comes only from retrieved passages; citations are file + page.", ""]
    for i, r in enumerate(rows, 1):
        lines += [f"## {i}. {r['question']}", "", f"**Scope:** {r['bid']}", "", r["answer"], ""]
        lines += [f"- {c['file']} p.{c['page']}" for c in r["citations"]] or ["- (no citation)"]
        lines.append("")
    (out / "qa_log.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
