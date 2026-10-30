"""Retrieval evaluation: Recall@k and MRR across search configurations -> eval/report.md.

Gold labels are (file-name substring, page) pairs; page null = any page of that file.
"BidNet" matches the saved .html bid pages. Run: python -m eval.run_eval
"""
from __future__ import annotations

import json
import time
from pathlib import Path

from core.logging import setup_logging
from core.models import SearchFilters
from search.engine import search

HERE = Path(__file__).parent
KS = (1, 3, 5)
DEPTH = 10
CONFIGS = [  # (label, mode, query expansion)
    ("vector only", "dense", False),
    ("BM25 only", "bm25", True),
    ("hybrid (RRF)", "hybrid", True),
    ("hybrid + rerank, no expansion", "hybrid_rerank", False),
    ("hybrid + rerank", "hybrid_rerank", True),
]


def _is_gold(hit, gold) -> bool:
    c = hit.chunk
    return any(sub in c.file_name and (page is None or c.page_start <= page <= c.page_end) for sub, page in gold)


def evaluate(questions: list[dict]) -> tuple[list[dict], dict]:
    summary, per_q = [], {}
    for label, mode, expand in CONFIGS:
        recalls, rr, t0 = {k: 0 for k in KS}, 0.0, time.time()
        for q in questions:
            hits = search(q["question"], SearchFilters(bid_id=q["bid_id"]), k=DEPTH, mode=mode, expand=expand)
            rank = next((h.rank for h in hits if _is_gold(h, q["gold"])), None)
            per_q.setdefault(q["id"], {})[label] = rank
            for k in KS:
                recalls[k] += rank is not None and rank <= k
            rr += 1 / rank if rank else 0
        n = len(questions)
        summary.append({"config": label, **{f"R@{k}": recalls[k] / n for k in KS}, "MRR": rr / n,
                        "ms/query": (time.time() - t0) * 1000 / n})
    return summary, per_q


def write_report(questions, summary, per_q) -> str:
    labels = [c[0] for c in CONFIGS]
    lines = ["# Retrieval evaluation", "",
             f"{len(questions)} questions over Bid1 and Bid2 (`eval/questions.json`), each with gold (file, page) "
             f"passages. Bid filter applied where the question names a bid. MRR is computed over the top {DEPTH}.", "",
             "| Config | " + " | ".join(f"R@{k}" for k in KS) + " | MRR | ms/query |",
             "|---|" + "---|" * (len(KS) + 2)]
    for s in summary:
        lines.append(f"| {s['config']} | " + " | ".join(f"{s[f'R@{k}']:.2f}" for k in KS)
                     + f" | {s['MRR']:.3f} | {s['ms/query']:.0f} |")
    lines += ["", "## Rank of first gold hit per question", "", "`-` = not in top 10.", "",
              "| Question | " + " | ".join(labels) + " |", "|---|" + "---|" * len(labels)]
    for q in questions:
        ranks = per_q[q["id"]]
        lines.append(f"| {q['question']} | " + " | ".join(str(ranks[l] or "-") for l in labels) + " |")
    return "\n".join(lines) + "\n"


def main() -> None:
    setup_logging()
    questions = json.loads((HERE / "questions.json").read_text(encoding="utf-8"))
    summary, per_q = evaluate(questions)
    report = write_report(questions, summary, per_q)
    (HERE / "report.md").write_text(report, encoding="utf-8")
    print(report)


if __name__ == "__main__":
    main()
