"""Command-line entry point.

  python main.py --bid ./data/bids/Bid1                 index + multi-agent extraction -> outputs/Bid1.json
  python main.py --bid ./data/bids/Bid1 --index-only    index only (incremental)
  python main.py --search "warranty" --bid-id Bid2      hybrid search with citations
  python main.py --ask "What is the deadline?" --bid-id Bid1
  python main.py --serve                                start the REST API on :8000
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import warnings
from pathlib import Path

os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
warnings.filterwarnings("ignore")


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")  # Windows consoles default to cp1252
    p = argparse.ArgumentParser(description="RFP Intelligence")
    p.add_argument("--bid", type=Path, help="bid folder to index")
    p.add_argument("--force", action="store_true", help="re-index even unchanged files")
    p.add_argument("--index-only", action="store_true", help="index the bid without running extraction")
    p.add_argument("--search", metavar="QUERY")
    p.add_argument("--ask", metavar="QUESTION")
    p.add_argument("--bid-id", help="restrict --search/--ask to one bid")
    p.add_argument("--doc-type", action="append", help="restrict to doc types (repeatable)")
    p.add_argument("-k", type=int, default=5)
    p.add_argument("--serve", action="store_true")
    p.add_argument("--port", type=int, default=8000)
    args = p.parse_args()

    from core.config import get_settings
    from core.logging import setup_logging
    from core.models import SearchFilters
    setup_logging(get_settings().log_dir)
    filters = SearchFilters(bid_id=args.bid_id, doc_types=args.doc_type) if (args.bid_id or args.doc_type) else None

    if args.serve:
        import uvicorn
        uvicorn.run("api.app:app", host="127.0.0.1", port=args.port)
        return 0
    if not (args.bid or args.search or args.ask):
        p.print_help()
        return 1
    if args.bid:
        if not args.bid.is_dir():
            print(f"bid folder not found: {args.bid}", file=sys.stderr)
            return 1
        if args.index_only:
            from search.indexer import index_bid
            stats = index_bid(args.bid, force=args.force)
            print(f"{stats['bid_id']}: indexed {len(stats['indexed'])}, skipped {len(stats['skipped'])} unchanged, "
                  f"failed {len(stats['failed'])}, {stats['chunks']} chunks")
        else:
            from agents.orchestrator import Orchestrator
            record = Orchestrator(args.bid).run_extraction(force_index=args.force)
            print(f"{record['bid_id']}: validation {record['validation']}, "
                  f"{len(record['addendum_changes'])} addendum changes -> outputs/{record['bid_id']}.json")
    if args.search:
        from search.engine import search
        for h in search(args.search, filters, k=args.k):
            c = h.chunk
            print(f"#{h.rank} [{c.file_name} p.{c.page_start}] ({c.doc_type})\n{c.text[:400]}\n")
    if args.ask:
        from agents.qa import answer_question
        print(json.dumps(answer_question(args.ask, filters).model_dump(), indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
