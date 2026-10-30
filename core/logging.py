"""Structured (JSON lines) logging shared by ingestion, search and agents."""
from __future__ import annotations

import json
import logging
import sys
import time
from pathlib import Path
from typing import Any

_CONFIGURED = False


class _JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "ts": time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(record.created)),
            "level": record.levelname,
            "logger": record.name,
            "event": record.getMessage(),
        }
        payload.update(getattr(record, "fields", {}))
        return json.dumps(payload, default=str, ensure_ascii=False)


def setup_logging(log_dir: Path = Path("logs"), level: int = logging.INFO) -> None:
    global _CONFIGURED
    if _CONFIGURED:
        return
    log_dir.mkdir(parents=True, exist_ok=True)
    root = logging.getLogger("rfp")
    root.setLevel(level)
    file_handler = logging.FileHandler(log_dir / "rfp.jsonl", encoding="utf-8")
    file_handler.setFormatter(_JsonFormatter())
    console = logging.StreamHandler(sys.stderr)
    console.setFormatter(logging.Formatter("%(levelname)s %(name)s: %(message)s"))
    console.setLevel(logging.WARNING)
    root.addHandler(file_handler)
    root.addHandler(console)
    _CONFIGURED = True


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(f"rfp.{name}")


def log_event(logger: logging.Logger, event: str, level: int = logging.INFO, **fields: Any) -> None:
    """Log an event with structured fields (written as JSON to logs/rfp.jsonl)."""
    logger.log(level, event, extra={"fields": fields})
