"""Infer doc_type, addendum number, title and date for a file without bid-specific rules."""
from __future__ import annotations

import re
from datetime import datetime
from pathlib import Path

from core.models import DocType

_ADDENDUM_RE = re.compile(r"\b(?:addendum|amendment)\s*(?:no\.?|number|#)?\s*(\d+)", re.I)


def _head(text: str, n_lines: int = 6) -> str:
    return "\n".join([l for l in text.split("\n") if l.strip()][:n_lines]).lower()


def classify_document(path: Path, first_page_text: str) -> tuple[DocType, int | None]:
    name = path.stem.lower().replace("_", " ")
    head = _head(first_page_text)

    if path.suffix.lower() in {".html", ".htm"}:
        return "bid_page", None

    m = _ADDENDUM_RE.search(name) or _ADDENDUM_RE.search(head)
    if m or "addendum" in name:
        return "addendum", int(m.group(1)) if m else None
    if "affidavit" in name or "affidavit" in head:
        return "affidavit", None
    if re.search(r"\bspec(s|ification)?\b", name) or re.search(r"\bsku\b", head):
        return "specs", None
    if re.search(r"\b(w-?9|form|certificate)\b", name):
        return "form", None
    if re.search(r"\b(rfp|rfq|ifb|itb|porfp|bid|solicitation|proposal)\b", name + "\n" + head) \
            or re.search(r"\b[a-z]{1,4}-\d{4,}\b", name):
        return "rfp", None
    return "other", None


def pdf_date(meta: dict) -> str | None:
    """Parse a PDF metadata date (D:20240613175856-05'00') to YYYY-MM-DD."""
    for key in ("ModDate", "CreationDate", "modDate", "creationDate"):
        raw = meta.get(key)
        if isinstance(raw, str):
            m = re.search(r"(\d{8})", raw)
            if m:
                try:
                    return datetime.strptime(m.group(1), "%Y%m%d").date().isoformat()
                except ValueError:
                    continue
    return None


def guess_title(path: Path, meta: dict, first_page_text: str) -> str:
    title = meta.get("Title") or meta.get("title")
    if isinstance(title, str) and title.strip() and not title.lower().startswith("microsoft word"):
        return title.strip()
    return path.stem.replace("_", " ").strip()
