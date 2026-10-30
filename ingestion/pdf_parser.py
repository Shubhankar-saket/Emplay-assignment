"""PDF parsing with pdfplumber (primary). Pages without a text layer are flagged for OCR."""
from __future__ import annotations

from pathlib import Path

import pdfplumber

from core.logging import get_logger, log_event
from core.models import Page, Table

log = get_logger("ingestion.pdf")


def _clean_table(raw: list[list[str | None]]) -> Table | None:
    """Drop empty rows/columns; keep only grids that look like real tables
    (pdfplumber also reports page-layout boxes full of prose as 'tables')."""
    rows = [[(c or "").replace("\n", " ").strip() for c in row] for row in raw]
    rows = [r for r in rows if any(r)]
    if not rows:
        return None
    width = max(len(r) for r in rows)
    rows = [r + [""] * (width - len(r)) for r in rows]
    keep_cols = [j for j in range(width) if any(r[j] for r in rows)]
    rows = [[r[j] for j in keep_cols] for r in rows]
    if len(rows) < 2 or len(keep_cols) < 2:
        return None
    cells = [c for r in rows for c in r if c]
    if sum(len(c) for c in cells) / len(cells) > 300:  # prose box, not a table
        return None
    return Table(rows=rows)


def parse_pdf(path: Path, y_tolerance: float = 5.0) -> tuple[list[Page], dict]:
    """Return one Page per PDF page plus the PDF metadata dict."""
    pages: list[Page] = []
    with pdfplumber.open(path) as pdf:
        meta = dict(pdf.metadata or {})
        for i, pl_page in enumerate(pdf.pages, start=1):
            page = Page(page_number=i)
            try:
                page.text = pl_page.extract_text(y_tolerance=y_tolerance) or ""
            except Exception as exc:  # corrupt content stream etc.
                page.warnings.append(f"text extraction failed: {exc}")
                log_event(log, "page_text_failed", file=path.name, page=i, error=str(exc))
            try:
                page.tables = [t for raw in pl_page.extract_tables() if (t := _clean_table(raw))]
            except Exception as exc:
                page.warnings.append(f"table extraction failed: {exc}")
                log_event(log, "page_tables_failed", file=path.name, page=i, error=str(exc))
            pages.append(page)
    return pages, meta


def is_low_text(page: Page, min_chars: int) -> bool:
    return len(page.text.strip()) < min_chars
