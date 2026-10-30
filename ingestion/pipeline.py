"""Ingestion pipeline: file -> ParsedDocument (cleaned, classified, OCR fallback, cached)."""
from __future__ import annotations

import hashlib
from pathlib import Path

from core.config import Settings
from core.logging import get_logger, log_event
from core.models import Page, ParsedDocument
from ingestion.classify import classify_document, guess_title, pdf_date
from ingestion.cleaning import clean_pages, clean_text
from ingestion.docx_parser import parse_docx
from ingestion.html_parser import parse_html
from ingestion.ocr import recover_pages
from ingestion.pdf_parser import is_low_text, parse_pdf

log = get_logger("ingestion")
SUPPORTED_SUFFIXES = {".pdf", ".html", ".htm", ".docx"}
# Bump when parsing/cleaning output changes so cached ParsedDocuments are rebuilt.
PARSER_VERSION = 3


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def list_bid_files(bid_dir: Path) -> list[Path]:
    files = sorted(p for p in bid_dir.rglob("*") if p.is_file() and not p.name.startswith((".", "~$")))
    skipped = [p.name for p in files if p.suffix.lower() not in SUPPORTED_SUFFIXES]
    if skipped:  # e.g. .doc / .xlsx: reported, never silently dropped
        log_event(log, "unsupported_files_skipped", level=30, bid_dir=str(bid_dir), files=skipped)
    return [p for p in files if p.suffix.lower() in SUPPORTED_SUFFIXES]


def _parse_pdf_pages(path: Path, settings: Settings) -> tuple[list[Page], dict]:
    """pdfplumber pages, with low-text pages recovered by OCR, then cleaned."""
    pages, meta = parse_pdf(path, y_tolerance=settings.pdf_y_tolerance)
    low = [p for p in pages if is_low_text(p, settings.min_page_chars)]
    recovered = recover_pages(path, [p.page_number for p in low], settings) if low else {}
    for page in low:
        if page.page_number in recovered:
            page.text, page.extraction = recovered[page.page_number]
        else:
            page.extraction = "empty"
            page.warnings.append("no text layer and OCR unavailable/failed")
    return clean_pages(pages), meta


def _parse_docx_pages(path: Path) -> tuple[list[Page], dict]:
    pages, meta = parse_docx(path)
    return clean_pages(pages), meta


def _parse(path: Path, bid_id: str, file_hash: str, settings: Settings) -> ParsedDocument:
    fields: dict[str, str] = {}
    if path.suffix.lower() in (".pdf", ".docx"):
        pages, meta = _parse_pdf_pages(path, settings) if path.suffix.lower() == ".pdf" else _parse_docx_pages(path)
        title, doc_date = guess_title(path, meta, pages[0].text if pages else ""), pdf_date(meta)
    else:
        pages, fields, html_title = parse_html(path)
        for page in pages:
            page.text = clean_text(page.text)
        title, doc_date = fields.get("Title") or html_title or path.stem, None

    doc_type, addendum_number = classify_document(path, pages[0].text if pages else "")
    doc = ParsedDocument(
        bid_id=bid_id, file_name=path.name, file_hash=file_hash, doc_type=doc_type,
        addendum_number=addendum_number, doc_date=doc_date, title=title, pages=pages, fields=fields,
        warnings=[f"p{p.page_number}: {w}" for p in pages for w in p.warnings],
    )
    log_event(log, "file_parsed", bid_id=bid_id, file=path.name, doc_type=doc_type,
              addendum=addendum_number, pages=len(pages),
              empty_pages=[p.page_number for p in pages if p.extraction == "empty"],
              ocr_pages=[p.page_number for p in pages if p.extraction in ("ocr_local", "unstructured_api")])
    return doc


def parse_file(path: Path, bid_id: str, settings: Settings, file_hash: str | None = None) -> ParsedDocument | None:
    """Parse one file, using the on-disk cache keyed by content hash. Never raises."""
    file_hash = file_hash or file_sha256(path)
    cache = settings.parsed_cache_dir / bid_id / f"{file_hash[:16]}.v{PARSER_VERSION}.json"
    if cache.exists():
        try:
            return ParsedDocument.model_validate_json(cache.read_text(encoding="utf-8"))
        except Exception:
            cache.unlink(missing_ok=True)
    try:
        doc = _parse(path, bid_id, file_hash, settings)
    except Exception as exc:
        log_event(log, "file_unreadable", level=40, bid_id=bid_id, file=path.name, error=repr(exc))
        return None
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(doc.model_dump_json(indent=1), encoding="utf-8")
    return doc


def ingest_bid(bid_dir: Path, settings: Settings) -> list[ParsedDocument]:
    bid_id = bid_dir.name
    docs = [d for p in list_bid_files(bid_dir) if (d := parse_file(p, bid_id, settings))]
    log_event(log, "bid_ingested", bid_id=bid_id, documents=len(docs))
    return docs
