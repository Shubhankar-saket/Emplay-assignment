"""Fallback text recovery for pages pdfplumber cannot read (scans, vector-outlined text).

Order: Unstructured Transform API if UNSTRUCTURED_API_KEY is set -> local RapidOCR if
installed -> give up and log. Only the affected pages are sent, never the whole file, and
all of a file's flagged pages go to the API in one request.
"""
from __future__ import annotations

from pathlib import Path

import httpx
import pymupdf

from core.config import Settings
from core.logging import get_logger, log_event
from core.models import Extraction

log = get_logger("ingestion.ocr")
_rapid_engine = None


def _subset_pdf(path: Path, page_numbers: list[int]) -> bytes:
    """A new PDF holding only the given (1-based) pages, in order."""
    with pymupdf.open(path) as src, pymupdf.open() as out:
        for n in page_numbers:
            out.insert_pdf(src, from_page=n - 1, to_page=n - 1)
        return out.tobytes()


def _unstructured_api(path: Path, page_numbers: list[int], settings: Settings) -> dict[int, str]:
    """POST the flagged pages to Transform's /parse and map elements back to original pages."""
    url = settings.unstructured_api_url.rstrip("/") + "/parse"
    resp = httpx.post(
        url,
        headers={"unstructured-api-key": settings.unstructured_api_key,
                 "Prefer": f"wait={settings.unstructured_timeout_s}"},
        files={"input": (f"{path.stem}_ocr.pdf", _subset_pdf(path, page_numbers), "application/pdf")},
        data={"output": "elements"},
        timeout=settings.unstructured_timeout_s + 30,
    )
    resp.raise_for_status()
    if resp.status_code == 202:  # still processing after the wait window
        raise TimeoutError(f"parse not finished after {settings.unstructured_timeout_s}s")
    texts: dict[int, list[str]] = {}
    for el in resp.json().get("elements") or []:
        sub_page = (el.get("metadata") or {}).get("page_number") or 1
        if el.get("text") and 1 <= sub_page <= len(page_numbers):
            texts.setdefault(page_numbers[sub_page - 1], []).append(el["text"])
    return {n: "\n".join(t) for n, t in texts.items()}


def _rapidocr_page(path: Path, page_number: int, settings: Settings) -> str | None:
    global _rapid_engine
    import numpy as np
    from rapidocr_onnxruntime import RapidOCR  # optional dependency

    if _rapid_engine is None:
        _rapid_engine = RapidOCR()
    with pymupdf.open(path) as doc:
        pix = doc[page_number - 1].get_pixmap(dpi=settings.ocr_dpi)
        img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
        if pix.n == 4:
            img = img[:, :, :3]
    result, _ = _rapid_engine(img)
    if not result:
        return None
    # result: [box, text, score]; rebuild reading order line by line
    items = sorted(((b[0][1], b[0][0], t) for b, t, _ in result), key=lambda x: (round(x[0] / 12), x[1]))
    lines: list[str] = []
    last_row = None
    for y, _, text in items:
        row = round(y / 12)
        if row == last_row:
            lines[-1] += " " + text
        else:
            lines.append(text)
        last_row = row
    return "\n".join(lines)


def _rapidocr(path: Path, page_numbers: list[int], settings: Settings) -> dict[int, str]:
    out = {}
    for n in page_numbers:
        if text := _rapidocr_page(path, n, settings):
            out[n] = text
    return out


def recover_pages(path: Path, page_numbers: list[int], settings: Settings) -> dict[int, tuple[str, Extraction]]:
    """Recover text for the given pages. Pages no backend could read are absent from the result."""
    attempts = []
    if settings.unstructured_api_key and settings.unstructured_api_url:
        attempts.append(("unstructured_api", _unstructured_api))
    if settings.enable_local_ocr:
        attempts.append(("ocr_local", _rapidocr))

    recovered: dict[int, tuple[str, Extraction]] = {}
    for method, fn in attempts:
        todo = [n for n in page_numbers if n not in recovered]
        if not todo:
            break
        try:
            for n, text in fn(path, todo, settings).items():
                if text.strip():
                    recovered[n] = (text, method)  # type: ignore[assignment]
            log_event(log, "pages_recovered", file=path.name, method=method,
                      pages=[n for n in todo if n in recovered])
        except ImportError as exc:
            log_event(log, "ocr_backend_missing", file=path.name, method=method, error=str(exc))
        except Exception as exc:
            log_event(log, "ocr_failed", file=path.name, pages=todo, method=method, error=repr(exc)[:300])
    missing = [n for n in page_numbers if n not in recovered]
    if missing:
        log_event(log, "pages_unrecoverable", file=path.name, pages=missing)
    return recovered
