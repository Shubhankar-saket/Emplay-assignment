"""ParsedDocument -> Chunks: section-aware text chunks, one chunk per table, one for HTML fields."""
from __future__ import annotations

import re
import uuid

from core.models import Chunk, ChunkType, ParsedDocument, Table

# "1.1 General Information", "2 Price Schedule", or a short ALL-CAPS line
_HEADING_RE = re.compile(r"^(\d+(\.\d+)*\.?\s+[A-Z][^\n]{2,80}|[A-Z][A-Z0-9 &/,()'\-]{4,80})$")


def chunk_uuid(bid_id: str, file_hash: str, page: int, idx: int) -> str:
    """Deterministic point id: re-indexing the same file overwrites instead of duplicating."""
    return str(uuid.uuid5(uuid.NAMESPACE_URL, f"{bid_id}:{file_hash}:{page}:{idx}"))


def _is_heading(line: str) -> bool:
    return len(line) <= 90 and bool(_HEADING_RE.match(line.strip()))


def _pack(lines: list[str], max_chars: int, overlap: int) -> list[str]:
    """Greedy line packing up to max_chars, carrying ~overlap chars of trailing lines forward."""
    out, cur = [], []
    for line in lines:
        if cur and sum(len(l) + 1 for l in cur) + len(line) > max_chars:
            out.append("\n".join(cur))
            tail, size = [], 0
            for l in reversed(cur):
                if size + len(l) > overlap:
                    break
                tail.insert(0, l)
                size += len(l) + 1
            cur = tail
        cur.append(line)
    if cur:
        out.append("\n".join(cur))
    return out


def _table_texts(table: Table, max_chars: int) -> list[str]:
    """Markdown, split by rows (header repeated). Form layouts return nothing: their label/value
    text is already in the page text chunk, and a second copy only crowds the rerank pool."""
    if table.is_form:
        return []
    header, body = table.rows[0], table.rows[1:]
    parts, rows = [], []
    for row in body:
        rows.append(row)
        if len(Table(rows=[header, *rows]).to_markdown()) > max_chars and len(rows) > 1:
            parts.append(Table(rows=[header, *rows[:-1]]).to_markdown())
            rows = [row]
    parts.append(Table(rows=[header, *rows]).to_markdown())
    return parts


def chunk_document(doc: ParsedDocument, max_chars: int = 1500, overlap: int = 200,
                   min_chars: int = 300) -> list[Chunk]:
    chunks: list[Chunk] = []
    base = doc.model_dump(include={"bid_id", "file_name", "file_hash", "doc_type", "addendum_number", "doc_date"})

    def add(text: str, page_start: int, page_end: int, section: str | None, ctype: ChunkType, extraction) -> None:
        if not text.strip():
            return
        idx = len(chunks)
        chunks.append(Chunk(chunk_id=chunk_uuid(doc.bid_id, doc.file_hash, page_start, idx), page_start=page_start,
                            page_end=page_end, section=section, chunk_type=ctype, extraction=extraction,
                            text=text, **base))

    # HTML field blocks are already the first lines of the page text, so no separate chunk.
    section: str | None = None
    for page in doc.pages:
        runs, section = _section_runs(page.text, section)
        for sec, lines in _merge_small(runs, min_chars):
            for text in _pack(lines, max_chars, overlap):
                add(text, page.page_number, page.page_number, sec, "text", page.extraction)
        for table in page.tables:
            for text in _table_texts(table, max_chars):
                add(text, page.page_number, page.page_number, section, "table", page.extraction)
    return chunks


def _section_runs(text: str, section: str | None) -> tuple[list[tuple[str | None, list[str]]], str | None]:
    """Split page text into (section, lines) runs at headings; the section carries across pages."""
    runs: list[tuple[str | None, list[str]]] = [(section, [])]
    for line in (l for l in text.split("\n") if l.strip()):
        if _is_heading(line):
            section = line.strip()
            runs.append((section, [line]))
        else:
            runs[-1][1].append(line)
    return runs, section


def _merge_small(runs: list[tuple[str | None, list[str]]], min_chars: int) -> list[tuple[str | None, list[str]]]:
    """Fold runs shorter than min_chars into the next run on the page (or the previous, at the end),
    so a lone header line like 'Buyer ... Email ...' keeps the context around it."""
    out: list[tuple[str | None, list[str]]] = []
    carry: list[str] = []
    for sec, lines in runs:
        lines = carry + lines
        if sum(len(l) for l in lines) < min_chars:
            carry = lines
            continue
        out.append((sec, lines))
        carry = []
    if carry:
        if out:
            out[-1] = (out[-1][0], out[-1][1] + carry)
        else:
            out.append((runs[-1][0], carry))
    return out
