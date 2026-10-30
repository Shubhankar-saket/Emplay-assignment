"""Data models shared across ingestion, search and agents."""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

DocType = Literal["bid_page", "rfp", "addendum", "specs", "affidavit", "form", "other"]
Extraction = Literal["text", "html", "ocr_local", "unstructured_api", "empty"]
ChunkType = Literal["text", "table", "fields"]
SearchMode = Literal["dense", "bm25", "hybrid", "hybrid_rerank"]


class Table(BaseModel):
    rows: list[list[str]]

    @property
    def is_form(self) -> bool:
        """Form layouts (label/value boxes) come out as wide grids of mostly empty cells."""
        cells = [c for r in self.rows for c in r]
        return bool(cells) and sum(not c for c in cells) / len(cells) > 0.5

    def form_lines(self) -> list[str]:
        """One line per row with empty and repeated cells dropped: 'Agency POC Name: Tamaira Hawkins ...'."""
        return [" ".join(dict.fromkeys(c for c in r if c)) for r in self.rows if any(r)]

    def to_markdown(self) -> str:
        if not self.rows:
            return ""
        width = max(len(r) for r in self.rows)
        rows = [r + [""] * (width - len(r)) for r in self.rows]
        lines = ["| " + " | ".join(rows[0]) + " |", "|" + " --- |" * width]
        lines += ["| " + " | ".join(r) + " |" for r in rows[1:]]
        return "\n".join(lines)


class Page(BaseModel):
    page_number: int
    text: str = ""
    tables: list[Table] = Field(default_factory=list)
    extraction: Extraction = "text"
    warnings: list[str] = Field(default_factory=list)


class ParsedDocument(BaseModel):
    bid_id: str
    file_name: str
    file_hash: str
    doc_type: DocType
    addendum_number: int | None = None
    doc_date: str | None = None  # ISO date (YYYY-MM-DD) when available
    title: str | None = None
    pages: list[Page]
    fields: dict[str, str] = Field(default_factory=dict)  # label -> value pairs (HTML bid pages)
    warnings: list[str] = Field(default_factory=list)


class Chunk(BaseModel):
    chunk_id: str
    bid_id: str
    file_name: str
    file_hash: str
    doc_type: DocType
    addendum_number: int | None = None
    doc_date: str | None = None
    page_start: int
    page_end: int
    section: str | None = None
    chunk_type: ChunkType = "text"
    extraction: Extraction = "text"
    text: str

    def embedding_text(self) -> str:
        """Text sent to the embedders: a short context header + the chunk body."""
        label = self.doc_type + (f" {self.addendum_number}" if self.addendum_number else "")
        header = f"[{self.bid_id} | {label} | {self.file_name} p.{self.page_start}]"
        if self.section:
            header += f" {self.section}"
        return f"{header}\n{self.text}"


class SearchFilters(BaseModel):
    bid_id: str | None = None
    doc_types: list[DocType] | None = None
    addendum_number: int | None = None
    file_name: str | None = None


class Citation(BaseModel):
    file: str
    page: int
    page_end: int | None = None


class SearchHit(BaseModel):
    chunk: Chunk
    score: float
    rank: int

    @property
    def citation(self) -> Citation:
        c = self.chunk
        return Citation(file=c.file_name, page=c.page_start,
                        page_end=c.page_end if c.page_end != c.page_start else None)
