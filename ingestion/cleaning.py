"""Text cleaning: whitespace, broken lines, hyphenation and repeated headers/footers."""
from __future__ import annotations

import re
from collections import Counter

from core.models import Page

_CHAR_MAP = str.maketrans({
    " ": " ", " ": " ", "​": "", "﻿": "",
    "‘": "'", "’": "'", "“": '"', "”": '"',
    "–": "-", "—": "-", "•": "*", "": "*",
})


def normalize_text(text: str) -> str:
    """Normalise unicode punctuation and whitespace, keeping line structure."""
    text = text.translate(_CHAR_MAP).replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = "\n".join(line.strip() for line in text.split("\n"))
    return re.sub(r"\n{3,}", "\n\n", text).strip()


def fix_broken_lines(text: str) -> str:
    """Re-join words split across lines and sentences wrapped mid-way.

    Hyphenated line breaks keep the hyphen ("pre-\\nproposal" -> "pre-proposal") because
    RFPs are full of hyphenated identifiers (JA-207652, 210-BLYZ) that must survive intact.
    """
    text = re.sub(r"(\w)-\n(\w)", r"\1-\2", text)
    # a line ending mid-sentence followed by a lowercase continuation
    return re.sub(r"(?<=[a-z0-9,;(])\n(?=[a-z(])", " ", text)


def fix_split_emails(text: str) -> str:
    """'thawkins@treasurer.state.md .us' -> 'thawkins@treasurer.state.md.us' (form cells wrap mid-address)."""
    return re.sub(r"(\b[\w.+-]+@[\w-]+(?:\.[\w-]+)*)\s+(\.[a-z]{2,4})\b", r"\1\2", text)


def clean_text(text: str) -> str:
    return fix_split_emails(fix_broken_lines(normalize_text(text)))


_PAGE_NUM = re.compile(r"\bpage\b|#\s*(of|/)\s*#|^\W*#\W*$")  # on digit-masked lines


def _line_key(line: str) -> str:
    """Normalise a line for header/footer comparison (page numbers vary per page)."""
    return re.sub(r"\d+", "#", line.strip().lower())


def find_repeated_lines(pages: list[Page], edge_lines: int = 3, min_ratio: float = 0.5) -> set[str]:
    """Lines that recur at the top/bottom of many pages (running headers, footers, page numbers)."""
    texts = [p.text for p in pages if p.text.strip()]
    if len(texts) < 3:
        return set()
    masked: Counter[str] = Counter()
    exact: Counter[str] = Counter()
    for text in texts:
        lines = [l.strip() for l in text.split("\n") if l.strip()]
        edges = set(lines[:edge_lines] + lines[-edge_lines:])
        exact.update({l.lower() for l in edges})
        masked.update({_line_key(l) for l in edges})
    threshold = max(3, int(len(texts) * min_ratio))
    # identical lines are headers/footers; lines that only repeat once digits are masked must also look like
    # page numbering ("Page 3 of 40", "3/40", "- 3 -"), or ordinary content like "Item 1", "Item 2" is lost
    return ({_line_key(l) for l, n in exact.items() if n >= threshold}
            | {k for k, n in masked.items() if n >= threshold and _PAGE_NUM.search(k)})


def strip_repeated_lines(pages: list[Page], repeated: set[str]) -> None:
    """Remove running headers/footers in place. The first occurrence is kept so that
    information they carry (e.g. the bid number in a running header) stays searchable."""
    seen: set[str] = set()
    for page in pages:
        kept = []
        for line in page.text.split("\n"):
            key = _line_key(line)
            if key in repeated and line.strip():
                if key in seen:
                    continue
                seen.add(key)
            kept.append(line)
        page.text = "\n".join(kept).strip()


def clean_pages(pages: list[Page]) -> list[Page]:
    for page in pages:
        page.text = normalize_text(page.text)
    strip_repeated_lines(pages, find_repeated_lines(pages))
    for page in pages:
        page.text = fix_split_emails(fix_broken_lines(page.text))
        for table in page.tables:
            table.rows = [[fix_split_emails(c) for c in row] for row in table.rows]
    return pages
