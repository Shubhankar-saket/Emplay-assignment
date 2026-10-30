"""HTML bid-page parsing: label/value fields + cleaned body text, portal chrome removed."""
from __future__ import annotations

import re
from pathlib import Path

from bs4 import BeautifulSoup, Tag

from core.models import Page

# Not "form": portals (BidNet) wrap the whole notice in a <form>; only its controls are noise.
_NOISE_TAGS = ["script", "style", "noscript", "nav", "header", "footer", "input", "select", "textarea",
               "button", "svg", "iframe"]
_NOISE_CLASS = re.compile(r"cookie|navbar|menu|breadcrumb|modal|share|social|print-options|skip", re.I)
_LABEL_CLASS = re.compile(r"(^|[-_ ])label", re.I)
_VALUE_LIKE = re.compile(r"@|\d{3}[-.) ]\d{3}")  # email or phone number


def _text(el: Tag) -> str:
    return re.sub(r"\s+", " ", el.get_text(" ", strip=True)).strip()


def _add_field(fields: dict[str, str], label: str, value: str) -> None:
    label, value = label.strip().rstrip(":"), value.strip()
    if not (label and value and len(label) <= 60):
        return
    # contact blocks chain name -> phone -> email; a "label" that is itself an earlier
    # value (a phone number) means this value continues that field
    owner = next((k for k, v in fields.items() if v == label or v.endswith(", " + label)), None)
    if owner and _VALUE_LIKE.search(label):
        fields[owner] += ", " + value
    elif label not in fields:
        fields[label] = value


def _extract_fields(soup: BeautifulSoup) -> dict[str, str]:
    """Generic label -> value extraction: <dt>/<dd>, <th>/<td> rows and 'label' class + sibling."""
    fields: dict[str, str] = {}

    def add(label: str, value: str) -> None:
        _add_field(fields, label, value)

    for dt in soup.find_all("dt"):
        dd = dt.find_next_sibling("dd")
        if dd:
            add(_text(dt), _text(dd))
    for tr in soup.find_all("tr"):
        cells = tr.find_all(["th", "td"], recursive=False)
        if len(cells) == 2:
            add(_text(cells[0]), _text(cells[1]))
    for el in soup.find_all(class_=_LABEL_CLASS):
        label = _text(el)
        value_el = el.find_next_sibling()
        if value_el is not None:
            add(label, _text(value_el))
        elif el.parent is not None:  # value is loose text in the same container
            add(label, _text(el.parent).removeprefix(label))
    return fields


def parse_html(path: Path) -> tuple[list[Page], dict[str, str], str | None]:
    """Return a single Page (HTML has no pages), the extracted fields and the <title>."""
    soup = BeautifulSoup(path.read_text(encoding="utf-8", errors="ignore"), "html.parser")
    title = _text(soup.title) if soup.title else None
    for tag in soup(_NOISE_TAGS):
        tag.decompose()
    for tag in soup.find_all(class_=_NOISE_CLASS):
        tag.decompose()

    fields = _extract_fields(soup)
    root = soup.find("main") or soup.find(attrs={"role": "main"}) or soup.body or soup
    lines = [l.strip() for l in root.get_text("\n").split("\n") if l.strip()]
    # drop very short UI fragments ("Follow", "Print") but keep values like dates and numbers
    lines = [l for l in lines if len(l) > 3 or re.search(r"\d", l)]
    consumed = set(fields) | set(fields.values())  # already emitted in the field block
    lines = [l for l in lines if l.rstrip(":") not in consumed]
    body = "\n".join(dict.fromkeys(lines))  # de-duplicate repeated menu entries, keep order

    field_block = "\n".join(f"{k}: {v}" for k, v in fields.items())
    text = (field_block + "\n\n" + body).strip() if field_block else body
    return [Page(page_number=1, text=text, extraction="html")], fields, title
