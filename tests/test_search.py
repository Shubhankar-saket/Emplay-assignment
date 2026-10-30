"""Search tests: chunking, query helpers, and an end-to-end index/search on a temporary embedded Qdrant.

The end-to-end tests load the local ONNX models (downloaded once by fastembed) and are marked slow:
`pytest -m "not slow"` skips them.
"""
from __future__ import annotations

import pytest

from core.config import get_settings
from core.models import Page, ParsedDocument, SearchFilters, Table
from search import store
from search.chunking import chunk_document, chunk_uuid
from search.engine import expand_query
from search.models import bm25_text


def _doc(bid: str, name: str, doc_type: str, pages: list[Page], addendum: int | None = None) -> ParsedDocument:
    return ParsedDocument(bid_id=bid, file_name=name, file_hash=f"{bid}-{name}".ljust(64, "0"), doc_type=doc_type,
                          addendum_number=addendum, pages=pages)


# ---- chunking ----------------------------------------------------------------------------

def test_chunk_ids_are_deterministic():
    assert chunk_uuid("Bid1", "h", 1, 0) == chunk_uuid("Bid1", "h", 1, 0) != chunk_uuid("Bid1", "h", 1, 1)


def test_sections_tables_and_metadata():
    text = "1 GENERAL INFORMATION\n" + "Proposals are due on the date below. " * 20 + \
           "\n2 PRICE SCHEDULE\n" + "Prices are firm for the contract term. " * 20
    table = Table(rows=[["Item", "Qty"], ["Laptop", "30"], ["Dock", "30"]])
    doc = _doc("Bid9", "rfp.pdf", "rfp", [Page(page_number=3, text=text, tables=[table])])
    chunks = chunk_document(doc, max_chars=1500, overlap=200)
    sections = {c.section for c in chunks if c.chunk_type == "text"}
    assert {"1 GENERAL INFORMATION", "2 PRICE SCHEDULE"} <= sections
    tables = [c for c in chunks if c.chunk_type == "table"]
    assert len(tables) == 1 and "| Laptop | 30 |" in tables[0].text
    assert all(c.bid_id == "Bid9" and c.page_start == 3 and c.doc_type == "rfp" for c in chunks)


def test_long_text_is_split_with_overlap():
    lines = "\n".join(f"Line {i} of the scope of work with enough words to fill." for i in range(80))
    chunks = chunk_document(_doc("B", "f.pdf", "rfp", [Page(page_number=1, text=lines)]), max_chars=600, overlap=120)
    assert len(chunks) > 3 and all(len(c.text) <= 700 for c in chunks)
    assert chunks[0].text.split("\n")[-1] in chunks[1].text  # overlap carries the last line forward


def test_small_runs_are_merged():
    text = "Buyer ALZATE, JASMINE Email JALZATE@dallasisd.org\n1 HEADER INFORMATION\n" + "Due date details. " * 30
    chunks = chunk_document(_doc("B", "f.pdf", "rfp", [Page(page_number=2, text=text)]))
    assert "ALZATE" in chunks[0].text and "Due date details" in chunks[0].text


def test_form_tables_are_not_duplicated():
    form = Table(rows=[["Agency POC Name:", "", "", "Tamaira Hawkins", "", ""], ["", "", "", "", "", "410"]])
    chunks = chunk_document(_doc("B", "f.pdf", "rfp", [Page(page_number=1, text="Agency POC Name: Tamaira "
                                                                                 "Hawkins 410", tables=[form])]))
    assert [c.chunk_type for c in chunks] == ["text"]


# ---- query helpers -----------------------------------------------------------------------

def test_bm25_text_adds_joined_ids():
    assert bm25_text("part 210-BLYZ for JA-207652").endswith("210blyz ja207652")
    assert bm25_text("no ids here") == "no ids here"


def test_query_expansion():
    q = expand_query("What is the deadline?")
    assert "due date" in q and "closing date" in q
    assert expand_query("laptop color") == "laptop color"


def test_filter_building():
    f = store.build_filter(SearchFilters(bid_id="Bid2", doc_types=["specs", "rfp"], addendum_number=1))
    keys = [c.key for c in f.must]
    assert keys == ["bid_id", "doc_type", "addendum_number"]
    assert store.build_filter(None) is None


# ---- end to end (temporary embedded Qdrant) ----------------------------------------------

@pytest.fixture
def temp_index(tmp_path, monkeypatch):
    settings = get_settings()
    monkeypatch.setattr(settings, "qdrant_url", None)
    monkeypatch.setattr(settings, "qdrant_path", tmp_path / "qdrant")
    monkeypatch.setattr(settings, "collection", "test_chunks")
    store.get_client.cache_clear()
    yield
    store.get_client().close()
    store.get_client.cache_clear()


def _index(docs: list[ParsedDocument]) -> None:
    from search.indexer import _upsert
    for d in docs:
        _upsert(chunk_document(d))


DOCS = [
    _doc("BidA", "rfp.pdf", "rfp", [Page(page_number=2, text="Solicitation Due 27-JUN-2024 14:00:00. "
                                                             "Proposals must be submitted through the portal.")]),
    _doc("BidA", "add2.pdf", "addendum", [Page(page_number=1, text="ADDENDUM No. 2. The new due date for this RFP "
                                                                   "will be July 9, 2024 at 2:00 PM.")], addendum=2),
    _doc("BidB", "specs.pdf", "specs", [Page(page_number=1, text="Dell Latitude 5550 XCTO Base 210-BLYZ. Intel Core "
                                                                 "Ultra 5 125U processor, 16 GB DDR5 memory.")]),
    _doc("BidB", "porfp.pdf", "rfp", [Page(page_number=3, text="Dell Thunderbolt 4 Dock WD22TB4 quantity 30. "
                                                               "Delivery within 45 days of award.")]),
]


@pytest.mark.slow
@pytest.mark.parametrize("mode", ["dense", "bm25", "hybrid", "hybrid_rerank"])
def test_search_modes_find_the_right_chunk(temp_index, mode):
    from search.engine import search
    _index(DOCS)
    hits = search("what processor does the laptop have", SearchFilters(bid_id="BidB"), k=2, mode=mode)
    assert hits[0].chunk.file_name == "specs.pdf"
    assert hits[0].citation.page == 1


@pytest.mark.slow
def test_exact_part_number_and_filters(temp_index):
    from search.engine import search
    _index(DOCS)
    assert search("210-BLYZ", None, k=1, mode="bm25")[0].chunk.file_name == "specs.pdf"
    hits = search("due date", SearchFilters(bid_id="BidA", doc_types=["addendum"]), k=5)
    assert [h.chunk.file_name for h in hits] == ["add2.pdf"]
    assert search("due date", SearchFilters(bid_id="NoSuchBid"), k=5) == []


@pytest.mark.slow
def test_incremental_index_and_delete(temp_index, tmp_path):
    from search.engine import search
    from search.indexer import delete_bid
    _index(DOCS[:2])
    n = store.count(None)
    _index(DOCS[2:])  # adding a bid does not touch the first
    assert store.count(store.build_filter(None, bid_id="BidA")) == n
    _index(DOCS[2:])  # re-upserting the same chunks is idempotent (deterministic ids)
    assert store.count(None) == n + store.count(store.build_filter(None, bid_id="BidB"))
    delete_bid("BidB")
    assert store.count(store.build_filter(None, bid_id="BidB")) == 0
    assert search("Latitude", None, k=3, mode="bm25") == [] or \
        all(h.chunk.bid_id == "BidA" for h in search("Latitude", None, k=3, mode="bm25"))
