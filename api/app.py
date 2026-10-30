"""REST API: POST /index, POST /search, POST /ask. Run: uvicorn api.app:app (or python main.py --serve)."""
from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from agents.qa import Answer, answer_question
from core.config import get_settings
from core.logging import setup_logging
from core.models import Citation, SearchFilters, SearchMode
from search.engine import search
from search.indexer import index_bid

setup_logging(get_settings().log_dir)
app = FastAPI(title="RFP Intelligence", version="0.1.0")


class IndexRequest(BaseModel):
    bid_dir: str
    force: bool = False


class SearchRequest(BaseModel):
    query: str
    filters: SearchFilters | None = None
    k: int = 5
    mode: SearchMode = "hybrid_rerank"


class SearchResult(BaseModel):
    rank: int
    score: float
    citation: Citation
    doc_type: str
    section: str | None
    text: str


class AskRequest(BaseModel):
    question: str
    filters: SearchFilters | None = None


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/index")
def index(req: IndexRequest) -> dict:
    bid_dir = Path(req.bid_dir)
    if not bid_dir.is_dir():
        raise HTTPException(404, f"bid folder not found: {bid_dir}")
    return index_bid(bid_dir, force=req.force)


@app.post("/search", response_model=list[SearchResult])
def search_endpoint(req: SearchRequest) -> list[SearchResult]:
    return [SearchResult(rank=h.rank, score=h.score, citation=h.citation, doc_type=h.chunk.doc_type,
                         section=h.chunk.section, text=h.chunk.text)
            for h in search(req.query, req.filters, k=req.k, mode=req.mode)]


@app.post("/extract")
def extract(req: IndexRequest) -> dict:
    """Index (incremental) + multi-agent extraction; returns the bid's JSON record."""
    from agents.orchestrator import Orchestrator
    bid_dir = Path(req.bid_dir)
    if not bid_dir.is_dir():
        raise HTTPException(404, f"bid folder not found: {bid_dir}")
    return Orchestrator(bid_dir).run_extraction(force_index=req.force)


@app.post("/ask", response_model=Answer)
def ask(req: AskRequest) -> Answer:
    return answer_question(req.question, req.filters)
