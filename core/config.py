"""Central configuration. Every tunable comes from environment variables or .env."""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # paths
    data_dir: Path = Path("data")
    parsed_cache_dir: Path = Path("data/parsed")
    log_dir: Path = Path("logs")

    # vector store: QDRANT_URL set -> server/cloud, else embedded on-disk
    qdrant_url: str | None = None
    qdrant_api_key: str | None = None
    qdrant_path: Path = Path("data/qdrant")
    collection: str = "rfp_chunks"

    # models (local ONNX via fastembed)
    dense_model: str = "BAAI/bge-small-en-v1.5"
    sparse_model: str = "Qdrant/bm25"
    rerank_model: str = "Xenova/ms-marco-MiniLM-L-6-v2"
    bm25_avg_len: int = 220  # approx. avg tokens per chunk, used by BM25 length normalisation

    # chunking (~1500 chars ~= 350 tokens; leaves headroom under the 512-token model limits)
    chunk_max_chars: int = 1500
    chunk_overlap_chars: int = 200

    # retrieval
    prefetch_limit: int = 40
    rerank_candidates: int = 25

    # parsing
    pdf_y_tolerance: float = 5.0  # merges label/value cells that sit a few points apart
    min_page_chars: int = 50  # below this a page is treated as "no text layer"
    unstructured_api_key: str | None = None
    unstructured_api_url: str = "https://transform.unstructured.io/api/v2"
    unstructured_timeout_s: int = 300
    enable_local_ocr: bool = True
    ocr_dpi: int = 200

    # LLM (any OpenAI-compatible endpoint: Groq, OpenRouter, ...)
    llm_base_url: str = "https://api.groq.com/openai/v1"
    llm_api_key: str | None = None
    llm_model: str = "openai/gpt-oss-120b"
    llm_fast_model: str = "openai/gpt-oss-20b"
    llm_timeout_s: int = 60
    llm_max_retries: int = 4
    llm_concurrency: int = 2  # parallel agent LLM calls (free tiers rate-limit hard)
    agent_max_retries: int = 2  # validator -> retrieval -> extraction loops per field


@lru_cache
def get_settings() -> Settings:
    return Settings()
