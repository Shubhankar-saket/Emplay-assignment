FROM python:3.11-slim

# libgl/glib: OpenCV (RapidOCR); everything else ships as wheels
RUN apt-get update && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    FASTEMBED_CACHE_PATH=/models \
    HF_HUB_DISABLE_SYMLINKS_WARNING=1

COPY pyproject.toml README.md ./
COPY core core
COPY ingestion ingestion
COPY search search
COPY agents agents
COPY api api
COPY eval eval
COPY ui ui
COPY main.py ./
RUN pip install ".[ocr,ui]"

# bake the local embedding / BM25 / rerank models into the image so containers start offline
RUN python -c "from search.models import dense_model, sparse_model, reranker; dense_model(); sparse_model(); reranker()"

EXPOSE 8000 8501
CMD ["uvicorn", "api.app:app", "--host", "0.0.0.0", "--port", "8000"]
