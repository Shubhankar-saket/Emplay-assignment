# RFP Intelligence: Build Plan

_Last updated: 2026-10-30_

We are building a Python platform that ingests bid folders (HTML + PDF), makes them searchable with hybrid RAG, and uses LLM agents to extract a 20-field JSON record per bid and answer questions with citations. It must run on an unseen bid folder with no code changes.

## 1. Grading

| Part | Weight |
|---|---|
| B. RAG search engine | 25% |
| C. Multi-agent system | 25% |
| D. Structured extraction (20 fields) | 20% |
| Robustness | 10% |
| Code quality | 10% |
| Documentation + demo | 10% |

Part A (ingestion) has no separate weight; everything above depends on it.

## 2. Stack and decisions

| Layer | Choice | Why |
|---|---|---|
| PDF parsing | pdfplumber (text + tables), PyMuPDF for rendering | Handles all 9 files; gives tables |
| OCR fallback | Unstructured API (`hi_res`) when `UNSTRUCTURED_API_KEY` is set, else local RapidOCR | Only for pages with < 50 chars of text (Bid1 pp. 54–59, W-9) |
| HTML parsing | BeautifulSoup, labelled fields as key/value | BidNet pages are mostly navigation |
| Embeddings | fastembed: `BAAI/bge-small-en-v1.5` (dense), `Qdrant/bm25` (sparse) | Local ONNX, no API key to search |
| Vector store | Qdrant: embedded on disk if `QDRANT_URL` unset, else Docker/Cloud | Same code both ways; BM25 + RRF built in |
| Search | Dense + sparse prefetch → RRF fusion → cross-encoder rerank (`Xenova/ms-marco-MiniLM-L-6-v2`) | Hybrid search as the assignment asks |
| Incremental index | File-hash manifest; deterministic chunk IDs; delete by `bid_id` | New bid never rebuilds the index |
| LLM | Groq via OpenAI-compatible client. `openai/gpt-oss-120b` (main), `openai/gpt-oss-20b` (fast). Fallback: `qwen/qwen3.8-27b` or OpenRouter free | Free tier; JSON mode works (tested) |
| API / CLI | FastAPI (`/index`, `/search`, `/ask`) + `python main.py --bid ./Bid1` | Required entry points |

All keys and model names live in `.env` (gitignored). `.env.example` documents them.

Rules carried from earlier tests:
- Normalise ID-like tokens (`JA-207652` → `ja207652`) before BM25 tokenising.
- Time-zone conversion is done in code, not by the LLM (the 20B model got CST vs CDT wrong).
- Close the embedded Qdrant client explicitly on Windows.
- Agents only see search results, never whole documents.

## 3. Current status

| Area | Status |
|---|---|
| `pyproject.toml`, `.env`, `.env.example`, `.gitignore` | Done |
| `core/` (config, logging, models) | Done; LLM settings added |
| `ingestion/` (pdf, html, ocr, cleaning, classify, pipeline) | Done; all 9 files parse (Phase 1) |
| `.venv` | All dependencies installed |
| `search/`, `agents/`, `api/`, `eval/`, `tests/`, `main.py` | Empty |
| Unstructured | Transform API `POST /api/v2/parse` via httpx (no SDK); verified on W-9 pages |
| Groq key + models | Verified |

## 4. Build phases

Each phase ends with a check we run before moving on.

### Phase 1: Ingestion working end to end (done 2026-10-30)
- [x] Unstructured: the key is for the Transform API; switched OCR fallback to `POST /api/v2/parse` (all flagged pages of a file in one request, ~90 s for 6 pages)
- [x] Add `LLM_*` settings to `core/config.py` and `.env.example`
- [x] Run the pipeline on Bid1 and Bid2; inspect parsed JSON in `data/parsed/`
- [x] Check: headers/footers removed, doc types right, addendum numbers set, Bid2 spec SKUs paired with descriptions, W-9 pages OCR'd and tagged
- [x] Fix: HTML parser dropped `<form>`, which wraps the whole BidNet notice; now 19 / 14 fields extracted
- [x] Parse cache key includes `PARSER_VERSION` so parser changes invalidate old cache entries

**Done when:** all 9 files parse with no crash and metadata looks right.

Added 2026-10-31: `.docx` support (`ingestion/docx_parser.py`: paragraphs + tables in order, pages split at Word page breaks, core-properties date). Unsupported files (`.doc`, `.xlsx`, ...) are logged as `unsupported_files_skipped`, not silently dropped.

Known leftovers (minor): HTML contact block pairs name → phone → email as label/value; HTML description appears in both the field block and the body.

### Phase 2: Search engine (`search/`) (done 2026-10-30)
- [x] Chunker: section-aware text, tables split by rows with header repeated, HTML fields chunk (422 chunks total)
- [x] Indexer: dense + sparse (IDF) vectors; skips files whose hash is indexed (re-run 0.1 s); changed file replaces its chunks
- [x] Hybrid search (RRF) with filters (`bid_id`, `doc_type`, `addendum_number`, `file_name`)
- [x] Query expansion via synonym map (LLM rewrite comes with the Retrieval agent)
- [x] Cross-encoder rerank
- [x] Every hit returns file, page, chunk text, score

**Result:** "submission deadline" for Bid1 ranks Addendum 2 #1 with an addendum filter, #5 without. The Retrieval/Reconciliation agents always run the addendum-filtered query for date fields.

### Phase 3: Evaluation (`eval/`) (done 2026-10-30)
- [x] 21 questions with gold passages (file + page) across both bids (`eval/questions.json`)
- [x] Recall@1/3/5 and MRR for 5 configs (`python -m eval.run_eval`)
- [x] `eval/report.md`

**Result (v1):** hybrid + rerank R@5 0.95 / MRR 0.82 vs vector-only 0.86 / 0.68.

**Ingestion fixes (v2):** form tables (mostly empty cells) no longer indexed as a second copy of the page text; small runs (< 300 chars) merged into neighbours; split emails rejoined; HTML contact name/phone/email merged into one field; HTML field block no longer duplicated. Rerank now blends cross-encoder rank with hybrid rank (RRF) instead of replacing it.

**Result (v2):** hybrid + rerank R@1 0.81 / MRR 0.85 (best). Contact questions recovered from "not in top 10" to rank 4–7, but stay weak: the documents say "Agency POC" / "Buyer", which the small reranker does not link to "procurement officer". Contact extraction will lean on the HTML fields and a targeted query. Rerank ~2 s/query on CPU.

### Phase 4: API + CLI (done 2026-10-30)
- [x] FastAPI `/health`, `/index`, `/search`, `/ask` (`api/app.py`, `python main.py --serve`)
- [x] `main.py --bid <folder>`, `--search`, `--ask`, `--bid-id`, `--doc-type` filters
- [x] `agents/llm.py`: OpenAI-compatible client, JSON mode, retries with backoff, disk cache, token usage
- [x] `agents/qa.py`: grounded answer + citations + quotes; "Not found in documents" when unsupported

**Result:** Bid1 deadline → "July 9, 2024 at 2:00 PM CST" citing Addendum 2 p.1; Bid2 bid bond → "Not found in documents". `--bid` will also run extraction once Phase 5 lands.

### Phase 5: Agents (`agents/`) (done 2026-10-31)
- [x] LLM client: OpenAI-compatible, JSON mode, retries (Retry-After aware), no retry on 4xx/413, disk cache, token usage
- [x] Shared state (`BidState`) + Pydantic messages (`RetrievalRequest/Result`, `ExtractionRequest/Result`, `ValidationVerdict`, `AddendumChange`)
- [x] Orchestrator, Ingestion, Retrieval (query rewrite on rejection), Extraction x3 (dates / commercial / product), Addendum reconciliation, Validator (max 2 retries), Report, Q&A
- [x] Extraction groups and validator batches run in parallel (2 workers: Groq free tier is 8K tokens/min on gpt-oss-120b)
- [x] Every step in `outputs/<bid>_trace.json` and `logs/rfp.jsonl`
- [x] "Not found in documents" when evidence is missing; date fields normalised in code (`agents/timeutil.py`)
- [x] Validator + query rewrite run on `gpt-oss-20b` (separate quota); prompts capped (4 fields/call, 700-char sources)

**Result:** Bid1 15 accepted / 4 not found, 4 addendum changes (due date July 9 2:00 PM CDT from Addendum 2); Bid2 16 accepted / 3 not found. ~1 min and ~14 LLM calls per bid.
Known gaps: Bid1 contact lacks the buyer name; Bid1 required documents only lists Form 1295; Bid2 misses the Mercury affidavit; change log `old_value` sometimes null.

### Phase 6: Extraction output (done 2026-10-31)
- [x] 20 fields, each with value, sources (file + page), confidence, notes, status (+ `normalized` for dates)
- [x] Addendum change log with original value + validation counts + LLM usage
- [x] `outputs/Bid1.json`, `outputs/Bid2.json` (+ `_summary.md`, `_trace.json`)
- [x] Retrieval agent fuses its queries with RRF (k=4 per query); extraction sees 5 sources per field; extractor reports `original_value` when an addendum overrides

**Result:** Bid1 due date "July 9, 2024 at 2:00 PM CST" → `2024-07-09T14:00:00-05:00`, change logged from 27-JUN-2024 14:00 (Addendum 2). Both bids 15 accepted / 4 not found / 0 rejected.

Known limits (documented, not tuned further to avoid overfitting to two bids): Bid1 contact misses the buyer name (only in the RFP's running header, ranks low); Bid1 required documents includes W-7/SS-4 mentioned inside W-9 instructions; Bid2 required documents judged "not found" although the affidavits are in the package (LLM judgement; a stronger model should fix it). Results vary slightly between runs with different prompts (LLM non-determinism).

### Phase 7: Tests, docs, deliverables
- [ ] Unit tests for parsing and search (`pytest`)
- [ ] README: setup, design decisions, trade-offs
- [ ] Architecture diagram
- [ ] Q&A log: 10+ questions with cited answers
- [ ] One full agent trace
- [ ] Demo video (5–10 min)

### Phase 8: Bonus (if time allows)
- [ ] Streamlit UI
- [ ] Bid comparison agent + go/no-go recommendation
- [ ] Dockerfile + docker-compose + CI

## 5. Data traps and how we handle them

| Trap | Handling |
|---|---|
| Bid1 RFP pp. 54–59 have no text layer (W-9 drawn as vectors) | Flag pages < 50 chars, OCR them, tag `ocr: true`, lower confidence |
| Due date differs: RFP 27-JUN-2024, Addendum 2 July 9 2:00 PM CST, HTML 3:00 PM EDT | Reconciliation agent: latest addendum wins; normalise time zones in code; log the change |
| Bid2 spec PDF splits SKUs from descriptions | Pair by word positions (y-coordinate) |
| Form tables come out one word per line | Table-aware extraction + line rejoining in cleaning |
| Bid2 has two IDs: BPM044557 and #E20P4600040 | Return both in Bid Number |
| Source errors ("Clavert"/"Calvert", laptops under "Printers", model field "SI# CC7802") | Validator notes them; not treated as extraction failures |
| Repeated header "Request For Proposal 168884 … Page X of 40" | Removed in cleaning before chunking |
| HTML pages mostly navigation; Bid1 description truncated | Extract labelled fields only |
| Fields legitimately absent (Bid2 pre-bid, bid bond, addendums) | "Not found in documents" |
| BM25 splits `JA-207652` | Normalise ID tokens before tokenising |

## 6. Deliverables checklist

- [ ] Git repo
- [ ] README (setup + design decisions)
- [ ] Architecture diagram
- [ ] JSON output per bid
- [ ] Retrieval evaluation report
- [ ] Q&A log (10+ cited answers)
- [ ] One full agent trace
- [ ] Demo video

## 7. Risks and open questions

- Unstructured URL may be the platform endpoint, not the partition endpoint. Test in Phase 1.
- Groq free-tier rate limits may slow extraction runs. Mitigate with caching, backoff, and the 20B model for light steps.
- Qdrant Cloud free clusters sleep when idle. Embedded mode is the default so reviewers are not blocked.
- Embedded Qdrant locks its folder to one process: API and CLI cannot run at the same time in that mode. Document it; use Docker for both.
- Groq model list changes over time. Model names are config only.
