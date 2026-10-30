# Demo video script (about 8 minutes)

**Before recording**
- `.env` has `LLM_API_KEY`. Run `python main.py --bid ./data/bids/Bid1` and `--bid ./data/bids/Bid2` once, so
  every LLM response is cached and the live runs in the video finish in seconds.
- Close anything using the index, then open two windows: a terminal in the project folder, and the browser.
- Have open in the editor: `README.md` (scrolled to the diagram), `outputs/Bid1.json`, `outputs/Bid1_trace.json`.

---

## 1. The problem (0:00–0:45)

**Show:** the `data/bids/Bid1` and `Bid2` folders.

**Say:** "Bid folders are a mix of RFP PDFs, addenda, spec sheets, affidavits and saved portal pages. The
answers a bidder needs are scattered across them, and the addenda change them. Bid1's due date is June 27
in the RFP, but Addendum 2 moves it to July 9. This system turns a folder like this into a searchable index
and a 20-field record where every value cites its file and page, and anything the documents don't say is
'Not found in documents' instead of a guess."

## 2. Architecture (0:45–2:00)

**Show:** the Mermaid diagram in the README.

**Say, following the diagram left to right:**
- "Ingestion: pdfplumber for PDFs and tables, python-docx for Word, BeautifulSoup for the portal pages.
  Pages with no text layer, like a W-9 in Bid1 drawn as vector shapes, go to OCR. Text is cleaned: running
  headers removed, broken lines and split emails rejoined. Each file is tagged with bid, document type,
  addendum number, page and date."
- "Search: chunks follow sections and tables. Each chunk gets a dense embedding and a BM25 vector, both
  computed locally, so search needs no API key. Qdrant fuses the two with RRF, applies metadata filters, and
  a cross-encoder re-ranks the result."
- "Agents: an orchestrator runs ingestion, then three extraction agents in parallel. Each gets evidence only
  through the retrieval agent, never whole documents. Then addendum reconciliation, a validator that sends
  rejected fields back for another retrieval attempt, and a report agent."

## 3. Extraction run (2:00–3:45)

**Do:** in the terminal, `python main.py --bid ./data/bids/Bid1`

**Say:** "One command: incremental indexing (unchanged files are skipped by content hash), then extraction."

**Show:** `outputs/Bid1.json`, the Due Date field.

**Say:** "Value 'July 9, 2:00 PM CST', cited to Addendum 2 page 1. The normalized time is UTC minus 5,
not minus 6: July is daylight time, so 'CST' really means CDT. That conversion is done in code, because
the model got it wrong. Below, the addendum change log records the original June 27 date and which
addendum changed it. Fields the documents don't state, like the bid bond, say 'Not found in documents'."

**Show:** `outputs/Bid1_trace.json`, scrolled through briefly.

**Say:** "Every agent step is in the trace: what each retrieval found, what each extraction returned, the
validator's verdicts, and the retry routing."

## 4. Streamlit UI (3:45–6:00)

**Do:** `streamlit run ui/app.py`, open http://localhost:8501.

- **Extract tab:** "The same record: validation counts, each field with confidence and sources, addendum
  changes." Scroll to the go/no-go panel: "A go/no-go agent checks the bid against a bidder profile in
  `config/bidder_profile.json`. For this example reseller, Bid1 is NO-GO: about 100,000 devices exceed its
  20,000-unit capacity. Bid2 is a GO. Deadline status is computed separately; both closed in 2024."
- **Search tab:** search `210-BLYZ`. "An exact part number: BM25 finds it, which vector search alone
  misses." Then search `warranty` with Bid = Bid2 and type = rfp: "metadata filters."
- **Ask tab:** "What is the deadline after all addendums?" with Bid1: "answer from Addendum 2, with the quote
  and the agent trace." Then "Compare the warranties of both bids." with All bids: "evidence from each bid."
- **Compare tab:** Bid1 vs Bid2: "field-by-field comparison built from the two validated records."

## 5. Evaluation (6:00–7:00)

**Show:** the Evaluation tab or `eval/report.md`.

**Say:** "21 questions with gold file and page. Vector-only finds the right passage first 62% of the time,
hybrid with re-ranking 81%, MRR 0.73 to 0.85. Vector search misses identifiers, BM25 misses paraphrases,
hybrid gets both. The Q&A log has 12 questions with cited answers, including one correct 'Not found'."

## 6. Engineering and limits (7:00–8:00)

**Show:** the GitHub repo: green CI badge or Actions tab, `docker-compose.yml`, `tests/`.

**Say:**
- "34 unit tests run in CI on every push, then the Docker image is built. `docker compose up` starts the API,
  the UI and a Qdrant server."
- "It runs on any bid folder with no code changes: the folder name is the bid ID, and document types come
  from names and content. Corrupt or unsupported files are logged and skipped."
- "Built for Groq's free tier: prompts are capped, the validator runs on a smaller model, retries honour
  rate limits, and every response is cached."
- "Known limits: the Bid1 buyer's name sits only in a page header and is missed, and Required Documents is
  the weakest field. A stronger model is a config change."

**Close:** "Thanks for watching."
