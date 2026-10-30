"""Streamlit UI: extract a bid, search, ask questions, read the evaluation.

Run: streamlit run ui/app.py   (embedded Qdrant allows one process: stop the API/CLI first, or set QDRANT_URL)
"""
from __future__ import annotations

import json
import os
import sys
import warnings
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)  # settings use project-relative paths (data/, outputs/)
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
warnings.filterwarnings("ignore")

import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402

from core.config import get_settings  # noqa: E402
from core.logging import setup_logging  # noqa: E402
from core.models import SearchFilters  # noqa: E402

DOC_TYPES = ["rfp", "addendum", "specs", "affidavit", "form", "bid_page", "other"]
MODES = ["hybrid_rerank", "hybrid", "bm25", "dense"]

st.set_page_config(page_title="RFP Intelligence", layout="wide")
setup_logging(get_settings().log_dir)


@st.cache_data(ttl=30)
def indexed_bids() -> list[str]:
    from search import store
    return store.list_bids()


def bid_folders() -> list[Path]:
    root = ROOT / "data" / "bids"
    return sorted(p for p in root.iterdir() if p.is_dir()) if root.exists() else []


def fmt(value) -> str:
    return value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)


def cite(c) -> str:
    c = c if isinstance(c, dict) else c.model_dump()
    return f"{c['file']} p.{c['page']}"


def show_trace(trace) -> None:
    rows = [{"agent": s.agent, "action": s.action, "status": s.status, "ms": s.duration_ms,
             "detail": json.dumps(s.detail, ensure_ascii=False)[:300]} for s in trace]
    st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)


def show_record(record: dict) -> None:
    v = record["validation"]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Accepted", v["accepted"])
    c2.metric("Not found", v["not_found"])
    c3.metric("Rejected", v["rejected"])
    c4.metric("Retries", v["retried"])
    st.write(record["fields"]["Summary"]["value"])
    rows = [{"field": k, "value": fmt(f["value"]), "status": f["status"], "confidence": f["confidence"],
             "sources": "; ".join(cite(s) for s in f["sources"]), "normalized": f.get("normalized", ""),
             "notes": f["notes"]} for k, f in record["fields"].items() if k != "Summary"]
    st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True,
                 column_config={"confidence": st.column_config.ProgressColumn(min_value=0, max_value=1)})
    if record["addendum_changes"]:
        st.subheader("Addendum changes")
        st.dataframe(pd.DataFrame([{"field": c["field"], "addendum": c["addendum_number"],
                                    "old": fmt(c["old_value"]), "new": fmt(c["new_value"]),
                                    "source": cite(c["source"])} for c in record["addendum_changes"]]),
                     width="stretch", hide_index=True)
    st.download_button("Download JSON", json.dumps(record, indent=2, ensure_ascii=False),
                       file_name=f"{record['bid_id']}.json", mime="application/json")


st.title("RFP Intelligence")
st.caption("Hybrid RAG search and multi-agent extraction over bid documents. "
           "Every value cites file and page; missing facts are \"Not found in documents\".")
extract_tab, search_tab, ask_tab, eval_tab = st.tabs(["Extract", "Search", "Ask", "Evaluation"])

with extract_tab:
    folders = bid_folders()
    col1, col2 = st.columns([3, 1])
    choice = col1.selectbox("Bid folder (data/bids)", [p.name for p in folders] + ["Other path..."])
    path = Path(col1.text_input("Folder path")) if choice == "Other path..." else ROOT / "data" / "bids" / choice
    force = col2.checkbox("Re-index unchanged files")
    run = col2.button("Run extraction", type="primary", disabled=not get_settings().llm_api_key)
    if not get_settings().llm_api_key:
        st.warning("Set LLM_API_KEY in .env to run extraction. Saved outputs are shown below.")
    if run:
        if not path.is_dir():
            st.error(f"Folder not found: {path}")
        else:
            from agents.orchestrator import Orchestrator
            with st.spinner(f"Indexing and extracting {path.name} (about a minute on the free Groq tier)..."):
                orch = Orchestrator(path)
                record = orch.run_extraction(force_index=force)
            indexed_bids.clear()
            st.session_state["trace"] = orch.state.trace
            st.success(f"{record['bid_id']} done")
    saved = ROOT / "outputs" / f"{path.name}.json"
    if saved.exists():
        show_record(json.loads(saved.read_text(encoding="utf-8")))
        trace_file = ROOT / "outputs" / f"{path.name}_trace.json"
        if trace_file.exists():
            with st.expander("Agent trace"):
                from agents.state import AgentStep
                show_trace([AgentStep(**s) for s in json.loads(trace_file.read_text(encoding="utf-8"))])

with search_tab:
    q = st.text_input("Query", placeholder="warranty requirements")
    c1, c2, c3, c4 = st.columns(4)
    bid = c1.selectbox("Bid", ["All"] + indexed_bids(), key="search_bid")
    types = c2.multiselect("Document types", DOC_TYPES)
    mode = c3.selectbox("Mode", MODES)
    k = c4.slider("Results", 1, 20, 5)
    if q:
        from search.engine import search
        filters = SearchFilters(bid_id=None if bid == "All" else bid, doc_types=types or None)
        for h in search(q, filters, k=k, mode=mode):
            ch = h.chunk
            with st.container(border=True):
                st.markdown(f"**#{h.rank}** · `{ch.file_name}` p.{ch.page_start} · {ch.doc_type}"
                            + (f" {ch.addendum_number}" if ch.addendum_number else "")
                            + f" · {ch.chunk_type} · score {h.score:.3f}")
                if ch.section:
                    st.caption(ch.section)
                st.text(ch.text[:1500])

with ask_tab:
    question = st.text_input("Question", placeholder="What is the deadline after all addendums?")
    bid = st.selectbox("Bid", ["All bids"] + indexed_bids(), key="ask_bid")
    if st.button("Ask", type="primary", disabled=not (question and get_settings().llm_api_key)):
        from agents.qa import answer_question
        with st.spinner("Retrieving and answering..."):
            ans = answer_question(question, SearchFilters(bid_id=bid) if bid != "All bids" else None)
        (st.success if ans.found else st.info)(ans.answer)
        for c, quote in zip(ans.citations, ans.quotes):
            with st.expander(cite(c)):
                st.text(quote)
        with st.expander("Agent trace"):
            show_trace(ans.trace)

with eval_tab:
    report = ROOT / "eval" / "report.md"
    qa_log = ROOT / "outputs" / "qa_log.md"
    st.markdown(report.read_text(encoding="utf-8") if report.exists() else "Run `python -m eval.run_eval`.")
    if qa_log.exists():
        with st.expander("Q&A log"):
            st.markdown(qa_log.read_text(encoding="utf-8"))
