"""
Insurance Document RAG — Streamlit UI
======================================
Full web interface for the P&C / Reinsurance Hybrid Search PoC.
Wraps the existing src/ pipeline with a clean, professional UI.

Run:
    streamlit run streamlit_app.py
"""

import os
import sys
import io
import math
import tempfile
import time
import re

# ── Path fix so src/ imports work regardless of CWD ──────────────────────
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import streamlit as st

# ── Page config (must be FIRST streamlit call) ────────────────────────────
st.set_page_config(
    page_title="Insurance RAG · Hybrid Search",
    page_icon="🏛",
    layout="wide",
    initial_sidebar_state="expanded",
)

from src.parser     import InsuranceDocumentParser
from src.chunker    import InsuranceChunker
from src.search_index import InsuranceHybridSearchIndex, EMBEDDER_OPTIONS
from src.rag_engine import InsuranceRAGEngine, SearchEvaluator, RAGResponse, GroqLLM, GROQ_MODELS
from src.sample_docs import SAMPLE_DOCS, get_sample_queries

# ─────────────────────────────────────────────────────────────────────────
# STYLING
# ─────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* ── Global font & background ── */
html, body, [class*="css"]  { font-family: 'Inter', 'Segoe UI', Arial, sans-serif; }

/* ── Sidebar ── */
section[data-testid="stSidebar"] { background: #0f1e3d; }
section[data-testid="stSidebar"] * { color: orange !important; }
section[data-testid="stSidebar"] .stSelectbox label,
section[data-testid="stSidebar"] .stSlider label,
section[data-testid="stSidebar"] .stCheckbox label { color: #a8c4e8 !important; font-size: 13px !important; }

/* ── Top header bar ── */
.rag-header {
    background: linear-gradient(135deg, #0f1e3d 0%, #1b3a6b 60%, #2e75b6 100%);
    padding: 22px 32px 18px 32px;
    border-radius: 12px;
    margin-bottom: 20px;
    border-left: 5px solid #1d9e75;
}
.rag-header h1 { color: #ffffff !important; font-size: 26px !important; margin: 0 0 4px 0; }
.rag-header p  { color: #a8d4f5 !important; font-size: 13px; margin: 0; }

/* ── Metric cards ── */
.metric-card {
    background: #f5f8ff;
    border: 1px solid #d0ddf5;
    border-left: 4px solid #2e75b6;
    border-radius: 8px;
    padding: 14px 18px;
    margin-bottom: 8px;
}
.metric-card .label { font-size: 11px; color: #666; text-transform: uppercase; letter-spacing: 0.5px; }
.metric-card .value { font-size: 22px; font-weight: 700; color: #1b3a6b; margin-top: 2px; }

/* ── Result cards ── */
.result-card {
    background: #ffffff;
    border: 1px solid #e0e8f5;
    border-left: 4px solid #2e75b6;
    border-radius: 10px;
    padding: 16px 20px;
    margin-bottom: 14px;
    box-shadow: 0 2px 8px rgba(30,58,107,0.06);
    transition: box-shadow 0.2s;
}
.result-card:hover { box-shadow: 0 4px 16px rgba(30,58,107,0.12); }
.result-card.rank-1 { border-left-color: #1d9e75; background: #f8fffc; }
.result-card .result-header {
    display: flex; justify-content: space-between; align-items: flex-start;
    margin-bottom: 8px; gap: 12px;
}
.result-card .doc-name  { font-weight: 600; color: #1b3a6b; font-size: 14px; }
.result-card .section   { color: #2e75b6; font-size: 13px; }
.result-card .snippet   { color: #333; font-size: 13px; line-height: 1.65; margin-top: 8px; }
.result-card .figures   { background: #fff8e6; border: 1px solid #f0d080;
                           border-radius: 6px; padding: 6px 10px; margin-top: 8px;
                           font-size: 12px; color: #7a5500; }
.score-badge {
    display: inline-block; background: #1b3a6b; color: #fff;
    border-radius: 20px; padding: 3px 10px; font-size: 12px; font-weight: 600;
    white-space: nowrap;
}
.score-row  { display: flex; gap: 8px; flex-wrap: wrap; margin-top: 8px; }
.score-chip {
    font-size: 11px; border-radius: 4px; padding: 2px 8px; font-weight: 500;
}
.chip-bm25   { background: #e8f0fb; color: #1b4a9e; }
.chip-vec    { background: #e6faf3; color: #0e6642; }
.chip-rrf    { background: #fff0e6; color: #8b3a00; }
.chip-rerank { background: #f0e6ff; color: #5a0099; }

/* ── Info boxes ── */
.info-box {
    border-radius: 8px; padding: 12px 16px; margin: 10px 0; font-size: 13px;
}
.info-green  { background: #eafaf3; border-left: 4px solid #1d9e75; color: #0e5432; }
.info-blue   { background: #e8f0fb; border-left: 4px solid #2e75b6; color: #1b3a6b; }
.info-orange { background: #fff4e6; border-left: 4px solid #e07b00; color: #7a3d00; }
.info-red    { background: #fdecea; border-left: 4px solid #c0392b; color: #7a1a12; }

/* ── Score progress bars ── */
.score-bar-wrap { background: #e8eef8; border-radius: 4px; height: 6px; margin: 4px 0 8px 0; }
.score-bar-fill { height: 6px; border-radius: 4px; }

/* ── Query input ── */
.stTextInput input {
    border: 2px solid #d0ddf5 !important; border-radius: 8px !important;
    font-size: 15px !important; padding: 10px 14px !important;
}
.stTextInput input:focus { border-color: #2e75b6 !important; box-shadow: 0 0 0 3px rgba(46,117,182,0.15) !important; }

/* ── Tabs ── */
.stTabs [data-baseweb="tab-list"] { background: #f0f4fa; border-radius: 8px; padding: 4px; }
.stTabs [data-baseweb="tab"]      { border-radius: 6px; font-weight: 500; }

/* ── Code blocks ── */
code { background: #1e2030 !important; color: #e8f4fd !important; border-radius: 4px !important; }

/* ── Expander ── */
.streamlit-expanderHeader { font-weight: 600; color: #1b3a6b; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────
# SESSION STATE INITIALISATION
# ─────────────────────────────────────────────────────────────────────────
def init_state():
    defaults = {
        "engine":         None,
        "index":          None,
        "indexed_docs":   [],
        "query_history":  [],
        "last_response":  None,
        "eval_results":   None,
        "building":       False,
        "last_elapsed_ms": 0,
        "groq_api_key":   "",
        "groq_enabled":   False,
        "embedder_type":  "tfidf",
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

init_state()


# ─────────────────────────────────────────────────────────────────────────
# DOCUMENT TYPE DETECTION
# ─────────────────────────────────────────────────────────────────────────
DOC_TYPE_HINTS = {
    "policy":      ("Property & Casualty", "Policy"),
    "treaty":      ("Reinsurance",         "Treaty"),
    "claims":      ("Property & Casualty", "Claims"),
    "sop":         ("Property & Casualty", "SOP"),
    "endorsement": ("Property & Casualty", "Endorsement"),
    "compliance":  ("Compliance",          "Regulatory"),
    "underwriting":("Reinsurance",         "Underwriting"),
}

def detect_doc_type(name: str):
    nl = name.lower()
    for kw, (lob, cat) in DOC_TYPE_HINTS.items():
        if kw in nl:
            return lob, cat
    return "Property & Casualty", "Policy"


# ─────────────────────────────────────────────────────────────────────────
# PIPELINE BUILDER  (cached to avoid re-indexing on every rerun)
# ─────────────────────────────────────────────────────────────────────────
def build_pipeline(
    use_demo: bool,
    pdf_bytes_list: list,
    use_reranker: bool,
    top_k: int,
    embedder_type: str = "tfidf",
    llm_fn=None,
):
    """Parse, chunk, index all documents. Returns (engine, index, indexed_docs)."""
    parser = InsuranceDocumentParser(verbose=False)
    index  = InsuranceHybridSearchIndex(
        use_reranker=use_reranker,
        verbose=False,
        embedder_type=embedder_type,
    )
    all_chunks   = []
    indexed_docs = []

    if use_demo:
        for doc_name, doc_text in SAMPLE_DOCS.items():
            lob, cat = detect_doc_type(doc_name)
            parsed = parser.parse(doc_text)
            parsed.doc_name = doc_name
            chunker = InsuranceChunker(lob=lob, doc_category=cat, verbose=False)
            chunks  = chunker.chunk(parsed)
            all_chunks.extend(chunks)
            n_child = sum(1 for c in chunks if c.chunk_type == "child")
            indexed_docs.append({
                "name": doc_name, "lob": lob, "category": cat,
                "chunks": n_child, "source": "demo"
            })

    for name, data in pdf_bytes_list:
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as f:
            f.write(data)
            tmp_path = f.name
        try:
            lob, cat = detect_doc_type(name)
            parsed  = parser.parse(tmp_path)
            chunker = InsuranceChunker(lob=lob, doc_category=cat, verbose=False)
            chunks  = chunker.chunk(parsed)
            all_chunks.extend(chunks)
            n_child = sum(1 for c in chunks if c.chunk_type == "child")
            indexed_docs.append({
                "name": os.path.splitext(name)[0], "lob": lob,
                "category": cat, "chunks": n_child, "source": "upload"
            })
        finally:
            os.unlink(tmp_path)

    if all_chunks:
        index.add_chunks(all_chunks)

    engine = InsuranceRAGEngine(
        search_index=index,
        top_k=top_k,
        use_parent_context=True,
        context_only=(llm_fn is None),
        llm_fn=llm_fn,
        verbose=False,
    )
    return engine, index, indexed_docs


# ─────────────────────────────────────────────────────────────────────────
# RESULT CARD RENDERER
# ─────────────────────────────────────────────────────────────────────────
def render_result_card(r, rank: int):
    card_class = "result-card rank-1" if rank == 1 else "result-card"
    rank_icon  = "🥇" if rank == 1 else f"#{rank}"

    # Scores
    bm25_pct   = min(100, int(r.bm25_score  * 100 / max(r.bm25_score, 1)))
    vec_pct    = min(100, int(r.vector_score * 100))
    rerank_pct = min(100, max(0, int((r.rerank_score or 0) / 10 * 100))) if r.rerank_score else None
    final_pct  = min(100, int(r.final_score * 100 / max(r.final_score, 0.001)))

    snippet = r.chunk.raw_text.replace("\n", " ").strip()
    snippet = snippet[:420] + "…" if len(snippet) > 420 else snippet

    figs_html = ""
    if r.chunk.numeric_values:
        figs = ", ".join(r.chunk.numeric_values[:6])
        figs_html = f'<div class="figures">💰 Key figures: {figs}</div>'

    rerank_chip = ""
    if r.rerank_score is not None:
        rerank_chip = f'<span class="score-chip chip-rerank">Rerank {r.rerank_score:.3f}</span>'

    st.markdown(f"""
<div class="{card_class}">
  <div class="result-header">
    <div>
      <div class="doc-name">{rank_icon}  {r.chunk.doc_name}</div>
      <div class="section">📂 {r.chunk.section_title} &nbsp;·&nbsp; Page {r.chunk.page_num}</div>
    </div>
    <div>
      <span class="score-badge">Score {r.final_score:.4f}</span>
    </div>
  </div>
  <div class="score-row">
    <span class="score-chip chip-bm25">BM25 {r.bm25_score:.3f}</span>
    <span class="score-chip chip-vec">Cosine {r.vector_score:.3f}</span>
    <span class="score-chip chip-rrf">RRF {r.rrf_score:.4f}</span>
    {rerank_chip}
  </div>
  <div class="score-bar-wrap"><div class="score-bar-fill" style="width:{final_pct}%;background:#2e75b6;"></div></div>
  <div class="snippet">{snippet}</div>
  {figs_html}
</div>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────
# GUARDRAIL PANEL
# ─────────────────────────────────────────────────────────────────────────
def render_guardrail_panel(resp: RAGResponse):
    gs = resp.groundedness_score
    icon  = "✅" if resp.guardrail_passed else "⚠️"
    color = "info-green" if resp.guardrail_passed else "info-orange"
    status = "PASSED" if resp.guardrail_passed else "WARNINGS RAISED"

    st.markdown(f"""
<div class="info-box {color}">
  <strong>{icon} Guardrail: {status}</strong><br>
  📊 Groundedness score: <strong>{gs:.1%}</strong>
  {"".join(f"<br>• {w}" for w in resp.guardrail_warnings)}
</div>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🏛 Insurance RAG")
    st.markdown("<hr style='border-color:#2e5080;margin:8px 0'>", unsafe_allow_html=True)

    st.markdown("#### 📂 Document Sources")
    use_demo = st.checkbox("Use built-in sample docs", value=True,
                           help="Loads 3 sample documents: Property Policy, Reinsurance Treaty, Claims SOP")
    uploaded = st.file_uploader("Upload PDFs", type=["pdf"],
                                accept_multiple_files=True,
                                help="Upload your own P&C or Reinsurance PDFs")

    st.markdown("<hr style='border-color:#2e5080;margin:8px 0'>", unsafe_allow_html=True)
    st.markdown("#### 🧠 Embedding Model")

    embedder_choice = st.selectbox(
        "Embedding engine",
        options=list(EMBEDDER_OPTIONS.keys()),
        index=0,
        format_func=lambda k: {
            "tfidf":   "TF-IDF (offline, no GPU)",
            "bge":     "BGE-large-en-v1.5 (neural)",
            "qwen3vl": "Qwen3-VL-8B (best for PDFs)",
        }.get(k, k),
        help="TF-IDF works offline. BGE and Qwen3-VL need HuggingFace model download.",
    )
    if embedder_choice == "qwen3vl":
        st.caption("⚠️ Requires ~16 GB download + GPU recommended")
    elif embedder_choice == "bge":
        st.caption("⚠️ Requires HuggingFace model download (~1.3 GB)")

    st.markdown("<hr style='border-color:#2e5080;margin:8px 0'>", unsafe_allow_html=True)
    st.markdown("#### ⚙️ Search Settings")

    top_k = st.slider("Results per query", min_value=1, max_value=10, value=5)
    use_reranker = st.checkbox("Enable reranker", value=True,
                               help="BM25 position-weighted reranker improves precision")
    bm25_w = st.slider("BM25 weight (α)", 0.0, 1.0, 0.40, 0.05,
                       help="Weight for keyword matching. 1-α = semantic weight.")
    vec_w  = round(1.0 - bm25_w, 2)
    st.caption(f"Semantic weight (β) = **{vec_w}**")

    st.markdown("<hr style='border-color:#2e5080;margin:8px 0'>", unsafe_allow_html=True)
    st.markdown("#### 🤖 LLM — ChatGroq")

    groq_enabled = st.checkbox(
        "Enable Groq answer generation",
        value=False,
        help="When enabled, retrieved passages are sent to Groq to generate a full answer.",
    )
    groq_api_key = ""
    groq_model   = GroqLLM.DEFAULT_MODEL
    if groq_enabled:
        groq_api_key = st.text_input(
            "Groq API key",
            type="password",
            placeholder="gsk_...",
            help="Get a free key at https://console.groq.com",
        )
        groq_model = st.selectbox(
            "Groq model",
            options=list(GROQ_MODELS.keys()),
            index=0,
            format_func=lambda k: GROQ_MODELS.get(k, k).split(" — ")[0],
        )
        if groq_api_key:
            st.caption(f"Model: `{groq_model}`")
        else:
            st.warning("Enter your Groq API key above to activate LLM answers.")

    st.markdown("<hr style='border-color:#2e5080;margin:8px 0'>", unsafe_allow_html=True)
    st.markdown("#### 🔍 Query Filter")
    doc_filter_opt = st.text_input("Filter by document name", placeholder="e.g. reinsurance_treaty")
    lob_filter_opt = st.selectbox("Filter by Line of Business",
                                  ["(all)", "Property & Casualty", "Reinsurance", "Compliance"])
    lob_filter = None if lob_filter_opt == "(all)" else lob_filter_opt

    st.markdown("<hr style='border-color:#2e5080;margin:8px 0'>", unsafe_allow_html=True)

    build_btn = st.button("🚀 Build / Rebuild Index", use_container_width=True, type="primary")

    if st.session_state.engine:
        stats = st.session_state.index.stats()
        st.markdown("<hr style='border-color:#2e5080;margin:8px 0'>", unsafe_allow_html=True)
        st.markdown("#### 📊 Index Stats")
        st.caption(f"Child chunks: **{stats['total_child_chunks']}**")
        st.caption(f"Parent sections: **{stats['total_parent_sections']}**")
        st.caption(f"Engine: **{stats['embedding_engine']}**")
        st.caption(f"Embedder type: **{stats.get('embedder_type','tfidf')}**")
        st.caption(f"Vector dim: **{stats['vector_dimensions']:,}**")
        st.caption(f"Reranker: **{stats['reranker']}**")
        st.caption(f"BM25 k1=1.5, b=0.6 | RRF k={stats['rrf_k']}")


# ─────────────────────────────────────────────────────────────────────────
# INDEX BUILD
# ─────────────────────────────────────────────────────────────────────────
if build_btn:
    pdf_bytes_list = [(f.name, f.read()) for f in uploaded] if uploaded else []
    if not use_demo and not pdf_bytes_list:
        st.sidebar.error("Select sample docs or upload at least one PDF.")
    else:
        # Build Groq LLM if enabled and key provided
        llm_fn = None
        if groq_enabled and groq_api_key.strip():
            try:
                llm_fn = GroqLLM(api_key=groq_api_key.strip(), model=groq_model)
            except Exception as e:
                st.sidebar.error(f"Groq init failed: {e}")
                llm_fn = None

        spinner_msg = "⚙️ Parsing, chunking and indexing documents"
        if embedder_choice == "qwen3vl":
            spinner_msg += " (Qwen3-VL encoding — this may take a few minutes)…"
        elif embedder_choice == "bge":
            spinner_msg += " (BGE neural encoding)…"
        else:
            spinner_msg += "…"

        with st.spinner(spinner_msg):
            t0 = time.time()
            engine, index, indexed_docs = build_pipeline(
                use_demo, pdf_bytes_list, use_reranker, top_k,
                embedder_type=embedder_choice,
                llm_fn=llm_fn,
            )
            # Apply custom weights
            index.BM25_WEIGHT   = bm25_w
            index.VECTOR_WEIGHT = vec_w
            elapsed = time.time() - t0

        st.session_state.engine       = engine
        st.session_state.index        = index
        st.session_state.indexed_docs = indexed_docs
        st.session_state.eval_results = None
        st.session_state.query_history= []

        st.sidebar.success(f"✅ Index built in {elapsed:.1f}s — {len(indexed_docs)} document(s) loaded")
        st.rerun()


# ─────────────────────────────────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="rag-header">
  <h1>🏛 Insurance Document RAG — Hybrid Search</h1>
  <p>BM25 · Hybrid Cosine Search · RRF Fusion · Reranker &nbsp;|&nbsp; Embeddings: TF-IDF / BGE / Qwen3-VL &nbsp;|&nbsp; LLM: ChatGroq &nbsp;|&nbsp; P&amp;C / Reinsurance</p>
</div>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────
# NO INDEX YET — WELCOME SCREEN
# ─────────────────────────────────────────────────────────────────────────
if not st.session_state.engine:
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("""
<div class="info-box info-blue">
  <strong>🚀 Quick Start</strong><br>
  1. Tick <em>Use built-in sample docs</em> in the sidebar<br>
  2. Click <strong>Build / Rebuild Index</strong><br>
  3. Type a query below and press Enter
</div>
""", unsafe_allow_html=True)
    with col2:
        st.markdown("""
<div class="info-box info-green">
  <strong>📄 Your own PDFs</strong><br>
  Upload any P&C or Reinsurance PDF from the sidebar.
  Name files with keywords like <em>policy</em>, <em>treaty</em>, <em>claims</em>
  for auto LOB tagging.
</div>
""", unsafe_allow_html=True)
    with col3:
        st.markdown("""
<div class="info-box info-orange">
  <strong>⚡ Architecture</strong><br>
  Phase 1: Layout-aware PDF parsing<br>
  Phase 2: Clause-boundary chunking<br>
  Phase 3: BM25 + cosine + RRF fusion<br>
  Phase 4: Parent context expansion<br>
  Phase 5: Numerical audit guardrails
</div>
""", unsafe_allow_html=True)

    st.markdown("---")
    st.subheader("Example queries you can ask after indexing")
    examples = [
        ("🔢 Deductible lookup", "What is the deductible for flood damage?"),
        ("🛡 Coverage check",   "Does this policy cover cyber-induced business interruption?"),
        ("📋 Treaty structure", "What is the attachment point for Layer 1 reinsurance?"),
        ("⏱ Claims SLA",       "How quickly must a large claim be notified?"),
        ("💰 Limit & capacity", "What is the maximum reinsurance liability per occurrence?"),
        ("👤 Authority",        "Who has authority to handle claims above $2 million?"),
    ]
    cols = st.columns(3)
    for i, (label, q) in enumerate(examples):
        cols[i % 3].markdown(f"**{label}**  \n`{q}`")

    st.stop()


# ─────────────────────────────────────────────────────────────────────────
# MAIN TABS
# ─────────────────────────────────────────────────────────────────────────
#tab_search, tab_history, tab_eval, tab_docs = st.tabs([
#    "🔍 Search", "📜 History", "📊 Evaluation", "📁 Indexed Documents"
#])
tab_search, tab_history, tab_docs = st.tabs([
    "🔍 Search", "📜 History", "📁 Indexed Documents"
])

# ══════════════════════════════════════════════════════════════════════════
# TAB 1 — SEARCH
# ══════════════════════════════════════════════════════════════════════════
with tab_search:
    # ── Suggested queries ──────────────────────────────────────────────
    st.markdown("**💡 Suggested queries** — click to fill:")
    suggest_queries = [
        "What is the deductible for flood damage?",
        "Does this policy cover cyber attacks?",
        "What is the attachment point for Layer 1?",
        "Who handles claims above $2 million?",
        "What is the maximum reinsurance liability?",
        "Basis of settlement for building losses",
    ]
    cols = st.columns(3)
    for i, sq in enumerate(suggest_queries):
        if cols[i % 3].button(sq, key=f"sug_{i}", use_container_width=True):
            st.session_state["prefill_query"] = sq

    st.markdown("---")

    # ── Query input ───────────────────────────────────────────────────
    prefill = st.session_state.pop("prefill_query", "")
    query = st.text_input(
        "Enter your insurance query",
        value=prefill,
        placeholder="e.g. What is the deductible for flood damage?",
        label_visibility="collapsed",
    )

    col_search, col_clear = st.columns([5, 1])
    search_clicked = col_search.button("🔍 Search", type="primary", use_container_width=True)
    if col_clear.button("Clear", use_container_width=True):
        st.session_state.last_response = None
        st.rerun()

    # ── Run search ─────────────────────────────────────────────────────
    if (search_clicked or query) and query.strip():
        doc_filter = doc_filter_opt.strip() or None
        lob_filter_val = None if lob_filter_opt == "(all)" else lob_filter_opt

        with st.spinner("🔎 Searching…"):
            t0 = time.time()
            resp = st.session_state.engine.query(
                query,
                doc_filter=doc_filter,
                lob_filter=lob_filter_val,
            )
            elapsed_ms = (time.time() - t0) * 1000

        st.session_state.last_response = resp
        st.session_state.last_elapsed_ms = elapsed_ms
        st.session_state.query_history.insert(0, {
            "query": query, "response": resp, "ms": elapsed_ms
        })

    # ── Display results ─────────────────────────────────────────────────
    resp = st.session_state.last_response
    if resp:
        n = len(resp.source_chunks)
        elapsed_ms = st.session_state.get("last_elapsed_ms", 0)
        col_a, col_b, col_c, col_d = st.columns(4)
        col_a.markdown(f"""<div class="metric-card"><div class="label">Results</div><div class="value">{n}</div></div>""", unsafe_allow_html=True)
        col_b.markdown(f"""<div class="metric-card"><div class="label">Latency</div><div class="value">{elapsed_ms:.0f} ms</div></div>""", unsafe_allow_html=True)
        col_c.markdown(f"""<div class="metric-card"><div class="label">Groundedness</div><div class="value">{resp.groundedness_score:.1%}</div></div>""", unsafe_allow_html=True)
        col_d.markdown(f"""<div class="metric-card"><div class="label">Guardrail</div><div class="value">{"✅" if resp.guardrail_passed else "⚠️"}</div></div>""", unsafe_allow_html=True)

        st.markdown("---")

        # ── LLM answer (Groq) — show when available ────────────────────
        if resp.answer and not resp.answer.startswith("Query:"):
            st.markdown("### 🤖 Groq Answer")
            guardrail_color = "success" if resp.guardrail_passed else "warning"
            with st.container():
                st.markdown(
                    f"""<div style="background:var(--background-color);border:1px solid {'#1d9e75' if resp.guardrail_passed else '#e07b00'};
                    border-left:4px solid {'#1d9e75' if resp.guardrail_passed else '#e07b00'};
                    border-radius:8px;padding:16px 20px;margin-bottom:12px;font-size:14px;line-height:1.7">
                    {resp.answer.replace(chr(10), '<br>')}
                    </div>""",
                    unsafe_allow_html=True,
                )
            if resp.guardrail_warnings:
                for w in resp.guardrail_warnings:
                    st.warning(w)
            if resp.numerical_audit.get("unverified"):
                st.warning(f"🔢 Unverified figures in answer: "
                           f"{', '.join(resp.numerical_audit['unverified'][:5])}")
            st.divider()

        # Score breakdown expander
        with st.expander("📐 Score breakdown — how the hybrid ranking works", expanded=False):
            st.markdown("""
| Component | Role | Weight |
|---|---|---|
| **BM25 (keyword)** | Exact term frequency matching — great for policy numbers, clause IDs | α |
| **TF-IDF cosine (semantic)** | Vector similarity — captures synonyms and concept overlap | β |
| **RRF fusion** | Merges both ranked lists by rank position — scale invariant | — |
| **Reranker** | Position-weighted term overlap re-scoring on top-N candidates | final sort |
""")
            if resp.source_chunks:
                import pandas as pd
                rows = []
                for i, r in enumerate(resp.source_chunks):
                    rows.append({
                        "Rank": i + 1,
                        "Document": r.chunk.doc_name,
                        "Section": r.chunk.section_title[:40],
                        "BM25": round(r.bm25_score, 4),
                        "Cosine": round(r.vector_score, 4),
                        "RRF": round(r.rrf_score, 5),
                        "Rerank": round(r.rerank_score, 3) if r.rerank_score else "—",
                        "Final": round(r.final_score, 4),
                    })
                st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

        # Result cards
        st.markdown(f"### 📄 Top {n} results for: *{resp.query}*")
        for i, r in enumerate(resp.source_chunks):
            render_result_card(r, i + 1)

        # Parent context expander
        if resp.parent_contexts:
            with st.expander(f"📖 Parent context sections ({len(resp.parent_contexts)} loaded for LLM)", expanded=False):
                st.markdown("*These full sections would be injected into an LLM prompt for answer generation:*")
                for pc in resp.parent_contexts:
                    st.markdown(f"**{pc.doc_name} · {pc.section_title}**")
                    st.text_area("", value=pc.raw_text[:1000] + ("…" if len(pc.raw_text) > 1000 else ""),
                                 height=140, disabled=True, label_visibility="collapsed",
                                 key=f"pc_{pc.chunk_id}")

        # Guardrail
        render_guardrail_panel(resp)

        # LLM prompt preview
        with st.expander("🔧 LLM prompt preview (connect your own LLM)", expanded=False):
            st.markdown("""
> **To enable full answer generation**, set `context_only=False` and pass an `llm_fn` callable to `InsuranceRAGEngine`.
> Below is the exact prompt that would be sent:
""")
            if resp.source_chunks:
                context_preview = "\n\n---\n\n".join(
                    f"[Source {i+1}: {r.chunk.doc_name}, Section: {r.chunk.section_title}]\n{r.chunk.raw_text[:300]}…"
                    for i, r in enumerate(resp.source_chunks)
                )
                prompt_preview = f"""You are an expert underwriter and claims auditor...

CONTEXT:
---
{context_preview}
---

USER QUERY: {resp.query}

ANSWER (cite sections explicitly):"""
                st.code(prompt_preview, language="text")


# ══════════════════════════════════════════════════════════════════════════
# TAB 2 — QUERY HISTORY
# ══════════════════════════════════════════════════════════════════════════
with tab_history:
    st.markdown("### 📜 Query History")
    history = st.session_state.query_history

    if not history:
        st.info("No queries yet. Run a search in the Search tab.")
    else:
        if st.button("🗑 Clear history"):
            st.session_state.query_history = []
            st.rerun()

        for i, item in enumerate(history):
            r = item["response"]
            n = len(r.source_chunks)
            top_doc  = r.source_chunks[0].chunk.doc_name if n else "—"
            top_sec  = r.source_chunks[0].chunk.section_title[:35] if n else "—"
            gs = r.groundedness_score

            with st.expander(f"**Q{len(history)-i}** — {item['query'][:70]}  ·  {item['ms']:.0f}ms  ·  {n} results", expanded=i == 0):
                col1, col2, col3 = st.columns(3)
                col1.metric("Results", n)
                col2.metric("Groundedness", f"{gs:.1%}")
                col3.metric("Guardrail", "✅ Pass" if r.guardrail_passed else "⚠️ Warn")

                st.markdown(f"**Top result:** {top_doc} · {top_sec}")
                if n:
                    r0 = r.source_chunks[0]
                    st.markdown(f"> {r0.chunk.raw_text[:350].replace(chr(10), ' ')}…")

                if st.button("Re-run this query", key=f"rerun_{i}"):
                    st.session_state["prefill_query"] = item["query"]
                    st.rerun()


# ══════════════════════════════════════════════════════════════════════════
# TAB 3 — EVALUATION
# ══════════════════════════════════════════════════════════════════════════
#with tab_eval:
#    st.markdown("### 📊 Search Quality Evaluation")
#    st.markdown("Runs the built-in labelled query set (8 queries with ground-truth sections) and reports nDCG, MRR, Precision and Recall.")
#
#    col_run, col_info = st.columns([2, 3])
#    with col_run:
#        run_eval = st.button("▶ Run Evaluation Suite", type="primary", use_container_width=True)
#
#    with col_info:
#        st.markdown("""
#<div class="info-box info-blue">
#Evaluation uses the <code>get_sample_queries()</code> function from <code>src/sample_docs.py</code>.
#Add your own labelled queries there for custom evaluation.
#</div>
#""", unsafe_allow_html=True)
#
#    if run_eval:
#        eval_set = get_sample_queries()
#        with st.spinner(f"Running {len(eval_set)} evaluation queries…"):
#            results = SearchEvaluator.evaluate(st.session_state.engine, eval_set)
#        st.session_state.eval_results = results
#        st.success("Evaluation complete!")
#
#    if st.session_state.eval_results:
#        ev = st.session_state.eval_results
#        st.markdown("#### Summary Metrics")
#
#        targets = {"precision_at_k": 0.85, "recall_at_k": 0.80, "mrr": 0.88, "ndcg_at_k": 0.82}
#        labels  = {"precision_at_k": "Precision@k", "recall_at_k": "Recall@k", "mrr": "MRR", "ndcg_at_k": "nDCG@k"}
#        descs   = {
#            "precision_at_k": "Are all top-k results relevant?",
#            "recall_at_k":    "Were all relevant sections found?",
#            "mrr":            "Is the best result near the top?",
#            "ndcg_at_k":      "Overall ranked retrieval quality",
#        }
#
#        cols = st.columns(4)
#        for i, (key, label) in enumerate(labels.items()):
#            score  = ev[key]
#            target = targets[key]
#            delta  = score - target
#            icon   = "✅" if score >= target else "⚠️"
#            cols[i].metric(
#                label=f"{icon} {label}",
#                value=f"{score:.4f}",
#                delta=f"{delta:+.4f} vs target {target}",
#                delta_color="normal",
#                help=descs[key],
#            )
#
#        st.markdown("#### Per-Query Breakdown")
#        import pandas as pd
#        pq = ev.get("per_query", [])
#        df = pd.DataFrame(pq)
#        df.columns = [c.replace("_", " ").title() for c in df.columns]
#        st.dataframe(df, use_container_width=True, hide_index=True)
#
#        st.markdown("""
#<div class="info-box info-orange">
#<strong>🔁 Improve scores:</strong> Switch from TF-IDF to <code>BAAI/bge-large-en-v1.5</code> neural embeddings
#for significant accuracy gains (expected nDCG ~0.87 vs current ~0.58 baseline).
#See <code>src/search_index.py</code> — replace <code>TFIDFEmbedder()</code> with <code>SentenceTransformerEmbedder(...)</code>.
#</div>
#""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════
# TAB 4 — INDEXED DOCUMENTS
# ══════════════════════════════════════════════════════════════════════════
with tab_docs:
    st.markdown("### 📁 Indexed Document Corpus")

    docs = st.session_state.indexed_docs
    if not docs:
        st.info("No documents indexed yet.")
    else:
        total_chunks = sum(d["chunks"] for d in docs)
        c1, c2, c3 = st.columns(3)
        c1.metric("Documents", len(docs))
        c2.metric("Child chunks", total_chunks)
        c3.metric("Parent sections", st.session_state.index.stats()["total_parent_sections"])

        st.markdown("---")
        for doc in docs:
            src_badge = "📤 Uploaded" if doc["source"] == "upload" else "🎛 Demo"
            lob_color = {
                "Reinsurance": "#6b3a9e",
                "Property & Casualty": "#1b3a6b",
                "Compliance": "#7a3d00",
            }.get(doc["lob"], "#333")

            st.markdown(f"""
<div class="result-card">
  <div class="result-header">
    <div>
      <div class="doc-name">📄 {doc['name']}</div>
      <div class="section" style="color:{lob_color}">
        {doc['lob']} &nbsp;·&nbsp; {doc['category']}
      </div>
    </div>
    <div style="text-align:right">
      <span class="score-badge">{doc['chunks']} chunks</span><br>
      <small style="color:#888;font-size:11px;margin-top:4px;display:block">{src_badge}</small>
    </div>
  </div>
</div>
""", unsafe_allow_html=True)

        st.markdown("---")
        st.markdown("#### 📝 Add a Custom Query to Evaluation")
        with st.expander("Custom evaluation query builder"):
            new_query = st.text_input("Query text", placeholder="What is the flood deductible?")
            new_sections = st.text_input("Relevant sections (comma-separated)",
                                         placeholder="SECTION 2: DEFINITIONS, SECTION 5: CONDITIONS")
            if st.button("Add to eval set") and new_query:
                if "custom_eval" not in st.session_state:
                    st.session_state.custom_eval = []
                secs = [s.strip() for s in new_sections.split(",") if s.strip()]
                st.session_state.custom_eval.append({"query": new_query, "relevant_sections": secs})
                st.success(f"Added: {new_query}")

            if st.session_state.get("custom_eval"):
                st.markdown("**Custom queries added:**")
                for item in st.session_state.custom_eval:
                    st.caption(f"• {item['query']} → {item['relevant_sections']}")
