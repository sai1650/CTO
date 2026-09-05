"""Streamlit user interface for Hindi document QA."""

import re

import streamlit as st

from src.config import get_settings
from src.embeddings import EmbeddingModel
from src.errors import LLMError
from src.llm import build_llm
from src.rag_pipeline import RAGPipeline
from src.retriever import Retriever
from src.vector_store import ChromaVectorStore

SAMPLE_QUESTIONS = [
    ("Wheat soil", "गेहूं की खेती के लिए कौन सी मिट्टी उपयुक्त है?"),
    ("Rice climate", "What climate is suitable for rice cultivation?"),
    (
        "Modern farming technology",
        "आधुनिक कृषि में ड्रोन का उपयोग क्यों किया जाता है?",
    ),
    (
        "Sustainable agriculture",
        "What is the purpose of sustainable agriculture?",
    ),
]


def set_question(value: str) -> None:
    """Populate the question widget before the next script rerun."""
    st.session_state.question = value


def clean_visible_answer(answer: str) -> str:
    """Hide internal chunk citations from the user-facing answer."""
    return re.sub(r"\[(?:\s*chunk-\d+\s*,?)+\]", "", answer).strip()


@st.cache_resource
def build_store():
    settings = get_settings()
    embeddings = EmbeddingModel(settings.embedding_model)
    return ChromaVectorStore(
        settings.chroma_persist_directory,
        settings.chroma_collection_name,
        embeddings,
    )


@st.cache_resource
def build_pipeline():
    settings = get_settings()
    store = build_store()
    retriever = Retriever(store, settings.top_k, settings.relevance_threshold)
    return RAGPipeline(retriever, build_llm(settings))


st.set_page_config(
    page_title="HindiRAG | Hindi Document Intelligence",
    page_icon=":material/auto_awesome:",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    :root {
        --ink: #17211b;
        --muted: #6b756d;
        --line: #dfe6df;
        --paper: #f7f9f6;
        --surface: #ffffff;
        --accent: #286b4b;
        --accent-dark: #1d5138;
        --accent-soft: #e8f2eb;
    }
    .stApp { background: var(--paper); color: var(--ink); }
    [data-testid="stHeader"] { background: transparent; height: 0; }
    [data-testid="stToolbar"] { display: none; }
    #MainMenu { visibility: hidden; }
    .stDeployButton { display: none; }
    [data-testid="stMainBlockContainer"] {
        max-width: 1120px;
        padding-top: 1.25rem;
        padding-bottom: 2rem;
    }
    [data-testid="stSidebar"] {
        border-right: 1px solid var(--line);
        background: #f3f7f2;
    }
    [data-testid="stSidebar"] .block-container { padding-top: 2rem; }
    .topbar {
        display: flex;
        align-items: center;
        justify-content: space-between;
        gap: 1rem;
        padding-bottom: 1.1rem;
        border-bottom: 1px solid var(--line);
    }
    .brand { display: flex; align-items: center; gap: .8rem; }
    .brand-mark {
        display: grid;
        place-items: center;
        width: 2.35rem;
        height: 2.35rem;
        border-radius: 9px;
        background: var(--accent);
        color: white;
        font-weight: 800;
        letter-spacing: -.04em;
    }
    .brand-name { font-size: 1.05rem; font-weight: 800; line-height: 1.1; }
    .brand-subtitle {
        color: var(--muted); font-size: .78rem; margin-top: .18rem;
    }
    .topbar-right {
        display: flex;
        align-items: center;
        gap: .55rem;
        flex-wrap: wrap;
        justify-content: flex-end;
    }
    .online {
        display: inline-flex;
        align-items: center;
        gap: .42rem;
        color: var(--accent-dark);
        font-size: .78rem;
        font-weight: 700;
        margin-right: .35rem;
    }
    .online-dot {
        width: .45rem; height: .45rem; border-radius: 50%; background: #36a269;
    }
    .tech-badge {
        border: 1px solid var(--line);
        border-radius: 999px;
        background: var(--surface);
        color: #4e5b52;
        padding: .34rem .68rem;
        font-size: .72rem;
        font-weight: 700;
    }
    .hero { padding: 2.25rem 0 1.6rem; max-width: 760px; }
    .eyebrow {
        color: var(--accent);
        font-size: .7rem;
        font-weight: 800;
        letter-spacing: .14em;
        text-transform: uppercase;
        margin-bottom: .85rem;
    }
    .hero h1 {
        color: var(--ink);
        font-size: clamp(2rem, 3.5vw, 3rem);
        line-height: 1.08;
        letter-spacing: -.045em;
        margin: 0 0 .8rem;
    }
    .hero p {
        color: var(--muted);
        font-size: 1.05rem;
        line-height: 1.6;
        margin: 0;
        max-width: 650px;
    }
    .section-label {
        color: var(--ink); font-size: 1rem; font-weight: 750;
        margin: .2rem 0 .7rem;
    }
    .input-shell {
        background: var(--surface);
        border: 1px solid var(--line);
        border-radius: 14px;
        padding: 1rem;
        box-shadow: 0 8px 24px rgba(31, 55, 39, .05);
    }
    .input-shell [data-testid="stTextArea"] textarea {
        border-color: #ccd8cd;
        border-radius: 10px;
        background: #fbfdfb;
        min-height: 112px;
        font-size: 1rem;
    }
    .input-shell [data-testid="stTextArea"] textarea:focus {
        border-color: var(--accent);
        box-shadow: 0 0 0 1px var(--accent);
    }
    .sample-label {
        color: var(--muted); font-size: .78rem; font-weight: 700;
        margin: 1.35rem 0 .55rem;
    }
    .sample-row [data-testid="stButton"] button {
        min-height: 2.65rem;
        border: 1px solid var(--line);
        border-radius: 9px;
        background: var(--surface);
        color: #3d4a41;
        font-size: .78rem;
        line-height: 1.25;
        padding: .55rem .7rem;
        transition: border-color .15s ease, background .15s ease;
    }
    .sample-row [data-testid="stButton"] button:hover {
        border-color: var(--accent);
        background: var(--accent-soft);
        color: var(--accent-dark);
    }
    .ask-button button,
    [data-testid="stBaseButton-primary"] {
        border-radius: 9px;
        background: var(--accent);
        border: 1px solid var(--accent);
        font-weight: 750;
        min-height: 2.8rem;
    }
    .ask-button button:hover,
    [data-testid="stBaseButton-primary"]:hover {
        background: var(--accent-dark); border-color: var(--accent-dark);
    }
    .answer-heading { margin-top: 2.4rem; }
    .answer-heading h2, .sources-heading h2 {
        font-size: .75rem;
        letter-spacing: .12em;
        text-transform: uppercase;
    }
    .answer-heading h2 { margin: 0 0 .7rem; }
    .answer-box {
        background: var(--surface);
        border: 1px solid var(--line);
        border-left: 4px solid var(--accent);
        border-radius: 4px 12px 12px 4px;
        padding: 1.1rem 1.3rem;
        line-height: 1.7;
        box-shadow: 0 8px 24px rgba(31, 55, 39, .04);
    }
    .sources-heading { margin-top: 2rem; }
    .sources-heading h2 { font-size: 1.08rem; margin: 0 0 .25rem; }
    .sources-heading p {
        color: var(--muted); font-size: .82rem; margin: 0 0 .75rem;
    }
    [data-testid="stExpander"] {
        border: 1px solid var(--line); border-radius: 9px;
        background: var(--surface);
    }
    [data-testid="stExpander"] summary p {
        font-size: .82rem; font-weight: 700;
    }
    .footer { color: #849087; font-size: .72rem; padding-top: 1.5rem; }
    @media (max-width: 700px) {
        [data-testid="stMainBlockContainer"] { padding: 1rem 1rem 2rem; }
        .topbar { align-items: flex-start; flex-direction: column; }
        .topbar-right { justify-content: flex-start; }
        .hero { padding: 2.25rem 0 1.5rem; }
        .hero h1 { font-size: 2.25rem; }
    }
    </style>
    """,
    unsafe_allow_html=True,
)

settings = get_settings()
store = build_store()
document_count = store.count()

with st.sidebar:
    st.markdown("### About HindiRAG")
    st.caption(
        "A retrieval-augmented generation system for querying Hindi "
        "documents using multilingual embeddings and ChromaDB."
    )
    st.divider()
    st.markdown("**Embedding**")
    st.caption("BAAI/bge-m3")
    st.markdown("**Vector Database**")
    st.caption("ChromaDB")
    st.markdown("**LLM**")
    st.caption("OpenRouter")
    st.divider()
    st.markdown("**Document**")
    st.caption("Hindi Agriculture")
    st.caption(
        "Crops, soil, irrigation, fertilizers, modern agricultural "
        "technology, and sustainable farming."
    )
    st.divider()
    st.caption("HindiRAG v1.0")
    st.caption("Multilingual Document Intelligence")

st.markdown(
    """
    <div class="topbar">
      <div class="brand">
        <div class="brand-mark">HR</div>
        <div><div class="brand-name">HindiRAG</div>
        <div class="brand-subtitle">Hindi Document Intelligence</div></div>
      </div>
      <div class="topbar-right">
        <span class="online"><span class="online-dot"></span>
        System Online</span>
        <span class="tech-badge">BGE-M3</span>
        <span class="tech-badge">ChromaDB</span>
        <span class="tech-badge">OpenRouter</span>
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <section class="hero">
      <div class="eyebrow">Multilingual RAG</div>
      <h1>Ask questions about the Hindi document.</h1>
    <p>Search a Hindi document using multilingual semantic retrieval and
    generate grounded answers with RAG.</p>
    </section>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="section-label">Ask your question</div>',
    unsafe_allow_html=True,
)
with st.container(border=True):
    question = st.text_area(
        "Question",
        key="question",
        height=112,
        label_visibility="collapsed",
        placeholder=(
            "Ask something about the document...\n"
            "Example: गेहूं की खेती के लिए कौन सी मिट्टी उपयुक्त है?"
        ),
    )
    st.markdown('<div class="ask-button">', unsafe_allow_html=True)
    ask = st.button("Ask Question", type="primary", width="stretch")
    st.markdown("</div>", unsafe_allow_html=True)

st.markdown(
    '<div class="sample-label">Try an example</div>',
    unsafe_allow_html=True,
)
st.markdown('<div class="sample-row">', unsafe_allow_html=True)
sample_columns = st.columns(4)
for index, (label, sample) in enumerate(SAMPLE_QUESTIONS):
    with sample_columns[index]:
        st.button(
            label,
            key=f"sample_{index}",
            width="stretch",
            on_click=set_question,
            args=(sample,),
        )
st.markdown("</div>", unsafe_allow_html=True)

if ask:
    if not question.strip():
        st.info("Enter a question to search the document.")
    elif document_count == 0:
        st.info(
            "No indexed document is available. Run `python ingest.py` first."
        )
    else:
        with st.spinner("Searching the document and preparing your answer..."):
            try:
                response = build_pipeline().ask(question.strip())
            except LLMError:
                st.error(
                    "Something went wrong while processing your question."
                )
                with st.expander("Technical details"):
                    st.caption(
                        "The configured language model could not complete "
                        "this request."
                    )
                st.stop()
            except Exception:
                st.error(
                    "Something went wrong while processing your question."
                )
                with st.expander("Technical details"):
                    st.caption("An unexpected application error occurred.")
                st.stop()

        st.markdown(
            '<div class="answer-heading"><h2>Answer</h2></div>',
            unsafe_allow_html=True,
        )
        with st.container(border=True):
            st.markdown(clean_visible_answer(response.answer))
        if response.sources:
            st.markdown(
                '<div class="sources-heading"><h2>Retrieved Sources</h2>'
                '<p>Relevant passages retrieved from the Hindi document.</p>'
                '</div>',
                unsafe_allow_html=True,
            )
            for source in response.sources:
                metadata = source["metadata"]
                similarity = source.get("similarity", 0.0)
                chunk_id = metadata.get("chunk_id", "chunk")
                chunk_number = re.search(r"\d+", chunk_id)
                chunk_label = (
                    f"Chunk {int(chunk_number.group()) + 1:02d}"
                    if chunk_number
                    else "Source chunk"
                )
                section = metadata.get("section", "Hindi document")
                label = (
                    f"{chunk_label}  ·  {similarity:.3f} similarity  · "
                    f"{section}"
                )
                with st.expander(label):
                    st.caption("View source context")
                    st.write(source["text"])
        else:
            st.info("No relevant information found in the document.")

st.markdown(
    '<div class="footer">'
    'HindiRAG  ·  Multilingual Document Intelligence</div>',
    unsafe_allow_html=True,
)
