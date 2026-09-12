"""Streamlit user interface for Hindi document QA."""

import logging
import re

import streamlit as st

from src.config import get_settings
from src.errors import LLMError
from src.embeddings import EmbeddingModel
from src.llm import build_llm
from src.rag_pipeline import RAGPipeline
from src.retriever import Retriever
from src.vector_store import ChromaVectorStore

logger = logging.getLogger(__name__)
logger.info("Starting HindiRAG application...")

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
def load_embeddings(model_name: str) -> EmbeddingModel:
    """Load one embedding adapter/model per Streamlit process."""
    return EmbeddingModel(model_name)


@st.cache_resource
def load_store(
    persist_directory: str, collection_name: str, model_name: str
) -> ChromaVectorStore:
    """Open one Chroma client per Streamlit process without ingesting."""
    store = ChromaVectorStore(
        persist_directory,
        collection_name,
        load_embeddings(model_name),
    )
    return store


@st.cache_resource
def build_pipeline():
    settings = get_settings()
    store = load_store(
        str(settings.chroma_persist_directory),
        settings.chroma_collection_name,
        settings.embedding_model,
    )
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
        --background: #F7F9F7;
        --card: #FFFFFF;
        --primary: #176B45;
        --primary-hover: #125638;
        --text: #17211B;
        --secondary-text: #68736D;
        --border: #DDE5DF;
        --soft-green: #EAF5EF;
    }
    .stApp { background: var(--background); color: var(--text); }
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
        border-right: 1px solid var(--border);
        background: var(--background);
    }
    [data-testid="stSidebar"] .block-container { padding-top: 2rem; }
    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] {
        color: var(--text);
    }
    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p,
    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] strong,
    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] h1,
    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] h2,
    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] h3,
    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] h4,
    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] h5,
    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] h6 {
        color: var(--text);
    }
    [data-testid="stSidebar"] [data-testid="stCaptionContainer"],
    [data-testid="stSidebar"] [data-testid="stCaptionContainer"] p {
        color: var(--secondary-text);
    }
    [data-testid="stSidebar"] [data-testid="stTextInput"] input,
    [data-testid="stSidebar"] [data-testid="stTextArea"] textarea,
    [data-testid="stSidebar"] label {
        color: var(--text);
    }
    .topbar {
        display: flex;
        align-items: center;
        justify-content: space-between;
        gap: 1rem;
        padding-bottom: 1.1rem;
        border-bottom: 1px solid var(--border);
    }
    .brand { display: flex; align-items: center; gap: .8rem; }
    .brand-mark {
        display: grid;
        place-items: center;
        width: 2.35rem;
        height: 2.35rem;
        border-radius: 9px;
        background: var(--primary);
        color: var(--card);
        font-weight: 800;
        letter-spacing: -.04em;
    }
    .brand-name { font-size: 1.05rem; font-weight: 800; line-height: 1.1; }
    .brand-subtitle {
        color: var(--secondary-text); font-size: .78rem; margin-top: .18rem;
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
        color: var(--primary-hover);
        font-size: .78rem;
        font-weight: 700;
        margin-right: .35rem;
    }
    .online-dot {
        width: .45rem; height: .45rem; border-radius: 50%;
        background: var(--primary);
    }
    .tech-badge {
        border: 1px solid var(--border);
        border-radius: 999px;
        background: var(--card);
        color: var(--secondary-text);
        padding: .34rem .68rem;
        font-size: .72rem;
        font-weight: 700;
    }
    .hero { padding: 2.25rem 0 1.6rem; max-width: 760px; }
    .eyebrow {
        color: var(--primary);
        font-size: .7rem;
        font-weight: 800;
        letter-spacing: .14em;
        text-transform: uppercase;
        margin-bottom: .85rem;
    }
    .hero h1 {
        color: var(--text);
        font-size: clamp(2rem, 3.5vw, 3rem);
        line-height: 1.08;
        letter-spacing: -.045em;
        margin: 0 0 .8rem;
    }
    .hero p {
        color: var(--secondary-text);
        font-size: 1.05rem;
        line-height: 1.6;
        margin: 0;
        max-width: 650px;
    }
    .section-label {
        color: var(--text); font-size: 1rem; font-weight: 750;
        margin: .2rem 0 .7rem;
    }
    .input-shell {
        background: var(--card);
        border: 1px solid var(--border);
        border-radius: 14px;
        padding: 1rem;
        box-shadow: 0 8px 24px rgba(31, 55, 39, .05);
    }
    [data-testid="stTextArea"] textarea {
        background-color: #ffffff;
        color: #17211b;
        border: 1px solid #d8e2dc;
        border-radius: 10px;
        min-height: 112px;
        font-size: 1rem;
        caret-color: #17211b;
    }
    [data-testid="stTextArea"] textarea::placeholder {
        color: #7a8580;
        opacity: 1;
    }
    [data-testid="stTextArea"] textarea:focus {
        border-color: #247a52;
        box-shadow: 0 0 0 1px #247a52;
    }
    .sample-label {
        color: var(--secondary-text); font-size: .78rem; font-weight: 700;
        margin: 1.35rem 0 .55rem;
    }
    .sample-row [data-testid="stButton"] button {
        min-height: 2.65rem;
        border: 1px solid var(--border);
        border-radius: 9px;
        background: var(--card);
        color: var(--text);
        font-size: .78rem;
        line-height: 1.25;
        padding: .55rem .7rem;
        transition: border-color .15s ease, background .15s ease;
    }
    .sample-row [data-testid="stButton"] button:hover {
        border-color: var(--primary);
        background: var(--soft-green);
        color: var(--primary-hover);
    }
    .ask-button button,
    [data-testid="stBaseButton-primary"] {
        border-radius: 9px;
        background: var(--primary);
        border: 1px solid var(--primary);
        color: #ffffff;
        font-weight: 750;
        min-height: 2.8rem;
    }
    .ask-button button:hover,
    [data-testid="stBaseButton-primary"]:hover {
        background: var(--primary-hover); border-color: var(--primary-hover);
        color: #ffffff;
    }
    [data-testid="stBaseButton-primary"] p,
    [data-testid="stBaseButton-primary"] span {
        color: #ffffff;
    }
    .answer-heading { margin-top: 2.4rem; }
    .answer-heading h2, .sources-heading h2 {
        font-size: .75rem;
        letter-spacing: .12em;
        text-transform: uppercase;
    }
    .answer-heading h2 { margin: 0 0 .7rem; }
    .answer-box {
        background: var(--card);
        border: 1px solid var(--border);
        border-left: 4px solid var(--primary);
        border-radius: 4px 12px 12px 4px;
        padding: 1.1rem 1.3rem;
        line-height: 1.7;
        box-shadow: 0 8px 24px rgba(31, 55, 39, .04);
    }
    .sources-heading { margin-top: 2rem; }
    .sources-heading h2 { font-size: 1.08rem; margin: 0 0 .25rem; }
    .sources-heading p {
        color: var(--secondary-text); font-size: .82rem; margin: 0 0 .75rem;
    }
    [data-testid="stExpander"] {
        border: 1px solid var(--border); border-radius: 9px;
        background: var(--card);
    }
    [data-testid="stExpander"] summary,
    [data-testid="stExpander"] summary p,
    [data-testid="stExpander"] [data-testid="stMarkdownContainer"],
    [data-testid="stExpander"] [data-testid="stCaptionContainer"] {
        color: var(--text);
    }
    [data-testid="stExpander"] [data-testid="stCaptionContainer"] p,
    [data-testid="stExpander"] [data-testid="stMarkdownContainer"] p {
        color: var(--secondary-text);
    }
    [data-testid="stExpander"] summary p {
        font-size: .82rem; font-weight: 700;
    }
    .footer {
        color: var(--secondary-text); font-size: .72rem; padding-top: 1.5rem;
    }
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
logger.info("Connecting to ChromaDB...")
logger.info("Checking ChromaDB index...")
store = load_store(
    str(settings.chroma_persist_directory),
    settings.chroma_collection_name,
    settings.embedding_model,
)
document_count = store.count()
logger.info("ChromaDB path: %s", store.persist_directory)
logger.info("Chroma collection: %s", settings.chroma_collection_name)
logger.info("Chroma document count: %d", document_count)
index_ready = document_count > 0 and store.has_compatible_embedding_model()
if index_ready:
    logger.info("Using existing index with %d chunks", document_count)
elif document_count:
    logger.warning(
        "ChromaDB index uses a different embedding model; run python ingest.py"
    )
else:
    logger.warning(
        "ChromaDB index is empty at %s; run python ingest.py",
        store.persist_directory,
    )
logger.info("Streamlit application ready.")

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
    elif not index_ready:
        st.info(
            "No indexed document is available. Run `python ingest.py` first."
        )
    else:
        with st.spinner("Searching the document and preparing your answer..."):
            try:
                question_text = question.strip()
                logger.info(
                    "RAG query started: question_length=%d",
                    len(question_text),
                )
                response = build_pipeline().ask(question_text)
                logger.info("Retrieved chunks: %d", len(response.sources))
                logger.info("RAG query completed")
            except LLMError:
                logger.exception("LLM generation failed during RAG query")
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
                logger.exception("RAG query failed unexpectedly")
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
