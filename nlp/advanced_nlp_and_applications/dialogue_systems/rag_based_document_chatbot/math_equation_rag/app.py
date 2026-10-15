from __future__ import annotations

import hashlib
import re
import shutil
import sys
from pathlib import Path

import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config.settings import settings
from core.embeddings import EmbeddingService
from core.llm_service import LLMService
from core.vector_store import ChromaVectorStore, FAISSVectorStore
from math_equation_rag.pipeline import MathEquationRAGPipeline
from math_equation_rag.rendering import format_math_markdown
from utils.validators import validate_uploaded_file

COLLECTION_NAME = "rag_math_equations_pdf"
SESSION_PREFIX = "math_pdf_"


@st.cache_resource
def _load_embeddings(model_name: str) -> EmbeddingService:
    return EmbeddingService(model_name=model_name)


@st.cache_resource
def _load_llm(api_key: str, model_name: str) -> LLMService:
    return LLMService(api_key=api_key, model_name=model_name)


def _get_vector_store(backend: str):
    if backend == "chroma":
        try:
            return ChromaVectorStore(collection_name=COLLECTION_NAME), "chroma"
        except ModuleNotFoundError as exc:
            if "ChromaDB is not installed" not in str(exc):
                raise
            return FAISSVectorStore(dimension=384, collection_name=COLLECTION_NAME), "faiss"
    if backend == "faiss":
        return FAISSVectorStore(dimension=384, collection_name=COLLECTION_NAME), "faiss"
    raise ValueError(f"Unsupported vector backend '{backend}'.")


def _save_upload(uploaded_file) -> Path:
    uploads_dir = PROJECT_ROOT / "uploads"
    uploads_dir.mkdir(parents=True, exist_ok=True)
    safe_name = re.sub(r"[^A-Za-z0-9_.-]+", "_", Path(uploaded_file.name).name)
    digest = hashlib.sha256(uploaded_file.getvalue()).hexdigest()[:12]
    target = uploads_dir / f"{Path(safe_name).stem}_{digest}.pdf"
    target.write_bytes(uploaded_file.getvalue())
    return target


def _pipeline() -> MathEquationRAGPipeline:
    pipeline = st.session_state.get(f"{SESSION_PREFIX}pipeline")
    if pipeline is not None:
        return pipeline
    embedding_model = st.session_state.get(
        f"{SESSION_PREFIX}embedding_model", settings.embedding_model
    )
    backend = st.session_state.get(f"{SESSION_PREFIX}vector_backend", settings.vector_store)
    vector_store, active_backend = _get_vector_store(backend)
    st.session_state[f"{SESSION_PREFIX}active_backend"] = active_backend
    pipeline = MathEquationRAGPipeline(
        embedding_service=_load_embeddings(embedding_model),
        vector_store=vector_store,
        chunk_size=st.session_state.get(f"{SESSION_PREFIX}chunk_size", 1200),
        chunk_overlap=st.session_state.get(f"{SESSION_PREFIX}chunk_overlap", 150),
        top_k=st.session_state.get(f"{SESSION_PREFIX}top_k", settings.retrieval_top_k),
    )
    st.session_state[f"{SESSION_PREFIX}pipeline"] = pipeline
    return pipeline


def render_home() -> None:
    st.markdown(
        "Extract and chat with mathematics-heavy PDFs. Native extraction handles selectable "
        "PDF text; the optional Marker and Nougat engines convert equation-rich papers into "
        "Markdown with LaTeX before indexing."
    )
    st.code(
        "PDF → text / LaTeX extraction → equation-aware passages → embeddings → "
        "retrieval → cited answer",
        language="text",
    )
    if not shutil.which("marker_single"):
        st.info("For equation recognition from PDF page images, install Marker with `pip install marker-pdf`.")
    if not shutil.which("nougat"):
        st.caption("Nougat is an optional academic-paper parser: `pip install nougat-ocr`.")


def render_upload_and_index() -> None:
    st.subheader("1. Extract and index a mathematics PDF")
    selected_backend = st.session_state.get(
        f"{SESSION_PREFIX}vector_backend", settings.vector_store
    )
    active_backend = st.session_state.get(f"{SESSION_PREFIX}active_backend")
    if selected_backend == "chroma" and active_backend == "faiss":
        st.warning(
            "ChromaDB is unavailable in the active Python environment, so this session "
            "is using FAISS. FAISS vectors are available only for this session. Install "
            "requirements.txt in the environment running Streamlit; on Windows, "
            "chroma-hnswlib may also require Microsoft C++ Build Tools."
        )
    with st.sidebar:
        st.markdown("### Math PDF settings")
        st.session_state[f"{SESSION_PREFIX}embedding_model"] = st.text_input(
            "Embedding model",
            value=st.session_state.get(
                f"{SESSION_PREFIX}embedding_model", settings.embedding_model
            ),
            key=f"{SESSION_PREFIX}embedding_model_input",
        )
        backend_options = ["chroma", "faiss"]
        default_backend = st.session_state.get(
            f"{SESSION_PREFIX}vector_backend", settings.vector_store
        )
        st.session_state[f"{SESSION_PREFIX}vector_backend"] = st.selectbox(
            "Vector backend",
            backend_options,
            index=backend_options.index(default_backend),
            key=f"{SESSION_PREFIX}vector_backend_input",
        )
        st.session_state[f"{SESSION_PREFIX}chunk_size"] = st.slider(
            "Chunk size",
            min_value=300,
            max_value=2400,
            value=st.session_state.get(f"{SESSION_PREFIX}chunk_size", 1200),
            key=f"{SESSION_PREFIX}chunk_size_input",
        )
        overlap = min(
            st.session_state.get(f"{SESSION_PREFIX}chunk_overlap", 150),
            st.session_state[f"{SESSION_PREFIX}chunk_size"] - 1,
        )
        st.session_state[f"{SESSION_PREFIX}chunk_overlap"] = overlap
        st.session_state[f"{SESSION_PREFIX}chunk_overlap"] = st.slider(
            "Chunk overlap",
            min_value=0,
            max_value=min(500, st.session_state[f"{SESSION_PREFIX}chunk_size"] - 1),
            value=overlap,
            key=f"{SESSION_PREFIX}chunk_overlap_input",
        )
        st.session_state[f"{SESSION_PREFIX}top_k"] = st.slider(
            "Retrieved passages",
            min_value=1,
            max_value=10,
            value=st.session_state.get(f"{SESSION_PREFIX}top_k", settings.retrieval_top_k),
            key=f"{SESSION_PREFIX}top_k_input",
        )

    engine_label = st.radio(
        "Equation extraction engine",
        ["Native PDF text", "Marker (recommended for equations)", "Nougat (academic papers)"],
        help=(
            "Marker and Nougat are optional local tools that recognize equations in page "
            "images. Native extraction is fast but requires selectable PDF text."
        ),
        key=f"{SESSION_PREFIX}engine",
    )
    engine = {
        "Native PDF text": "native",
        "Marker (recommended for equations)": "marker",
        "Nougat (academic papers)": "nougat",
    }[engine_label]
    uploaded = st.file_uploader(
        "Choose a mathematics or equation-heavy PDF",
        type=["pdf"],
        key=f"{SESSION_PREFIX}file",
    )
    if st.button(
        "Extract equations and index PDF",
        type="primary",
        disabled=uploaded is None,
        key=f"{SESSION_PREFIX}index",
    ):
        try:
            validate_uploaded_file(uploaded.name, len(uploaded.getvalue()))
            saved_path = _save_upload(uploaded)
            st.session_state.pop(f"{SESSION_PREFIX}pipeline", None)
            pipeline = _pipeline()
            with st.spinner("Extracting mathematical text and creating embeddings..."):
                result = pipeline.index_pdf(saved_path, engine=engine)
            st.session_state[f"{SESSION_PREFIX}indexed_file"] = saved_path.name
            st.session_state[f"{SESSION_PREFIX}indexed_chunks"] = result["chunks"]
            st.session_state[f"{SESSION_PREFIX}chat_history"] = []
            st.success(
                f"Indexed {result['chunk_count']} passages from "
                f"{result['indexed_page_count']} extracted page(s) using {engine_label}."
            )
        except Exception as exc:
            st.error(f"Could not process this mathematics PDF: {exc}")

    chunks = st.session_state.get(f"{SESSION_PREFIX}indexed_chunks", [])
    if chunks:
        with st.expander("Review extracted equations and passages"):
            for chunk in chunks:
                page = chunk["metadata"].get("page_number")
                page_label = f"Page {page}" if page is not None else "Page unavailable"
                st.markdown(f"**{page_label} · {chunk['metadata']['extraction_method']}**")
                st.markdown(format_math_markdown(chunk["text"]))
                st.divider()


def render_chat() -> None:
    st.subheader("2. Ask questions about the equations")
    if not st.session_state.get(f"{SESSION_PREFIX}indexed_chunks"):
        st.info("Upload and index a mathematics PDF before asking questions.")
        return

    history = st.session_state.setdefault(f"{SESSION_PREFIX}chat_history", [])
    for message in history:
        with st.chat_message(message["role"]):
            st.markdown(format_math_markdown(message["content"]))
            for source in message.get("sources", []):
                page = source.get("page_number")
                page_label = f"page {page}" if page is not None else "page unavailable"
                with st.expander(f"Source: {source['filename']} — {page_label}"):
                    st.markdown(format_math_markdown(source["text"]))

    question = st.chat_input(
        "Ask about a formula, derivation, symbol, or result",
        key=f"{SESSION_PREFIX}question",
    )
    if not question:
        return
    history.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)
    try:
        if not settings.groq_api_key:
            raise ValueError(
                "GROQ_API_KEY is not set. Add it to the chatbot .env file to generate answers."
            )
        llm = _load_llm(settings.groq_api_key, settings.groq_model)
        with st.spinner("Searching equations and generating a grounded answer..."):
            pipeline = _pipeline()
            pipeline.llm_service = llm
            result = pipeline.answer_question(
                question,
                chat_history=[entry["content"] for entry in history[-5:-1]],
            )
        history.append(
            {
                "role": "assistant",
                "content": result["answer"],
                "sources": result["sources"],
            }
        )
        with st.chat_message("assistant"):
            st.markdown(format_math_markdown(result["answer"]))
            if result["sources"]:
                st.markdown("**PDF sources**")
                for source in result["sources"]:
                    page = source.get("page_number")
                    page_label = f"page {page}" if page is not None else "page unavailable"
                    with st.expander(f"{source['filename']} — {page_label}"):
                        st.markdown(format_math_markdown(source["text"]))
    except Exception as exc:
        history.pop()
        st.error(f"Could not answer this question: {exc}")


def render_app(embed: bool = False) -> None:
    if not embed:
        st.set_page_config(page_title="Math & Equations PDF Chatbot", page_icon="∑", layout="wide")
    st.title("Math & Equations PDF Chatbot")
    render_home()
    render_upload_and_index()
    render_chat()


if __name__ == "__main__":
    render_app()
