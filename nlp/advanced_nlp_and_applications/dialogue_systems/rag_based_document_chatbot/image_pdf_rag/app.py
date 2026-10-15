from __future__ import annotations

import hashlib
import re
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
from image_pdf_rag.ocr import get_tesseract_status
from utils.validators import validate_uploaded_file

if __package__:
    from .pipeline import ImagePdfRAGPipeline
else:
    from image_pdf_rag.pipeline import ImagePdfRAGPipeline

COLLECTION_NAME = "rag_scanned_image_pdf"
SESSION_PREFIX = "scanned_pdf_"


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


def _pipeline() -> ImagePdfRAGPipeline:
    pipeline = st.session_state.get(f"{SESSION_PREFIX}pipeline")
    if pipeline is not None:
        return pipeline

    embedding_model = st.session_state.get(
        f"{SESSION_PREFIX}embedding_model", settings.embedding_model
    )
    backend = st.session_state.get(f"{SESSION_PREFIX}vector_backend", settings.vector_store)
    vector_store, active_backend = _get_vector_store(backend)
    st.session_state[f"{SESSION_PREFIX}active_backend"] = active_backend
    pipeline = ImagePdfRAGPipeline(
        embedding_service=_load_embeddings(embedding_model),
        vector_store=vector_store,
        chunk_size=st.session_state.get(f"{SESSION_PREFIX}chunk_size", settings.chunk_size),
        chunk_overlap=st.session_state.get(
            f"{SESSION_PREFIX}chunk_overlap", settings.chunk_overlap
        ),
        top_k=st.session_state.get(f"{SESSION_PREFIX}top_k", settings.retrieval_top_k),
    )
    st.session_state[f"{SESSION_PREFIX}pipeline"] = pipeline
    return pipeline


def render_home() -> None:
    st.markdown(
        "Upload a scanned document, screenshot PDF, or image-based form. The pipeline "
        "renders each page, applies OCR, indexes the extracted text, and cites the "
        "page that supports each answer."
    )
    st.code(
        "PDF upload → page detection → OCR → text cleanup → chunking → embeddings → "
        "retrieval → grounded answer",
        language="text",
    )
    tesseract_ready, tesseract_message = get_tesseract_status()
    if tesseract_ready:
        st.success(tesseract_message)
    else:
        st.warning(tesseract_message)


def render_upload_and_index() -> None:
    st.subheader("1. Upload and index a PDF")
    selected_backend = st.session_state.get(
        f"{SESSION_PREFIX}vector_backend", settings.vector_store
    )
    active_backend = st.session_state.get(f"{SESSION_PREFIX}active_backend")
    if selected_backend == "chroma" and active_backend == "faiss":
        st.warning(
            "ChromaDB is not installed, so this session is using FAISS instead. "
            "Indexed vectors are available for this session. Install chromadb "
            "from requirements.txt to use persistent Chroma storage."
        )
    with st.sidebar:
        st.markdown("### Scanned PDF settings")
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
        backend_index = backend_options.index(default_backend)
        st.session_state[f"{SESSION_PREFIX}vector_backend"] = st.selectbox(
            "Vector backend",
            backend_options,
            index=backend_index,
            key=f"{SESSION_PREFIX}vector_backend_input",
        )
        st.session_state[f"{SESSION_PREFIX}chunk_size"] = st.slider(
            "Chunk size",
            min_value=200,
            max_value=2000,
            value=st.session_state.get(f"{SESSION_PREFIX}chunk_size", settings.chunk_size),
            key=f"{SESSION_PREFIX}chunk_size_input",
        )
        st.session_state[f"{SESSION_PREFIX}chunk_overlap"] = st.slider(
            "Chunk overlap",
            min_value=0,
            max_value=500,
            value=st.session_state.get(
                f"{SESSION_PREFIX}chunk_overlap", settings.chunk_overlap
            ),
            key=f"{SESSION_PREFIX}chunk_overlap_input",
        )
        st.session_state[f"{SESSION_PREFIX}top_k"] = st.slider(
            "Retrieved passages",
            min_value=1,
            max_value=10,
            value=st.session_state.get(f"{SESSION_PREFIX}top_k", settings.retrieval_top_k),
            key=f"{SESSION_PREFIX}top_k_input",
        )
        routing_label = st.radio(
            "Page extraction",
            ["Automatic (recommended)", "OCR every page", "Native text only"],
            help="Automatic mode sends image-heavy or text-sparse pages to OCR.",
            key=f"{SESSION_PREFIX}routing",
        )

    uploaded = st.file_uploader(
        "Choose a scanned or image-based PDF",
        type=["pdf"],
        key=f"{SESSION_PREFIX}file",
    )
    route = {
        "Automatic (recommended)": "auto",
        "OCR every page": "ocr",
        "Native text only": "native",
    }[routing_label]
    if st.button(
        "Extract text and index PDF",
        type="primary",
        disabled=uploaded is None,
        key=f"{SESSION_PREFIX}index",
    ):
        try:
            validate_uploaded_file(uploaded.name, len(uploaded.getvalue()))
            saved_path = _save_upload(uploaded)
            st.session_state.pop(f"{SESSION_PREFIX}pipeline", None)
            pipeline = _pipeline()
            with st.spinner("Rendering pages, extracting text, and creating embeddings..."):
                result = pipeline.index_pdf(saved_path, routing=route)
            st.session_state[f"{SESSION_PREFIX}indexed_file"] = saved_path.name
            st.session_state[f"{SESSION_PREFIX}indexed_chunks"] = result["chunks"]
            st.session_state[f"{SESSION_PREFIX}chat_history"] = []
            st.success(
                f"Indexed {result['chunk_count']} text chunks from "
                f"{result['indexed_page_count']} readable page(s). "
                f"OCR was used on {result['ocr_page_count']} of "
                f"{result['page_count']} page(s)."
            )
        except Exception as exc:
            st.error(f"Could not process this PDF: {exc}")

    chunks = st.session_state.get(f"{SESSION_PREFIX}indexed_chunks", [])
    if chunks:
        with st.expander("Review extracted text and page metadata"):
            for chunk in chunks:
                metadata = chunk["metadata"]
                st.markdown(
                    f"**Page {metadata['page_number']} · "
                    f"{metadata['extraction_method']} · "
                    f"{metadata['source_file']}**"
                )
                st.write(chunk["text"])
                st.divider()


def render_chat() -> None:
    st.subheader("2. Ask questions about the PDF")
    if not st.session_state.get(f"{SESSION_PREFIX}indexed_chunks"):
        st.info("Upload and index a PDF before asking questions.")
        return

    history = st.session_state.setdefault(f"{SESSION_PREFIX}chat_history", [])
    for message in history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            for source in message.get("sources", []):
                with st.expander(
                    f"Source: {source['filename']} — page {source['page_number']}"
                ):
                    st.caption(source.get("page_image", ""))
                    st.write(source["text"])

    question = st.chat_input(
        "Ask a question about the scanned document",
        key=f"{SESSION_PREFIX}question",
    )
    if not question:
        return

    history.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)
    try:
        api_key = settings.groq_api_key
        if not api_key:
            raise ValueError(
                "GROQ_API_KEY is not set. Add it to the chatbot .env file to generate answers."
            )
        llm = _load_llm(api_key, settings.groq_model)
        with st.spinner("Searching the extracted pages and generating an answer..."):
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
            st.markdown(result["answer"])
            if result["sources"]:
                st.markdown("**Page sources**")
                for source in result["sources"]:
                    with st.expander(
                        f"{source['filename']} — page {source['page_number']}"
                    ):
                        st.caption(source.get("page_image", ""))
                        st.write(source["text"])
            else:
                st.caption("No supporting page was found in the indexed PDF.")
    except Exception as exc:
        history.pop()
        st.error(f"Could not answer this question: {exc}")


def render_app(embed: bool = False) -> None:
    if not embed:
        st.set_page_config(page_title="Scanned PDF RAG Chatbot", page_icon="📄", layout="wide")
    st.title("Scanned / Image-based PDF Chatbot")
    render_home()
    render_upload_and_index()
    render_chat()


if __name__ == "__main__":
    render_app()
