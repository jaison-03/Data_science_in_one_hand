from __future__ import annotations

import hashlib
import os
import re
import sys
from io import BytesIO
from pathlib import Path

import fitz
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
from handwritten_pdf_rag.ocr import LocalTesseractOCR, TextractOCR
from handwritten_pdf_rag.pipeline import HandwrittenPdfRAGPipeline

COLLECTION_NAME = "rag_handwritten_pdf"
SESSION_PREFIX = "handwritten_pdf_"


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
            return FAISSVectorStore(
                dimension=384, collection_name=COLLECTION_NAME
            ), "faiss"
    if backend == "faiss":
        return FAISSVectorStore(dimension=384, collection_name=COLLECTION_NAME), "faiss"
    raise ValueError(f"Unsupported vector backend '{backend}'.")


def _save_upload(uploaded_file) -> tuple[Path, str]:
    content = uploaded_file.getvalue()
    document_id = hashlib.sha256(content).hexdigest()
    safe_name = re.sub(r"[^A-Za-z0-9_.-]+", "_", Path(uploaded_file.name).name)
    uploads_dir = PROJECT_ROOT / "uploads"
    uploads_dir.mkdir(parents=True, exist_ok=True)
    target = uploads_dir / f"{document_id[:12]}_{safe_name}"
    target.write_bytes(content)
    return target, document_id


def _pipeline() -> HandwrittenPdfRAGPipeline:
    pipeline = st.session_state.get(f"{SESSION_PREFIX}pipeline")
    if pipeline is not None:
        return pipeline
    embedding_model = st.session_state.get(
        f"{SESSION_PREFIX}embedding_model", settings.embedding_model
    )
    backend = st.session_state.get(
        f"{SESSION_PREFIX}vector_backend", settings.vector_store
    )
    vector_store, active_backend = _get_vector_store(backend)
    st.session_state[f"{SESSION_PREFIX}active_backend"] = active_backend
    pipeline = HandwrittenPdfRAGPipeline(
        embedding_service=_load_embeddings(embedding_model),
        vector_store=vector_store,
        chunk_size=st.session_state.get(
            f"{SESSION_PREFIX}chunk_size", settings.chunk_size
        ),
        chunk_overlap=st.session_state.get(
            f"{SESSION_PREFIX}chunk_overlap", settings.chunk_overlap
        ),
        top_k=st.session_state.get(f"{SESSION_PREFIX}top_k", settings.retrieval_top_k),
        dpi=st.session_state.get(f"{SESSION_PREFIX}dpi", 300),
    )
    st.session_state[f"{SESSION_PREFIX}pipeline"] = pipeline
    return pipeline


def _build_ocr_engine(mode: str):
    if mode == "AWS Textract (handwriting-aware)":
        region = os.environ.get("AWS_REGION") or os.environ.get("AWS_DEFAULT_REGION")
        return TextractOCR(region_name=region)
    return LocalTesseractOCR()


def _render_page(path: str, page_number: int) -> bytes:
    with fitz.open(path) as pdf:
        if page_number < 1 or page_number > pdf.page_count:
            raise ValueError(f"Page {page_number} is not available in the source PDF.")
        pixmap = pdf.load_page(page_number - 1).get_pixmap(dpi=140, alpha=False)
    return pixmap.tobytes("png")


def _render_home() -> None:
    st.markdown(
        "Upload handwritten notes, letters, forms, calculations, or scanned PDFs. "
        "Each page is inspected independently; recognized text is indexed with page "
        "and extraction metadata so answers can cite the original page."
    )
    st.code(
        "PDF → page inspection → handwriting-aware OCR → page-aware chunks → "
        "embeddings → grounded answers with page references",
        language="text",
    )
    st.info(
        "For handwriting, select AWS Textract. Local Tesseract is available as a "
        "fallback but is designed primarily for printed text and can misread handwriting."
    )


def _render_upload_and_index() -> None:
    st.subheader("1. Upload and read a PDF")
    selected_backend = st.session_state.get(
        f"{SESSION_PREFIX}vector_backend", settings.vector_store
    )
    if (
        selected_backend == "chroma"
        and st.session_state.get(f"{SESSION_PREFIX}active_backend") == "faiss"
    ):
        st.warning(
            "ChromaDB is unavailable, so this session is using FAISS. The index is "
            "available for this session only."
        )

    with st.sidebar:
        st.markdown("### Handwritten PDF settings")
        st.session_state[f"{SESSION_PREFIX}embedding_model"] = st.text_input(
            "Embedding model",
            value=st.session_state.get(
                f"{SESSION_PREFIX}embedding_model", settings.embedding_model
            ),
            key=f"{SESSION_PREFIX}embedding_model_input",
        )
        options = ["chroma", "faiss"]
        current = st.session_state.get(f"{SESSION_PREFIX}vector_backend", settings.vector_store)
        selected_index = options.index(current) if current in options else 0
        st.session_state[f"{SESSION_PREFIX}vector_backend"] = st.selectbox(
            "Vector backend",
            options,
            index=selected_index,
            key=f"{SESSION_PREFIX}vector_backend_input",
        )
        st.session_state[f"{SESSION_PREFIX}chunk_size"] = st.slider(
            "Chunk size", 200, 2000,
            st.session_state.get(f"{SESSION_PREFIX}chunk_size", settings.chunk_size),
            key=f"{SESSION_PREFIX}chunk_size_input",
        )
        overlap = st.session_state.get(
            f"{SESSION_PREFIX}chunk_overlap", settings.chunk_overlap
        )
        st.session_state[f"{SESSION_PREFIX}chunk_overlap"] = st.slider(
            "Chunk overlap", 0, 500, min(overlap, 500),
            key=f"{SESSION_PREFIX}chunk_overlap_input",
        )
        st.session_state[f"{SESSION_PREFIX}top_k"] = st.slider(
            "Retrieved passages", 1, 10,
            st.session_state.get(f"{SESSION_PREFIX}top_k", settings.retrieval_top_k),
            key=f"{SESSION_PREFIX}top_k_input",
        )
        st.session_state[f"{SESSION_PREFIX}dpi"] = st.slider(
            "OCR render resolution (DPI)",
            150,
            350,
            st.session_state.get(f"{SESSION_PREFIX}dpi", 300),
            step=50,
            help="Reduce resolution if a rendered page exceeds the OCR provider's image-size limit.",
            key=f"{SESSION_PREFIX}dpi_input",
        )

    ocr_mode = st.radio(
        "OCR engine",
        ["AWS Textract (handwriting-aware)", "Local Tesseract (printed-text fallback)"],
        help=(
            "Textract uses the AWS credential provider chain and sends page images "
            "to AWS. Tesseract runs locally but is not handwriting-specialized."
        ),
        key=f"{SESSION_PREFIX}ocr_mode",
    )
    if ocr_mode.startswith("AWS"):
        st.caption(
            "Configure AWS credentials using the standard AWS SDK provider chain "
            "(for example AWS_PROFILE or AWS_ACCESS_KEY_ID/AWS_SECRET_ACCESS_KEY) "
            "and AWS_REGION. Required permission: textract:DetectDocumentText."
        )
    else:
        ready, status = get_tesseract_status()
        if ready:
            st.success(status)
        else:
            st.warning(status)

    uploaded = st.file_uploader(
        "Choose a handwritten or mixed-content PDF",
        type=["pdf"],
        key=f"{SESSION_PREFIX}file",
    )
    if st.button(
        "Read and index PDF",
        type="primary",
        disabled=uploaded is None,
        key=f"{SESSION_PREFIX}index",
    ):
        try:
            validate_uploaded_file(uploaded.name, len(uploaded.getvalue()))
            saved_path, document_id = _save_upload(uploaded)
            st.session_state.pop(f"{SESSION_PREFIX}pipeline", None)
            pipeline = _pipeline()
            engine = _build_ocr_engine(
                "AWS Textract (handwriting-aware)"
                if ocr_mode.startswith("AWS")
                else "Local Tesseract"
            )
            progress = st.progress(0, text="Inspecting PDF pages...")

            def update_progress(page_number: int, page_count: int) -> None:
                progress.progress(
                    page_number / page_count,
                    text=f"Reading page {page_number} of {page_count}...",
                )

            with st.spinner("Extracting page text and building the document index..."):
                result = pipeline.index_pdf(
                    saved_path,
                    document_id,
                    engine,
                    on_page=update_progress,
                )
            progress.empty()
            st.session_state[f"{SESSION_PREFIX}indexed_file"] = str(saved_path)
            st.session_state[f"{SESSION_PREFIX}document_id"] = document_id
            st.session_state[f"{SESSION_PREFIX}result"] = result
            st.session_state[f"{SESSION_PREFIX}chat_history"] = []
            st.success(
                f"Read {result['indexed_page_count']} of {result['page_count']} pages "
                f"and indexed {result['chunk_count']} passages. OCR was used on "
                f"{result['ocr_page_count']} page(s)."
            )
        except Exception as exc:
            st.error(f"Could not process this PDF: {exc}")

    result = st.session_state.get(f"{SESSION_PREFIX}result")
    if not result:
        return
    needs_review = [
        page for page in result["pages"] if page["needs_review"]
    ]
    if needs_review:
        st.warning(
            "Some pages may contain unreadable or low-confidence handwriting. "
            "Review these pages before relying on names, dates, amounts, or symbols: "
            + ", ".join(str(page["page_number"]) for page in needs_review)
        )
    with st.expander("Review extracted text and page quality"):
        for page in result["pages"]:
            st.markdown(
                f"**Page {page['page_number']}** — {page['content_type']} · "
                f"{page['extraction_method']}"
                + (
                    f" · OCR confidence {page['ocr_confidence']:.1f}%"
                    if page["ocr_confidence"] is not None
                    else ""
                )
            )
            st.text(page["text"] or "No readable text detected.")
            if page["needs_review"]:
                st.caption("Please verify this page against the original image.")


def _render_chat() -> None:
    document_id = st.session_state.get(f"{SESSION_PREFIX}document_id")
    if not document_id:
        st.info("Read and index a PDF to start asking questions.")
        return
    if not settings.groq_api_key:
        st.warning("Add GROQ_API_KEY to your environment or .env file before chatting.")
        return
    st.subheader("2. Ask about the PDF")
    history = st.session_state.setdefault(f"{SESSION_PREFIX}chat_history", [])
    for item in history:
        with st.chat_message(item["role"]):
            st.markdown(item["content"])
            if item.get("sources"):
                with st.expander("Source pages and OCR evidence"):
                    for source in item["sources"]:
                        st.markdown(
                            f"**{source['filename']} — page {source['page_number']}** · "
                            f"{source['content_type']} · {source['extraction_method']}"
                            + (
                                f" · OCR confidence {source['ocr_confidence']:.1f}%"
                                if source["ocr_confidence"] is not None
                                else ""
                            )
                        )
                        st.text(source["text"])
                        image = _render_page(
                            source["source_path"], source["page_number"]
                        )
                        st.image(image, caption=f"Original PDF page {source['page_number']}")

    question = st.chat_input(
        "Ask a question about the handwriting",
        key=f"{SESSION_PREFIX}question",
    )
    if not question:
        return
    history.append({"role": "user", "content": question})
    pipeline = _pipeline()
    llm = _load_llm(settings.groq_api_key, settings.groq_model)
    pipeline.llm_service = llm
    prior_history = [
        item["content"] for item in history[:-1] if item["role"] == "assistant"
    ]
    try:
        with st.spinner("Searching the extracted pages..."):
            answer = pipeline.answer_question(
                question,
                document_id=document_id,
                chat_history=prior_history,
            )
        history.append(
            {
                "role": "assistant",
                "content": answer["answer"],
                "sources": answer["sources"],
            }
        )
        st.rerun()
    except Exception as exc:
        history.pop()
        st.error(f"Could not answer this question: {exc}")


def render_app(embed: bool = False) -> None:
    if not embed:
        st.title("Handwritten PDF Intelligence Chatbot")
    else:
        st.subheader("Handwritten PDF Intelligence Chatbot")
    _render_home()
    _render_upload_and_index()
    _render_chat()
