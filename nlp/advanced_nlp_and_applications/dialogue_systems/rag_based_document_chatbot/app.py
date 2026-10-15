from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from uuid import uuid4

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config.settings import settings
from core.chunking import chunk_document
from core.document_loader import DocumentLoader
from core.embeddings import EmbeddingService
from core.llm_service import LLMService
from core.rag_pipeline import RAGPipeline
from core.retriever import Retriever
from core.vector_store import create_vector_store
from utils.validators import validate_uploaded_file


def _init_session_state() -> None:
    st.session_state.setdefault("uploaded_documents", [])
    st.session_state.setdefault("indexed_chunks", [])
    st.session_state.setdefault("chat_history", [])
    st.session_state.setdefault("current_vector_backend", settings.vector_store)
    st.session_state.setdefault("last_indexed_doc_count", 0)
    st.session_state.setdefault("embedding_model_loaded", False)


@st.cache_resource
def load_embedding_service(model_name: str) -> EmbeddingService:
    return EmbeddingService(model_name=model_name)


@st.cache_resource
def load_llm_service(api_key: str, model_name: str) -> LLMService:
    return LLMService(api_key=api_key, model_name=model_name)


def _save_uploaded_file(uploaded_file) -> str:
    uploads_dir = ROOT / "uploads"
    uploads_dir.mkdir(exist_ok=True)
    sanitized = re.sub(r"[^A-Za-z0-9_.-]+", "_", uploaded_file.name)
    target = uploads_dir / sanitized
    counter = 1
    while target.exists():
        stem = Path(sanitized).stem
        suffix = Path(sanitized).suffix
        target = uploads_dir / f"{stem}_{counter}{suffix}"
        counter += 1
    target.write_bytes(uploaded_file.getvalue())
    return str(target)


def _remove_document_from_index(filename: str) -> None:
    vector_store = st.session_state.get("vector_store")
    if vector_store is not None:
        vector_store.delete_document(filename)
    st.session_state["indexed_chunks"] = [
        chunk for chunk in st.session_state.get("indexed_chunks", []) if chunk.get("metadata", {}).get("source_file") != filename
    ]


def _process_uploaded_file(uploaded_file) -> dict:
    try:
        validate_uploaded_file(uploaded_file.name, len(uploaded_file.getvalue()))
        saved_path = _save_uploaded_file(uploaded_file)
        document = DocumentLoader().load(saved_path)
        status = "processed"
        message = "Document extracted successfully."
        if document.is_scanned_pdf:
            status = "warning"
            message = "This PDF appears to be scanned or image-based. OCR is required for text extraction."
        return {
            "id": uuid4().hex,
            "filename": document.filename,
            "status": status,
            "message": message,
            "source_path": document.source_path,
            "document": document,
            "error": None,
            "metadata": document.metadata,
        }
    except Exception as exc:  # pragma: no cover - user-facing validation path
        return {
            "id": uuid4().hex,
            "filename": uploaded_file.name,
            "status": "error",
            "message": str(exc),
            "source_path": None,
            "document": None,
            "error": str(exc),
            "metadata": {},
        }


def _index_documents() -> None:
    docs = [entry["document"] for entry in st.session_state.uploaded_documents if entry.get("document") is not None]
    if not docs:
        st.warning("Upload and process documents before indexing.")
        return

    backend = st.session_state.get("current_vector_backend", settings.vector_store)
    embedding_service = load_embedding_service(st.session_state.get("embedding_model", settings.embedding_model))
    vector_store = create_vector_store(backend)
    st.session_state["vector_store"] = vector_store

    all_chunks = []
    for doc in docs:
        chunks = chunk_document(doc, st.session_state.get("chunk_size", settings.chunk_size), st.session_state.get("chunk_overlap", settings.chunk_overlap))
        for chunk in chunks:
            chunk["metadata"]["source_file"] = doc.filename
            chunk["metadata"]["source_path"] = doc.source_path
            chunk["metadata"]["document_type"] = doc.document_type
            all_chunks.append(chunk)

    if not all_chunks:
        st.warning("No valid chunks were generated from the uploaded documents.")
        return

    embeddings = embedding_service.embed_documents([chunk["text"] for chunk in all_chunks])
    vector_store.add_documents(all_chunks, embeddings)
    st.session_state["indexed_chunks"] = all_chunks
    st.session_state["last_indexed_doc_count"] = len(docs)
    st.session_state["embedding_model_loaded"] = True
    st.success(f"Indexed {len(all_chunks)} chunks from {len(docs)} document(s).")


def _build_pipeline() -> RAGPipeline | None:
    if not st.session_state.get("vector_store"):
        st.warning("Please index the documents before asking questions.")
        return None

    api_key = settings.groq_api_key
    if not api_key:
        st.warning("Add GROQ_API_KEY to your environment or .env file before using the chat.")
        return None

    embedding_service = load_embedding_service(st.session_state.get("embedding_model", settings.embedding_model))
    llm_service = load_llm_service(api_key, st.session_state.get("groq_model", settings.groq_model))
    retriever = Retriever(
        vector_store=st.session_state["vector_store"],
        embedding_service=embedding_service,
        top_k=st.session_state.get("retrieval_top_k", settings.retrieval_top_k),
        threshold=st.session_state.get("similarity_threshold", settings.similarity_threshold),
    )
    return RAGPipeline(
        vector_store=st.session_state["vector_store"],
        embedding_service=embedding_service,
        llm_service=llm_service,
        retriever=retriever,
    )


def render_home() -> None:
    st.title("Intelligent Document Intelligence — RAG Chatbot")
    st.subheader("Home dashboard")
    st.write(
        "Retrieval-Augmented Generation (RAG) combines document retrieval with large language model reasoning. "
        "The system searches the most relevant passages, then asks the LLM to answer using only that grounded context."
    )
    st.code(
        "Document upload → preprocessing → chunking → embeddings → vector search → LLM answer",
        language="text",
    )

    groq_ready = bool(settings.groq_api_key)
    embedding_ready = st.session_state.get("embedding_model_loaded", False)
    backend = st.session_state.get("current_vector_backend", settings.vector_store)
    uploaded_count = len(st.session_state.get("uploaded_documents", []))
    indexed_count = len(st.session_state.get("indexed_chunks", []))

    metrics = st.columns(5)
    metrics[0].metric("Groq API", "Ready" if groq_ready else "Missing")
    metrics[1].metric("Embedding model", "Loaded" if embedding_ready else "Idle")
    metrics[2].metric("Vector backend", backend.upper())
    metrics[3].metric("Uploaded docs", uploaded_count)
    metrics[4].metric("Indexed chunks", indexed_count)

    st.markdown("### Quick instructions")
    st.markdown(
        "- Upload one or more PDF, TXT, or Markdown files.\n"
        "- Adjust chunk size, overlap, and retrieval settings.\n"
        "- Index the documents to create searchable embeddings.\n"
        "- Ask questions in the chat panel and inspect the source snippets."
    )


def render_documents() -> None:
    st.subheader("Document management")
    uploaded_files = st.file_uploader(
        "Upload one or more PDFs, TXT files, or Markdown documents",
        type=["pdf", "txt", "md", "markdown"],
        accept_multiple_files=True,
    )

    if uploaded_files:
        existing = st.session_state.uploaded_documents
        for uploaded in uploaded_files:
            if any(item.get("filename") == uploaded.name for item in existing):
                st.info(f"'{uploaded.name}' is already in the active document list.")
                continue
            processed = _process_uploaded_file(uploaded)
            existing.append(processed)
        st.session_state.uploaded_documents = existing

    if not st.session_state.uploaded_documents:
        st.info("No uploaded documents yet. Add a few files to begin indexing.")
        return

    for entry in st.session_state.uploaded_documents:
        filename = entry["filename"]
        status = entry["status"]
        message = entry["message"]

        with st.container():
            col1, col2, col3 = st.columns([3, 2, 1])
            col1.caption(filename)
            if status == "processed":
                col2.success("Processed")
            elif status == "warning":
                col2.warning("Needs OCR")
            else:
                col2.error("Error")
            if col3.button("Remove", key=f"remove_{entry['id']}"):
                st.session_state.uploaded_documents = [item for item in st.session_state.uploaded_documents if item["id"] != entry["id"]]
                _remove_document_from_index(filename)
                st.rerun()

        if entry.get("error"):
            st.warning(message)
        else:
            st.write(message)
            if entry.get("metadata"):
                st.caption(json.dumps(entry["metadata"], indent=2, ensure_ascii=False)[:300])


def render_controls() -> None:
    st.subheader("Document processing controls")
    if "chunk_size" not in st.session_state:
        st.session_state.chunk_size = settings.chunk_size
    if "chunk_overlap" not in st.session_state:
        st.session_state.chunk_overlap = settings.chunk_overlap
    if "retrieval_top_k" not in st.session_state:
        st.session_state.retrieval_top_k = settings.retrieval_top_k
    if "similarity_threshold" not in st.session_state:
        st.session_state.similarity_threshold = settings.similarity_threshold
    if "embedding_model" not in st.session_state:
        st.session_state.embedding_model = settings.embedding_model
    if "groq_model" not in st.session_state:
        st.session_state.groq_model = settings.groq_model

    with st.sidebar:
        st.markdown("### Processing settings")
        st.session_state.chunk_size = st.slider("Chunk size", min_value=200, max_value=2000, value=st.session_state.chunk_size)
        st.session_state.chunk_overlap = st.slider("Chunk overlap", min_value=0, max_value=500, value=st.session_state.chunk_overlap)
        st.session_state.embedding_model = st.text_input("Embedding model", value=st.session_state.embedding_model)
        st.session_state.current_vector_backend = st.selectbox("Vector backend", ["chroma", "faiss"], index=0 if settings.vector_store == "chroma" else 1)
        st.session_state.retrieval_top_k = st.slider("Retrieval top-k", min_value=1, max_value=10, value=st.session_state.retrieval_top_k)
        st.session_state.similarity_threshold = st.number_input("Similarity threshold", value=st.session_state.similarity_threshold or 0.25, min_value=0.0, max_value=1.0, step=0.01)
        st.session_state.groq_model = st.text_input("Groq model", value=st.session_state.groq_model)

        metadata_filter = st.text_input("Optional metadata filter (JSON)", value="{}")
        st.caption("Example: {\"source_file\": \"report.pdf\"}")
        st.session_state.metadata_filter = metadata_filter

    if st.button("Index documents"):
        _index_documents()

    if st.session_state.get("indexed_chunks"):
        preview = st.session_state["indexed_chunks"][:5]
        preview_table = pd.DataFrame(
            [
                {
                    "Source": chunk["metadata"].get("source_file", "unknown"),
                    "Page": chunk["metadata"].get("page_number", "-"),
                    "Length": len(chunk["text"]),
                    "Text": chunk["text"][:160],
                }
                for chunk in preview
            ]
        )
        st.dataframe(preview_table, use_container_width=True)


def render_chat() -> None:
    st.subheader("Chat interface")
    if st.button("Clear chat history"):
        st.session_state.chat_history = []

    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message.get("sources"):
                with st.expander("View source references"):
                    for source in message["sources"]:
                        score = source.get("score")
                        score_label = f"Score: {score:.3f} | " if score is not None else ""
                        st.markdown(f"**{score_label}{source.get('filename', 'Unknown file')}**")
                        if source.get("page_number") is not None:
                            st.caption(f"Page {source['page_number']}")
                        st.write(source.get("text", ""))

    question = st.chat_input("Ask a question about your uploaded documents")
    if question:
        pipeline = _build_pipeline()
        if pipeline is None:
            return
        if not st.session_state.chat_history:
            st.session_state.chat_history = []
        recent_history = [message["content"] for message in st.session_state.chat_history[-4:]]
        answer = pipeline.answer_question(question, chat_history=recent_history)
        st.session_state.chat_history.append({"role": "user", "content": question})
        st.session_state.chat_history.append({"role": "assistant", "content": answer["answer"], "sources": answer.get("sources", [])})
        with st.chat_message("user"):
            st.markdown(question)
        with st.chat_message("assistant"):
            st.markdown(answer["answer"])
            with st.expander("View source references"):
                for source in answer.get("sources", []):
                    score = source.get("score")
                    score_label = f"Score: {score:.3f} | " if score is not None else ""
                    st.markdown(f"**{score_label}{source.get('filename', 'Unknown file')}**")
                    if source.get("page_number") is not None:
                        st.caption(f"Page {source['page_number']}")
                    st.write(source.get("text", ""))


def render_app(embed: bool = False) -> None:
    _init_session_state()
    if not embed:
        st.set_page_config(page_title="Intelligent Document Intelligence — RAG Chatbot", layout="wide")
        render_home()
    st.write("---")
    render_documents()
    render_controls()
    render_chat()


if __name__ == "__main__":
    render_app(embed=False)
