from __future__ import annotations

import re
from uuid import uuid4
from pathlib import Path
from typing import Any

import fitz
from langchain_text_splitters import RecursiveCharacterTextSplitter

from core.retriever import Retriever
from core.vector_store import BaseVectorStore
from math_equation_rag.extractor import extract_pdf
from utils.helpers import stable_chunk_id
from utils.validators import validate_chunk_overlap, validate_chunk_size, validate_question

_DISPLAY_EQUATION = re.compile(
    r"(\$\$.*?\$\$|\\\[.*?\\\]|"
    r"\\begin\{(?:equation\*?|align\*?|gather\*?|multline\*?|cases)\}"
    r".*?\\end\{(?:equation\*?|align\*?|gather\*?|multline\*?|cases)\})",
    re.DOTALL,
)


def _split_preserving_equations(text: str, splitter: RecursiveCharacterTextSplitter) -> list[str]:
    equations: dict[str, str] = {}

    def replace_equation(match: re.Match[str]) -> str:
        token = f"DISPLAYEQUATION{uuid4().hex}TOKEN"
        equations[token] = match.group(0)
        return token

    protected = _DISPLAY_EQUATION.sub(replace_equation, text)
    chunks = splitter.split_text(protected)
    token_pattern = re.compile(r"DISPLAYEQUATION[0-9a-f]{32}TOKEN")
    return [
        re.sub(
            token_pattern,
            lambda match: equations[match.group(0)],
            chunk,
        )
        for chunk in chunks
    ]


class MathEquationRAGPipeline:
    def __init__(
        self,
        embedding_service,
        vector_store: BaseVectorStore,
        llm_service=None,
        chunk_size: int = 1200,
        chunk_overlap: int = 150,
        top_k: int = 4,
    ):
        validate_chunk_size(chunk_size)
        validate_chunk_overlap(chunk_overlap, chunk_size)
        self.embedding_service = embedding_service
        self.vector_store = vector_store
        self.llm_service = llm_service
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.top_k = top_k

    def index_pdf(self, file_path: str | Path, engine: str = "native") -> dict[str, Any]:
        path = Path(file_path)
        pages = extract_pdf(path, engine=engine)
        if not pages:
            raise ValueError(f"No readable text or equations were extracted from '{path.name}'.")

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=["\n\n", "\n", " ", ""],
        )
        chunks: list[dict[str, Any]] = []
        for page_index, page in enumerate(pages, start=1):
            for chunk_index, text in enumerate(_split_preserving_equations(page.text, splitter)):
                text = text.strip()
                if not text:
                    continue
                page_number = page.page_number
                chunk_id = stable_chunk_id(
                    path.name,
                    len(chunks),
                    seed=str(page_number if page_number is not None else page_index),
                )
                metadata: dict[str, Any] = {
                    "source_file": path.name,
                    "document_type": "math_pdf",
                    "extraction_method": engine,
                    "chunk_index": chunk_index,
                }
                if page_number is not None:
                    metadata["page_number"] = page_number
                chunks.append({"chunk_id": chunk_id, "text": text, "metadata": metadata})

        if not chunks:
            raise ValueError(f"PDF '{path.name}' produced no indexable content.")

        with fitz.open(path) as pdf:
            page_count = pdf.page_count
        embeddings = self.embedding_service.embed_documents([chunk["text"] for chunk in chunks])
        self.vector_store.delete_document(path.name)
        self.vector_store.add_documents(chunks, embeddings)
        return {
            "filename": path.name,
            "engine": engine,
            "page_count": page_count,
            "indexed_page_count": len(pages),
            "chunk_count": len(chunks),
            "chunks": chunks,
        }

    def answer_question(
        self, question: str, chat_history: list[str] | None = None
    ) -> dict[str, Any]:
        if self.llm_service is None:
            raise ValueError("An LLM service is required to answer questions.")
        clean_question = validate_question(question)
        retriever = Retriever(
            vector_store=self.vector_store,
            embedding_service=self.embedding_service,
            top_k=self.top_k,
        )
        bundle = retriever.search(clean_question)
        if not bundle.results:
            return {
                "answer": (
                    "I could not find relevant text or equations in the indexed PDF. "
                    "Try rephrasing the question or using Marker/Nougat for image-based equations."
                ),
                "sources": [],
            }

        context_parts = []
        sources = []
        for result in bundle.results:
            metadata = result.metadata
            page_number = metadata.get("page_number")
            page_label = f", page {page_number}" if page_number is not None else ""
            context_parts.append(
                f"[Source: {metadata.get('source_file', 'PDF')}{page_label}]\n{result.text}"
            )
            sources.append(
                {
                    "filename": metadata.get("source_file", "PDF"),
                    "page_number": page_number,
                    "text": result.text,
                    "score": result.score,
                }
            )

        history = "\n".join(chat_history or [])
        context = "\n\n".join(context_parts)
        prompt = (
            "Answer the question using only the PDF excerpts below. Preserve mathematical "
            "notation, show inline formulas as $...$ and standalone equations as $$...$$, "
            "and explain symbols and steps when asked. Never output raw LaTeX commands "
            "without math delimiters. Do not invent formulas. If the excerpts do not contain "
            "the answer, say so clearly. Cite supporting pages when available.\n\n"
            f"Recent conversation:\n{history or '(none)'}\n\n"
            f"PDF excerpts:\n{context}\n\n"
            f"Question: {clean_question}"
        )
        return {"answer": self.llm_service.generate(prompt), "sources": sources}
