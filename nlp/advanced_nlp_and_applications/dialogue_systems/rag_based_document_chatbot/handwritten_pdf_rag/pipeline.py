from __future__ import annotations

import re
from io import BytesIO
from pathlib import Path
from typing import Any, Callable, Literal

import fitz
from langchain_core.messages import HumanMessage
from PIL import Image
from langchain_text_splitters import RecursiveCharacterTextSplitter

from config.settings import settings
from core.embeddings import EmbeddingService
from core.llm_service import LLMService
from core.prompts import build_rag_prompt
from core.retriever import Retriever
from core.vector_store import BaseVectorStore
from utils.helpers import stable_chunk_id
from utils.validators import validate_chunk_overlap, validate_chunk_size, validate_question
from .ocr import HandwritingOCREngine, HandwritingOCRResult

OCRMode = Literal["textract", "local"]


class HandwrittenPdfRAGPipeline:
    """Extract handwriting page by page, index it, and answer with page citations."""

    def __init__(
        self,
        embedding_service: Any,
        vector_store: BaseVectorStore,
        llm_service: Any | None = None,
        *,
        chunk_size: int | None = None,
        chunk_overlap: int | None = None,
        top_k: int | None = None,
        dpi: int = 300,
        image_coverage_threshold: float = 0.2,
    ):
        self.embedding_service = embedding_service
        self.vector_store = vector_store
        self.llm_service = llm_service
        self.chunk_size = settings.chunk_size if chunk_size is None else chunk_size
        self.chunk_overlap = (
            settings.chunk_overlap if chunk_overlap is None else chunk_overlap
        )
        self.top_k = settings.retrieval_top_k if top_k is None else top_k
        if dpi < 1:
            raise ValueError("dpi must be a positive integer.")
        validate_chunk_size(self.chunk_size)
        validate_chunk_overlap(self.chunk_overlap, self.chunk_size)
        if not isinstance(self.top_k, int) or not 1 <= self.top_k <= 20:
            raise ValueError("top_k must be an integer between 1 and 20.")
        self.dpi = dpi
        self.image_coverage_threshold = image_coverage_threshold

    def index_pdf(
        self,
        file_path: str | Path,
        document_id: str,
        ocr_engine: HandwritingOCREngine,
        *,
        on_page: Callable[[int, int], None] | None = None,
    ) -> dict[str, Any]:
        path = Path(file_path)
        if not path.is_file():
            raise FileNotFoundError(f"PDF not found: {path}")
        if path.suffix.lower() != ".pdf":
            raise ValueError(f"Expected a PDF file, received '{path.name}'.")
        if not document_id:
            raise ValueError("document_id must not be empty.")
        try:
            pdf = fitz.open(path)
        except Exception as exc:
            raise ValueError(f"Could not open PDF '{path.name}': {exc}") from exc
        if pdf.is_encrypted:
            pdf.close()
            raise ValueError(f"PDF '{path.name}' is encrypted and cannot be processed.")
        if pdf.page_count == 0:
            pdf.close()
            raise ValueError(f"PDF '{path.name}' contains no pages.")

        page_records: list[dict[str, Any]] = []
        page_count = pdf.page_count
        try:
            for page_index in range(page_count):
                page_number = page_index + 1
                page = pdf.load_page(page_index)
                native_text = self._clean_text(page.get_text("text"))
                needs_ocr = (
                    not native_text.strip()
                    or self._image_coverage(page) >= self.image_coverage_threshold
                )
                text = native_text
                method = "native"
                confidence = None
                ocr_text_type = "digital_text"
                if needs_ocr:
                    pixmap = page.get_pixmap(dpi=self.dpi, alpha=False)
                    with Image.open(BytesIO(pixmap.tobytes("png"))) as rendered:
                        ocr_result = ocr_engine.extract(rendered.copy())
                    if not isinstance(ocr_result, HandwritingOCRResult):
                        raise TypeError(
                            "OCR engine extract() must return HandwritingOCRResult."
                        )
                    text = self._merge_text(native_text, ocr_result.text)
                    method = "mixed_ocr" if native_text.strip() else "ocr"
                    confidence = ocr_result.confidence
                    ocr_text_type = ocr_result.text_type
                if on_page is not None:
                    on_page(page_number, page_count)
                if native_text.strip() and needs_ocr:
                    content_type = "mixed"
                elif needs_ocr and ocr_text_type == "handwriting":
                    content_type = "handwritten"
                elif needs_ocr and ocr_text_type == "printed":
                    content_type = "printed_scan"
                elif needs_ocr and ocr_text_type == "mixed":
                    content_type = "mixed_scan"
                elif needs_ocr:
                    content_type = "scanned_unclassified"
                else:
                    content_type = "digital_text"
                page_records.append(
                    {
                        "page_number": page_number,
                        "text": text,
                        "content_type": content_type,
                        "extraction_method": method,
                        "ocr_confidence": confidence,
                        "ocr_text_type": ocr_text_type,
                        "needs_review": needs_ocr
                        and (
                            not text.strip()
                            or ocr_text_type == "unknown"
                            or (confidence is not None and confidence < 60)
                        ),
                    }
                )
        finally:
            pdf.close()

        chunks = self._chunk_pages(path, document_id, page_records)
        if not chunks:
            raise ValueError(
                f"PDF '{path.name}' contains no readable text after extraction."
            )

        source_key = f"{document_id}:{path.name}"
        try:
            embeddings = self.embedding_service.embed_documents(
                [chunk["text"] for chunk in chunks]
            )
            if len(embeddings) != len(chunks):
                raise ValueError(
                    "Embedding service returned a different number of vectors than chunks."
                )
            self.vector_store.delete_document(source_key)
            self.vector_store.add_documents(chunks, embeddings)
        except Exception as exc:
            raise RuntimeError(f"Failed to index PDF '{path.name}': {exc}") from exc

        return {
            "filename": path.name,
            "document_id": document_id,
            "page_count": page_count,
            "indexed_page_count": sum(bool(record["text"].strip()) for record in page_records),
            "ocr_page_count": sum(record["extraction_method"] != "native" for record in page_records),
            "pages": page_records,
            "chunk_count": len(chunks),
            "chunks": chunks,
        }

    def answer_question(
        self,
        question: str,
        document_id: str,
        chat_history: list[str] | None = None,
    ) -> dict[str, Any]:
        cleaned_question = validate_question(question)
        retriever = Retriever(
            vector_store=self.vector_store,
            embedding_service=self.embedding_service,
            top_k=self.top_k,
            threshold=settings.similarity_threshold,
        )
        results = retriever.search(
            cleaned_question,
            metadata_filter={"document_id": document_id},
        ).results
        if not results:
            return {
                "answer": (
                    "I could not find enough readable information in this PDF to answer "
                    "confidently. Handwritten text may be unclear or unreadable; inspect "
                    "the cited page or provide a clearer scan."
                ),
                "sources": [],
                "grounded": False,
                "retrieval_results": [],
            }

        context_parts = []
        for result in results:
            metadata = result.metadata or {}
            filename = metadata.get("display_filename")
            if not filename:
                filename = metadata.get("source_file", "unknown source").split(":", 1)[-1]
            context_parts.append(
                f"[Source: {filename} "
                f"(Page {metadata.get('page_number', 'unknown')}; "
                f"extraction: {metadata.get('extraction_method', 'unknown')}; "
                f"OCR confidence: {metadata.get('ocr_confidence', 'not available')})]\n"
                f"{result.text}"
            )
        context = "\n\n---\n\n".join(context_parts)
        history = [] if not chat_history else chat_history[-4:]
        prompt = build_rag_prompt(cleaned_question, context, history)
        prompt[-1] = HumanMessage(
            content=prompt[-1].content
            + "\n\nThis is a handwriting-focused document. Treat OCR as potentially "
            "uncertain, especially names, dates, decimals, negative signs, and "
            "mathematical symbols. Do not guess unreadable content. Say when the "
            "evidence is ambiguous, and distinguish verbatim extracted text from "
            "your interpretation."
        )
        generated = (self.llm_service or LLMService()).generate(prompt)
        return {
            "answer": generated,
            "sources": [
                {
                    "chunk_id": result.chunk_id,
                    "filename": (result.metadata or {}).get("display_filename", "unknown"),
                    "page_number": (result.metadata or {}).get("page_number"),
                    "source_path": (result.metadata or {}).get("source_path"),
                    "content_type": (result.metadata or {}).get("content_type", "unknown"),
                    "extraction_method": (result.metadata or {}).get(
                        "extraction_method", "unknown"
                    ),
                    "ocr_confidence": (result.metadata or {}).get("ocr_confidence"),
                    "text": result.text,
                    "score": result.score,
                }
                for result in results
            ],
            "grounded": True,
            "retrieval_results": [
                {
                    "chunk_id": result.chunk_id,
                    "text": result.text,
                    "metadata": result.metadata,
                    "score": result.score,
                }
                for result in results
            ],
        }

    @staticmethod
    def _clean_text(text: str) -> str:
        lines = [re.sub(r"\s+", " ", line).strip() for line in text.splitlines()]
        return "\n".join(line for line in lines if line)

    @staticmethod
    def _merge_text(native_text: str, ocr_text: str) -> str:
        lines: list[str] = []
        seen: set[str] = set()
        for line in (native_text + "\n" + ocr_text).splitlines():
            cleaned = line.strip()
            key = re.sub(r"\W", "", cleaned.casefold(), flags=re.UNICODE)
            if cleaned and key and key not in seen:
                lines.append(cleaned)
                seen.add(key)
        return "\n".join(lines)

    @staticmethod
    def _image_coverage(page: fitz.Page) -> float:
        page_area = max(page.rect.width * page.rect.height, 1.0)
        image_area = 0.0
        for image in page.get_image_info(xrefs=True):
            bbox = fitz.Rect(image["bbox"]) & page.rect
            image_area += max(0.0, bbox.width) * max(0.0, bbox.height)
        return min(1.0, image_area / page_area)

    def _chunk_pages(
        self,
        path: Path,
        document_id: str,
        pages: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""],
        )
        source_key = f"{document_id}:{path.name}"
        chunks = []
        for page_record in pages:
            page_number = page_record["page_number"]
            for chunk_index, text in enumerate(splitter.split_text(page_record["text"])):
                cleaned = text.strip()
                if not cleaned:
                    continue
                chunk_id = stable_chunk_id(
                    f"handwritten:{document_id}:{path.name}",
                    page_number,
                    str(chunk_index),
                )
                metadata: dict[str, Any] = {
                    "source_file": source_key,
                    "display_filename": path.name,
                    "source_path": str(path.resolve()),
                    "document_id": document_id,
                    "document_type": "handwritten_pdf",
                    "content_type": page_record["content_type"],
                    "ocr_text_type": page_record["ocr_text_type"],
                    "page_number": page_number,
                    "chunk_index": chunk_index,
                    "chunk_size": len(cleaned),
                    "extraction_method": page_record["extraction_method"],
                }
                if page_record["ocr_confidence"] is not None:
                    metadata["ocr_confidence"] = float(page_record["ocr_confidence"])
                chunks.append(
                    {"chunk_id": chunk_id, "text": cleaned, "metadata": metadata}
                )
        return chunks
