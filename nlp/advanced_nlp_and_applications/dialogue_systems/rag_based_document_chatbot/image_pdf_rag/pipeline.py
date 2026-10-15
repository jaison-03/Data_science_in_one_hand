from __future__ import annotations

import re
from io import BytesIO
from pathlib import Path
from typing import Any, Literal

import fitz
from PIL import Image
from langchain_text_splitters import RecursiveCharacterTextSplitter

from config.settings import settings
from core.embeddings import EmbeddingService
from core.llm_service import LLMService
from core.prompts import build_rag_prompt
from core.retriever import Retriever
from core.vector_store import BaseVectorStore, create_vector_store
from .ocr import OCREngine, OCRResult, TesseractOCR, clean_ocr_text
from utils.helpers import stable_chunk_id
from utils.logging_config import get_logger
from utils.validators import validate_chunk_overlap, validate_chunk_size, validate_question

logger = get_logger(__name__)
RoutingMode = Literal["auto", "ocr", "native"]


class ImagePdfRAGPipeline:
    """OCR, index, retrieve, and answer questions about PDF page images."""

    def __init__(
        self,
        embedding_service: Any | None = None,
        vector_store: BaseVectorStore | None = None,
        llm_service: Any | None = None,
        ocr_engine: OCREngine | None = None,
        *,
        chunk_size: int | None = None,
        chunk_overlap: int | None = None,
        top_k: int | None = None,
        dpi: int = 250,
        image_coverage_threshold: float = 0.20,
        min_native_characters: int = 40,
    ):
        self.embedding_service = embedding_service or EmbeddingService()
        self.vector_store = vector_store or create_vector_store()
        self.llm_service = llm_service
        self.ocr_engine = ocr_engine
        self.chunk_size = settings.chunk_size if chunk_size is None else chunk_size
        self.chunk_overlap = chunk_overlap if chunk_overlap is not None else settings.chunk_overlap
        self.top_k = settings.retrieval_top_k if top_k is None else top_k
        validate_chunk_size(self.chunk_size)
        validate_chunk_overlap(self.chunk_overlap, self.chunk_size)
        if not isinstance(self.top_k, int) or not 1 <= self.top_k <= 20:
            raise ValueError("top_k must be an integer between 1 and 20.")
        if dpi < 1:
            raise ValueError("dpi must be a positive integer.")
        if not 0 <= image_coverage_threshold <= 1:
            raise ValueError("image_coverage_threshold must be between 0 and 1.")
        self.dpi = dpi
        self.image_coverage_threshold = image_coverage_threshold
        self.min_native_characters = min_native_characters

    def index_pdf(
        self,
        file_path: str | Path,
        routing: RoutingMode = "auto",
    ) -> dict[str, Any]:
        path = Path(file_path)
        if not path.is_file():
            raise FileNotFoundError(f"PDF not found: {path}")
        if path.suffix.lower() != ".pdf":
            raise ValueError(f"Expected a PDF file, received '{path.name}'.")
        if routing not in {"auto", "ocr", "native"}:
            raise ValueError("routing must be one of 'auto', 'ocr', or 'native'.")

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
        total_page_count = pdf.page_count

        pages: list[dict[str, Any]] = []
        ocr_page_count = 0
        try:
            for page_index in range(pdf.page_count):
                page = pdf.load_page(page_index)
                page_number = page_index + 1
                native_text = page.get_text("text")
                use_ocr = routing == "ocr" or (
                    routing == "auto" and self._needs_ocr(page, native_text)
                )
                text = native_text
                confidence: float | None = None
                method = "native"

                if use_ocr:
                    if self.ocr_engine is None:
                        self.ocr_engine = TesseractOCR()
                    try:
                        pixmap = page.get_pixmap(dpi=self.dpi, alpha=False)
                        with Image.open(BytesIO(pixmap.tobytes("png"))) as rendered:
                            ocr_result = self.ocr_engine.extract(rendered.copy())
                    except Exception as exc:
                        raise RuntimeError(
                            f"Failed to OCR page {page_number} of '{path.name}': {exc}"
                        ) from exc
                    if not isinstance(ocr_result, OCRResult):
                        raise TypeError("OCR engine extract() must return an OCRResult.")
                    text = self._merge_page_text(native_text, ocr_result.text)
                    confidence = ocr_result.confidence
                    method = "ocr"
                    ocr_page_count += 1

                cleaned = clean_ocr_text(text)
                if cleaned:
                    pages.append(
                        {
                            "page_number": page_number,
                            "text": cleaned,
                            "extraction_method": method,
                            "ocr_confidence": confidence,
                        }
                    )
                else:
                    logger.warning("No readable text found on page %s of %s", page_number, path.name)
        finally:
            pdf.close()

        chunks = self._chunk_pages(path, pages)
        if not chunks:
            raise ValueError(f"PDF '{path.name}' contains no readable text after extraction.")

        try:
            embeddings = self.embedding_service.embed_documents([chunk["text"] for chunk in chunks])
            if len(embeddings) != len(chunks):
                raise ValueError("Embedding service returned a different number of vectors than chunks.")
            self.vector_store.delete_document(path.name)
            self.vector_store.add_documents(chunks, embeddings)
        except Exception as exc:
            raise RuntimeError(f"Failed to index PDF '{path.name}': {exc}") from exc

        return {
            "filename": path.name,
            "page_count": total_page_count,
            "indexed_page_count": len(pages),
            "ocr_page_count": ocr_page_count,
            "chunk_count": len(chunks),
            "chunks": chunks,
        }

    def answer_question(
        self,
        question: str,
        chat_history: list[str] | None = None,
        metadata_filter: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        cleaned_question = validate_question(question)
        retriever = Retriever(
            vector_store=self.vector_store,
            embedding_service=self.embedding_service,
            top_k=self.top_k,
            threshold=settings.similarity_threshold,
        )
        results = retriever.search(cleaned_question, metadata_filter=metadata_filter).results
        if not results:
            return {
                "answer": (
                    "I could not find enough relevant information in the indexed PDF pages "
                    "to answer that question confidently."
                ),
                "sources": [],
                "grounded": False,
                "retrieval_results": [],
            }

        context_chunks = []
        for result in results:
            metadata = result.metadata or {}
            context_chunks.append(
                f"[Source: {metadata.get('source_file', 'unknown source')} "
                f"(Page {metadata.get('page_number', 'unknown')})]\n{result.text}"
            )
        context = "\n\n---\n\n".join(context_chunks)
        history = [] if not chat_history else chat_history[-4:]
        prompt = build_rag_prompt(cleaned_question, context, history)
        generated = (self.llm_service or LLMService()).generate(prompt)

        return {
            "answer": generated,
            "sources": [
                {
                    "chunk_id": result.chunk_id,
                    "filename": (result.metadata or {}).get("source_file", "unknown"),
                    "page_number": (result.metadata or {}).get("page_number"),
                    "page_image": (result.metadata or {}).get("source_page_image"),
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

    def _needs_ocr(self, page: fitz.Page, native_text: str) -> bool:
        native_characters = len(re.sub(r"\W", "", native_text, flags=re.UNICODE))
        return (
            native_characters < self.min_native_characters
            or self._image_coverage(page) >= self.image_coverage_threshold
        )

    @staticmethod
    def _image_coverage(page: fitz.Page) -> float:
        page_area = max(page.rect.width * page.rect.height, 1.0)
        image_area = 0.0
        for image in page.get_image_info(xrefs=True):
            bbox = fitz.Rect(image["bbox"]) & page.rect
            image_area += max(0.0, bbox.width) * max(0.0, bbox.height)
        return min(1.0, image_area / page_area)

    @staticmethod
    def _merge_page_text(native_text: str, ocr_text: str) -> str:
        lines: list[str] = []
        seen: set[str] = set()
        for line in (native_text + "\n" + ocr_text).splitlines():
            cleaned = line.strip()
            key = re.sub(r"\W", "", cleaned.casefold(), flags=re.UNICODE)
            if cleaned and key and key not in seen:
                lines.append(cleaned)
                seen.add(key)
        return "\n".join(lines)

    def _chunk_pages(self, path: Path, pages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""],
        )
        chunks: list[dict[str, Any]] = []
        for page in pages:
            for chunk_index, text in enumerate(splitter.split_text(page["text"])):
                cleaned = text.strip()
                if not cleaned:
                    continue
                page_number = page["page_number"]
                chunk_id = stable_chunk_id(
                    f"image_pdf:{path.name}", page_number, str(chunk_index)
                )
                metadata: dict[str, Any] = {
                    "source_file": path.name,
                    "source_path": str(path.resolve()),
                    "source_page_image": f"{path.name}#page={page_number}",
                    "document_type": "image_pdf",
                    "page_number": page_number,
                    "chunk_index": chunk_index,
                    "chunk_size": len(cleaned),
                    "extraction_method": page["extraction_method"],
                }
                if page["ocr_confidence"] is not None:
                    metadata["ocr_confidence"] = float(page["ocr_confidence"])
                chunks.append({"chunk_id": chunk_id, "text": cleaned, "metadata": metadata})
        return chunks
