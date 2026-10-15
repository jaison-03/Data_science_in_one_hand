from __future__ import annotations

from io import BytesIO

import fitz
from PIL import Image

from core.vector_store import FAISSVectorStore
from image_pdf_rag import ImagePdfRAGPipeline, OCRResult
from image_pdf_rag import app as scanned_app
from image_pdf_rag.ocr import clean_ocr_text, get_tesseract_status


class FakeEmbeddingService:
    def embed_documents(self, texts):
        return [[1.0, 0.0, 0.0] for _ in texts]

    def embed_query(self, question):
        return [1.0, 0.0, 0.0]


class FakeOCREngine:
    def extract(self, image):
        return OCRResult("Screenshot heading\nThe answer is forty two.", 96.0)


class FakeLLMService:
    def generate(self, prompt):
        return "The answer is forty two."


def _pipeline(ocr_engine=None):
    return ImagePdfRAGPipeline(
        embedding_service=FakeEmbeddingService(),
        vector_store=FAISSVectorStore(dimension=3),
        llm_service=FakeLLMService(),
        ocr_engine=ocr_engine,
        chunk_size=200,
        chunk_overlap=0,
        top_k=1,
    )


def test_clean_ocr_text_joins_hyphenated_line_breaks_and_removes_noise():
    assert clean_ocr_text("inter-\nnational\n---\ntext") == "international\ntext"


def test_auto_routing_ocr_indexes_image_only_pdf_and_returns_page_source(tmp_path):
    pdf_path = tmp_path / "scan.pdf"
    pdf = fitz.open()
    page = pdf.new_page()
    image_buffer = BytesIO()
    Image.new("RGB", (64, 64), "white").save(image_buffer, format="PNG")
    page.insert_image(
        fitz.Rect(0, 0, page.rect.width, page.rect.height),
        stream=image_buffer.getvalue(),
    )
    pdf.save(pdf_path)
    pdf.close()

    pipeline = _pipeline(FakeOCREngine())
    indexed = pipeline.index_pdf(pdf_path)
    result = pipeline.answer_question("What is the answer?")

    assert indexed["ocr_page_count"] == 1
    assert indexed["chunk_count"] == 1
    assert indexed["chunks"][0]["metadata"]["page_number"] == 1
    assert result["answer"] == "The answer is forty two."
    assert result["sources"][0]["page_image"] == "scan.pdf#page=1"
    pipeline.index_pdf(pdf_path)
    assert pipeline.vector_store.count() == 1


def test_auto_routing_keeps_native_text_without_ocr(tmp_path):
    pdf_path = tmp_path / "digital.pdf"
    pdf = fitz.open()
    page = pdf.new_page()
    page.insert_textbox(
        fitz.Rect(50, 50, 500, 150),
        "This digital PDF has a sufficiently long line of text to avoid OCR.",
    )
    pdf.save(pdf_path)
    pdf.close()

    indexed = _pipeline().index_pdf(pdf_path)

    assert indexed["ocr_page_count"] == 0
    assert indexed["chunks"][0]["metadata"]["extraction_method"] == "native"


def test_missing_pdf_is_reported():
    pipeline = _pipeline()

    try:
        pipeline.index_pdf("does-not-exist.pdf")
    except FileNotFoundError as exc:
        assert "PDF not found" in str(exc)
    else:
        raise AssertionError("Expected a missing-file error.")


def test_chroma_unavailable_falls_back_to_faiss(monkeypatch):
    def unavailable_chroma(**kwargs):
        raise ModuleNotFoundError("ChromaDB is not installed.")

    monkeypatch.setattr(scanned_app, "ChromaVectorStore", unavailable_chroma)
    vector_store, active_backend = scanned_app._get_vector_store("chroma")

    assert isinstance(vector_store, FAISSVectorStore)
    assert active_backend == "faiss"


def test_tesseract_status_explains_missing_ocr_application(monkeypatch):
    monkeypatch.setattr("image_pdf_rag.ocr._find_tesseract", lambda pytesseract: None)

    ready, message = get_tesseract_status()

    assert ready is False
    assert "Tesseract OCR application is not installed" in message
    assert "choco install tesseract -y" in message
