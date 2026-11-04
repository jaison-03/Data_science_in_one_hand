from io import BytesIO
from unittest.mock import patch

import fitz
import pytest
from PIL import Image

from core.vector_store import FAISSVectorStore
from handwritten_pdf_rag.ocr import HandwritingOCRResult, TextractOCR
from handwritten_pdf_rag.pipeline import HandwrittenPdfRAGPipeline


class FakeEmbeddingService:
    def embed_documents(self, texts):
        return [[1.0, 0.0, 0.0] for _ in texts]

    def embed_query(self, question):
        return [1.0, 0.0, 0.0]


class FakeOCREngine:
    def __init__(self, result=None):
        self.result = result or HandwritingOCRResult(
            "Meeting notes: bring the signed form on Friday.", 87.5, "handwriting"
        )
        self.calls = 0

    def extract(self, image):
        self.calls += 1
        assert image.width > 0
        return self.result


class FakeLLMService:
    def __init__(self):
        self.prompt = None

    def generate(self, prompt):
        self.prompt = prompt
        return "The notes say to bring the signed form on Friday."


class FakeTextractClient:
    def detect_document_text(self, *, Document):
        assert Document["Bytes"].startswith(b"\x89PNG")
        return {
            "Blocks": [
                {"BlockType": "LINE", "Text": "First handwritten line", "Confidence": 96},
                {
                    "BlockType": "WORD",
                    "Text": "ignored",
                    "TextType": "HANDWRITING",
                },
                {"BlockType": "LINE", "Text": "Second line", "Confidence": 80},
            ]
        }


def _pipeline():
    return HandwrittenPdfRAGPipeline(
        embedding_service=FakeEmbeddingService(),
        vector_store=FAISSVectorStore(dimension=3),
        chunk_size=200,
        chunk_overlap=0,
        top_k=2,
    )


def test_index_native_pdf_without_calling_ocr(tmp_path):
    path = tmp_path / "digital.pdf"
    pdf = fitz.open()
    page = pdf.new_page()
    page.insert_text((72, 72), "Date: 3/9")
    pdf.save(path)
    pdf.close()
    engine = FakeOCREngine()

    result = _pipeline().index_pdf(path, "document-1", engine)

    assert engine.calls == 0
    assert result["page_count"] == 1
    assert result["indexed_page_count"] == 1
    assert result["pages"][0]["extraction_method"] == "native"
    assert result["chunks"][0]["metadata"]["page_number"] == 1
    assert result["chunks"][0]["metadata"]["document_id"] == "document-1"


def test_index_scanned_pdf_and_answer_with_page_source(tmp_path):
    path = tmp_path / "scanned.pdf"
    image_stream = BytesIO()
    Image.new("RGB", (600, 800), "white").save(image_stream, format="PNG")
    pdf = fitz.open()
    page = pdf.new_page(width=600, height=800)
    page.insert_image(page.rect, stream=image_stream.getvalue())
    pdf.save(path)
    pdf.close()
    engine = FakeOCREngine()
    pipeline = _pipeline()

    result = pipeline.index_pdf(path, "scan-1", engine)
    llm = FakeLLMService()
    pipeline.llm_service = llm
    answer = pipeline.answer_question("What should I bring?", "scan-1")

    assert engine.calls == 1
    assert result["ocr_page_count"] == 1
    assert result["pages"][0]["content_type"] == "handwritten"
    assert result["chunks"][0]["metadata"]["ocr_confidence"] == 87.5
    assert answer["grounded"] is True
    assert answer["sources"][0]["filename"] == "scanned.pdf"
    assert answer["sources"][0]["page_number"] == 1
    assert "do not guess unreadable content" in llm.prompt[1].content.lower()


def test_textract_extracts_ordered_lines_and_average_confidence():
    image = Image.new("RGB", (120, 80), "white")

    result = TextractOCR(client=FakeTextractClient()).extract(image)

    assert result.text == "First handwritten line\nSecond line"
    assert result.confidence == 88
    assert result.text_type == "handwriting"


def test_empty_pdf_fails_clearly(tmp_path):
    class EmptyPdf:
        is_encrypted = False
        page_count = 0

        def close(self):
            pass

    path = tmp_path / "empty.pdf"
    path.write_bytes(b"%PDF-1.7")
    with patch("handwritten_pdf_rag.pipeline.fitz.open", return_value=EmptyPdf()):
        with pytest.raises(ValueError, match="no pages"):
            _pipeline().index_pdf(path, "empty-1", FakeOCREngine())
