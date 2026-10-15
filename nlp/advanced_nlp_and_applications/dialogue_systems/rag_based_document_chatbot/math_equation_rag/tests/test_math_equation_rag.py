from __future__ import annotations

from pathlib import Path

import fitz
import pytest
from langchain_text_splitters import RecursiveCharacterTextSplitter

from core.vector_store import FAISSVectorStore
from math_equation_rag.extractor import _split_markdown_pages, extract_pdf
from math_equation_rag.pipeline import MathEquationRAGPipeline, _split_preserving_equations


class FakeEmbeddingService:
    def embed_documents(self, texts):
        return [[1.0, 0.0, 0.0] for _ in texts]

    def embed_query(self, question):
        return [1.0, 0.0, 0.0]


class FakeLLMService:
    def generate(self, prompt):
        assert "x^2 + y^2 = z^2" in prompt
        return "The excerpt gives the Pythagorean theorem."


def _make_pdf(path: Path, page_texts: list[str]) -> None:
    with fitz.open() as pdf:
        for text in page_texts:
            page = pdf.new_page()
            page.insert_textbox(fitz.Rect(50, 50, 550, 250), text)
        pdf.save(path)


def test_native_extraction_preserves_equation_text_and_page_number(tmp_path):
    path = tmp_path / "theorem.pdf"
    _make_pdf(path, ["Pythagorean theorem\nx^2 + y^2 = z^2"])

    pages = extract_pdf(path)

    assert len(pages) == 1
    assert pages[0].page_number == 1
    assert "x^2 + y^2 = z^2" in pages[0].text


def test_marker_page_comments_are_used_for_source_attribution():
    pages = _split_markdown_pages(
        "<!-- Page 4 -->\nFirst equation: $a+b=c$\n\n<!-- Page 9 -->\nSecond result."
    )

    assert [(page.page_number, page.text) for page in pages] == [
        (4, "First equation: $a+b=c$"),
        (9, "Second result."),
    ]


def test_chunking_keeps_display_equations_whole():
    equation = r"\begin{align}x_1 + x_2 &= 10 \\ x_1 - x_2 &= 2\end{align}"
    text = f"{'Background text. ' * 30}\n\n{equation}\n\n{'Explanation text. ' * 30}"
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=300,
        chunk_overlap=0,
        separators=["\n\n", "\n", " ", ""],
    )

    chunks = _split_preserving_equations(text, splitter)

    assert sum(equation in chunk for chunk in chunks) == 1


def test_native_extraction_rejects_image_only_pdf_with_actionable_message(tmp_path):
    path = tmp_path / "blank.pdf"
    with fitz.open() as pdf:
        pdf.new_page()
        pdf.save(path)

    with pytest.raises(ValueError, match="Choose Marker or Nougat"):
        extract_pdf(path)


def test_pipeline_indexes_and_answers_from_equation_passages(tmp_path):
    path = tmp_path / "theorem.pdf"
    _make_pdf(path, ["Pythagorean theorem\nx^2 + y^2 = z^2"])
    store = FAISSVectorStore(dimension=3)
    pipeline = MathEquationRAGPipeline(
        embedding_service=FakeEmbeddingService(),
        vector_store=store,
        llm_service=FakeLLMService(),
        chunk_size=500,
        chunk_overlap=0,
        top_k=1,
    )

    indexed = pipeline.index_pdf(path)
    answer = pipeline.answer_question("What equation is shown?")

    assert indexed["chunk_count"] == 1
    assert indexed["chunks"][0]["metadata"]["extraction_method"] == "native"
    assert answer["answer"] == "The excerpt gives the Pythagorean theorem."
    assert answer["sources"][0]["page_number"] == 1
