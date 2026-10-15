from pathlib import Path

from core.document_loader import load_document


def test_load_document_from_text_file(tmp_path):
    file_path = tmp_path / "sample.txt"
    file_path.write_text("This is a sample document.\nIt contains useful information for testing.", encoding="utf-8")

    document = load_document(file_path)
    assert document.filename == "sample.txt"
    assert "sample document" in document.content.lower()
    assert document.document_type == "txt"
