from core.chunking import chunk_document
from core.document_loader import LoadedDocument


def test_chunk_document_returns_expected_number_of_chunks():
    document = LoadedDocument(
        filename="sample.txt",
        source_path="sample.txt",
        content="This is a test document. " * 20,
        document_type="txt",
        pages=["This is a test document. " * 20],
        page_number=1,
    )

    chunks = chunk_document(document, chunk_size=200, chunk_overlap=20)
    assert len(chunks) > 0
    assert all("text" in chunk for chunk in chunks)
    assert all("chunk_id" in chunk for chunk in chunks)
