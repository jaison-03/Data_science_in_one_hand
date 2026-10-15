from __future__ import annotations

from typing import Any

from langchain_text_splitters import RecursiveCharacterTextSplitter

from core.document_loader import LoadedDocument
from utils.helpers import stable_chunk_id
from utils.validators import validate_chunk_overlap, validate_chunk_size


def chunk_document(document: LoadedDocument, chunk_size: int, chunk_overlap: int) -> list[dict[str, Any]]:
    validate_chunk_size(chunk_size)
    validate_chunk_overlap(chunk_overlap, chunk_size)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    chunks: list[dict[str, Any]] = []
    pages = document.pages or [document.content]

    for page_number, page_text in enumerate(pages, start=1):
        if not page_text or not page_text.strip():
            continue
        split_pages = splitter.split_text(page_text)
        for chunk_index, text in enumerate(split_pages):
            cleaned = text.strip()
            if not cleaned:
                continue
            chunk_id = stable_chunk_id(document.filename, page_number, chunk_index)
            chunks.append(
                {
                    "chunk_id": chunk_id,
                    "text": cleaned,
                    "metadata": {
                        "source_file": document.filename,
                        "document_type": document.document_type,
                        "page_number": page_number,
                        "source_path": document.source_path,
                        "chunk_index": chunk_index,
                        "chunk_size": len(cleaned),
                    },
                }
            )

    if not chunks:
        raise ValueError("Document produced no valid chunks after preprocessing.")
    return chunks
