from __future__ import annotations

from pathlib import Path

SUPPORTED_EXTENSIONS = {".pdf", ".txt", ".md", ".markdown"}
MAX_FILE_SIZE_BYTES = 15 * 1024 * 1024


def validate_question(question: str) -> str:
    if question is None:
        raise ValueError("Question cannot be empty.")
    cleaned = question.strip()
    if not cleaned:
        raise ValueError("Question cannot be empty.")
    if len(cleaned) < 2:
        raise ValueError("Question is too short to be useful.")
    return cleaned


def validate_chunk_size(size: int) -> int:
    if not isinstance(size, int):
        raise TypeError("Chunk size must be an integer.")
    if size < 100 or size > 4000:
        raise ValueError("Chunk size must be between 100 and 4000 characters.")
    return size


def validate_chunk_overlap(overlap: int, chunk_size: int) -> int:
    if not isinstance(overlap, int):
        raise TypeError("Chunk overlap must be an integer.")
    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("Chunk overlap must be non-negative and smaller than chunk size.")
    return overlap


def validate_file_type(filename: str) -> str:
    path = Path(filename)
    ext = path.suffix.lower()
    if ext not in SUPPORTED_EXTENSIONS:
        raise ValueError(f"Unsupported file type '{ext}'. Allowed: {', '.join(sorted(SUPPORTED_EXTENSIONS))}.")
    return ext


def validate_uploaded_file(file_name: str, file_size_bytes: int) -> None:
    validate_file_type(file_name)
    if file_size_bytes <= 0:
        raise ValueError("Uploaded file is empty.")
    if file_size_bytes > MAX_FILE_SIZE_BYTES:
        raise ValueError(f"File exceeds the {MAX_FILE_SIZE_BYTES / (1024 * 1024):.0f} MB limit.")


def is_valid_text(value: str | None) -> bool:
    return bool(value and value.strip())
