import pytest

from utils.validators import validate_chunk_overlap, validate_chunk_size, validate_question, validate_uploaded_file


def test_validate_question_rejects_empty_input():
    with pytest.raises(ValueError):
        validate_question("   ")


def test_validate_chunk_size_rejects_small_values():
    with pytest.raises(ValueError):
        validate_chunk_size(50)


def test_validate_chunk_overlap_rejects_invalid_values():
    with pytest.raises(ValueError):
        validate_chunk_overlap(500, 300)


def test_validate_uploaded_file_rejects_invalid_extension():
    with pytest.raises(ValueError):
        validate_uploaded_file("notes.csv", 1024)
