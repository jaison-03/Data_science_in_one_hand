"""Input limit and whitespace validation tests."""

import pytest

from nlp.core_nlp_tasks.medical_ner.core.validators import (
    validate_comprehend_text,
    validate_concept_text,
)


def test_empty_and_whitespace_inputs_are_rejected():
    for value in ("", "  \n\t"):
        with pytest.raises(ValueError, match="non-empty"):
            validate_comprehend_text(value)


def test_detect_entities_uses_utf8_byte_limit_and_does_not_trim_offsets():
    assert validate_comprehend_text("  note  ") == "  note  "
    with pytest.raises(ValueError, match="UTF-8 bytes"):
        validate_comprehend_text("é" * 10_000)


def test_concept_operations_enforce_character_limit():
    with pytest.raises(ValueError, match="10,000 characters"):
        validate_concept_text("x" * 10_001)
