"""Input validation for AWS Comprehend Medical operations."""

from nlp.core_nlp_tasks.medical_ner.config.settings import (
    MAX_CONCEPT_CHARACTERS,
    MAX_DETECT_BYTES,
)


def validate_comprehend_text(text):
    if not isinstance(text, str):
        raise ValueError("Enter medical text as plain text.")
    if not text.strip():
        raise ValueError("Enter non-empty medical text before analysis.")
    encoded_length = len(text.encode("utf-8"))
    if encoded_length > MAX_DETECT_BYTES:
        raise ValueError(
            "DetectEntitiesV2 accepts fewer than 20,000 UTF-8 bytes. "
            f"Your text contains {encoded_length:,} bytes."
        )
    return text


def validate_concept_text(text):
    normalized = validate_comprehend_text(text)
    if len(normalized) > MAX_CONCEPT_CHARACTERS:
        raise ValueError(
            "ICD-10-CM and RxNorm lookups accept at most 10,000 characters. "
            "Shorten the text before requesting concept candidates."
        )
    return normalized
