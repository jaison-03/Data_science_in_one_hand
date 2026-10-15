"""Tests for escaped, offset-based highlighted spans."""

from nlp.core_nlp_tasks.medical_ner.core.text_highlighter import highlight_entities


def entity(start, end, text, group="Medication"):
    return {
        "begin_offset": start,
        "end_offset": end,
        "text": text,
        "category": "MEDICATION",
        "type": "GENERIC_NAME",
        "display_group": group,
    }


def test_repeated_entity_occurrences_use_given_offsets():
    html = highlight_entities("aspirin and aspirin", [entity(12, 19, "aspirin")])
    assert html.count("<mark") == 1
    assert "aspirin and <mark" in html


def test_overlapping_spans_preserve_text_and_highlight_intersection():
    html = highlight_entities(
        "amoxicillin 500 mg",
        [entity(0, 18, "amoxicillin 500 mg"), entity(12, 18, "500 mg", "Dosage")],
    )
    assert "amoxicillin " in html
    assert html.count("<mark") == 2
    assert "amoxicillin 500 mg (MEDICATION / GENERIC_NAME), 500 mg" in html
    assert "500 mg" in html


def test_user_text_is_html_escaped():
    html = highlight_entities("<script>alert(1)</script>", [])
    assert html == "&lt;script&gt;alert(1)&lt;/script&gt;"
    assert "<script>" not in html


def test_empty_entities_leave_original_text_escaped():
    assert highlight_entities("safe & sound", []) == "safe &amp; sound"
