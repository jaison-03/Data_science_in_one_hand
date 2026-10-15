"""Tests for export formatting and field safety."""

import csv
import io
import json

from nlp.core_nlp_tasks.medical_ner.utils.export_utils import (
    entities_csv,
    result_json,
    result_summary_text,
)


def test_exports_handle_empty_entities_and_exclude_secrets():
    result = {
        "operation": "DetectEntitiesV2",
        "region": "us-east-1",
        "entities": [],
        "raw_response": {"Entities": []},
    }
    assert next(csv.reader(io.StringIO(entities_csv([]))))[0] == "text"
    assert json.loads(result_json(result))["entities"] == []
    assert "AWS_SECRET_ACCESS_KEY" not in result_summary_text(result)


def test_csv_and_json_include_returned_fields():
    entity = {
        "text": "fever",
        "category": "MEDICAL_CONDITION",
        "type": "SYMPTOM",
        "display_group": "Symptom",
        "begin_offset": 4,
        "end_offset": 9,
        "score": 0.91,
        "traits": [],
        "attributes": [],
        "concepts": [],
    }
    csv_result = list(csv.DictReader(io.StringIO(entities_csv([entity]))))
    assert csv_result[0]["text"] == "fever"
    assert json.loads(result_json({"entities": [entity]}))["entities"][0]["score"] == 0.91
