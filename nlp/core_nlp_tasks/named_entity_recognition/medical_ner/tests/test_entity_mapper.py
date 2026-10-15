"""Tests for AWS response normalization."""

from nlp.core_nlp_tasks.medical_ner.core.entity_mapper import map_entities_response


def test_maps_supported_fields_without_inventing_optional_values():
    response = {
        "Entities": [
            {
                "Id": 4,
                "Text": "aspirin",
                "Category": "MEDICATION",
                "Type": "GENERIC_NAME",
                "BeginOffset": 12,
                "EndOffset": 19,
                "Score": 0.98,
                "Traits": [{"Name": "NEGATION", "Score": 0.81}],
                "Attributes": [
                    {
                        "Id": 5,
                        "Text": "100 mg",
                        "Type": "DOSAGE",
                        "RelationshipType": "DOSAGE",
                        "RelationshipScore": 0.75,
                    }
                ],
                "RxNormConcepts": [{"Code": "1191", "Description": "aspirin", "Score": 0.9}],
            }
        ]
    }

    entity = map_entities_response(response)[0]

    assert entity["text"] == "aspirin"
    assert entity["category"] == "MEDICATION"
    assert entity["type"] == "GENERIC_NAME"
    assert entity["begin_offset"] == 12
    assert entity["end_offset"] == 19
    assert entity["score"] == 0.98
    assert entity["display_group"] == "Medication"
    assert entity["attributes"][0]["relationshipType"] == "DOSAGE"
    assert entity["concepts"][0]["code"] == "1191"


def test_missing_optional_response_fields_remain_absent_or_empty():
    entities = map_entities_response({"Entities": [{"Text": "pain"}]})
    assert entities[0]["text"] == "pain"
    assert "score" not in entities[0]
    assert entities[0]["traits"] == []
    assert entities[0]["concepts"] == []


def test_empty_entity_response_is_empty():
    assert map_entities_response({"Entities": []}) == []
