"""Normalize AWS entities while preserving only source-returned values."""


CATEGORY_GROUPS = {
    "MEDICATION": "Medication",
    "MEDICAL_CONDITION": "Condition / diagnosis",
    "ANATOMY": "Anatomy",
    "TEST_TREATMENT_PROCEDURE": "Test / treatment / procedure",
    "PROTECTED_HEALTH_INFORMATION": "Protected health information",
    "TIME_EXPRESSION": "Time expression",
    "BEHAVIORAL_ENVIRONMENTAL_SOCIAL": "Behavioral / social",
}


def display_group(category, entity_type, traits=None):
    """Return an explicitly application-defined display grouping."""
    if category == "MEDICATION":
        if entity_type == "DOSAGE":
            return "Dosage"
        if entity_type in {"DURATION", "FREQUENCY", "RATE"}:
            return "Medication schedule"
        return "Medication"
    if category == "MEDICAL_CONDITION":
        trait_names = {trait.get("Name") for trait in traits or []}
        if "SIGN" in trait_names or entity_type == "SIGN":
            return "Sign"
        if "SYMPTOM" in trait_names or entity_type == "SYMPTOM":
            return "Symptom"
        return "Condition / diagnosis"
    if category == "TEST_TREATMENT_PROCEDURE":
        return {
            "TEST_NAME": "Medical test",
            "TREATMENT_NAME": "Treatment",
            "PROCEDURE_NAME": "Procedure",
            "TEST_VALUE": "Test value",
            "TEST_UNIT": "Test unit",
        }.get(entity_type, "Test / treatment / procedure")
    return CATEGORY_GROUPS.get(category, "Other AWS category")


def _concepts(raw_entity):
    concepts = []
    for key in ("ICD10CMConcepts", "RxNormConcepts", "SNOMEDCTConcepts"):
        for concept in raw_entity.get(key, []) or []:
            normalized = {
                "system": key.removesuffix("Concepts"),
                "code": concept.get("Code"),
                "description": concept.get("Description"),
            }
            if "Score" in concept:
                normalized["score"] = concept["Score"]
            concepts.append(normalized)
    return concepts


def map_entity(raw_entity, source_index=0):
    category = raw_entity.get("Category")
    entity_type = raw_entity.get("Type")
    traits = raw_entity.get("Traits", []) or []
    result = {
        "source_index": source_index,
        "text": raw_entity.get("Text"),
        "category": category,
        "type": entity_type,
        "display_group": display_group(category, entity_type, traits),
        "begin_offset": raw_entity.get("BeginOffset"),
        "end_offset": raw_entity.get("EndOffset"),
        "traits": [{"name": trait.get("Name"), "score": trait.get("Score")} for trait in traits],
        "attributes": [],
        "concepts": _concepts(raw_entity),
    }
    if "Score" in raw_entity:
        result["score"] = raw_entity["Score"]
    if "Id" in raw_entity:
        result["id"] = raw_entity["Id"]
    for attribute in raw_entity.get("Attributes", []) or []:
        normalized = {
            "text": attribute.get("Text"),
            "category": attribute.get("Category"),
            "type": attribute.get("Type"),
            "id": attribute.get("Id"),
            "begin_offset": attribute.get("BeginOffset"),
            "end_offset": attribute.get("EndOffset"),
            "display_group": display_group(
                attribute.get("Category"), attribute.get("Type"), attribute.get("Traits", [])
            ),
            "parent_entity": raw_entity.get("Text"),
            "traits": [
                {"name": trait.get("Name"), "score": trait.get("Score")}
                for trait in attribute.get("Traits", []) or []
            ],
        }
        for field in ("Score", "RelationshipType", "RelationshipScore"):
            if field in attribute:
                normalized[field[0].lower() + field[1:]] = attribute[field]
        result["attributes"].append(normalized)
    return result


def flatten_entity_spans(entities):
    """Return entities and nested attributes as separate source-returned spans."""
    spans = []
    for entity in entities or []:
        spans.append(entity)
        for attribute in entity.get("attributes", []):
            child = {
                "text": attribute.get("text"),
                "category": attribute.get("category"),
                "type": attribute.get("type"),
                "display_group": attribute.get("display_group", "Other AWS category"),
                "begin_offset": attribute.get("begin_offset"),
                "end_offset": attribute.get("end_offset"),
                "traits": attribute.get("traits", []),
                "attributes": [],
                "concepts": [],
                "parent_entity": entity.get("text"),
                "relationship_type": attribute.get("relationshipType"),
            }
            if "score" in attribute:
                child["score"] = attribute["score"]
            if "relationshipScore" in attribute:
                child["relationship_score"] = attribute["relationshipScore"]
            spans.append(child)
    return spans


def map_entities_response(response):
    if not isinstance(response, dict):
        raise ValueError("AWS returned an invalid response object.")
    raw_entities = response.get("Entities", []) or []
    if not isinstance(raw_entities, list) or not all(
        isinstance(entity, dict) for entity in raw_entities
    ):
        raise ValueError("AWS returned an invalid entity list.")
    return [
        map_entity(entity, source_index=index)
        for index, entity in enumerate(raw_entities)
    ]


def map_concept_response(response, concept_field):
    results = []
    for entity in response.get("Entities", []) or []:
        results.append(
            {
                "text": entity.get("Text"),
                "category": entity.get("Category"),
                "type": entity.get("Type"),
                "begin_offset": entity.get("BeginOffset"),
                "end_offset": entity.get("EndOffset"),
                "score": entity.get("Score"),
                "concepts": [
                    {
                        "code": item.get("Code"),
                        "description": item.get("Description"),
                        "score": item.get("Score"),
                    }
                    for item in entity.get(concept_field, []) or []
                ],
            }
        )
    return {"model_version": response.get("ModelVersion"), "entities": results}
