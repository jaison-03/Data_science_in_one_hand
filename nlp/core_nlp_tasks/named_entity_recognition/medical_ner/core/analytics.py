"""Statistics calculated exclusively from returned API entities."""

from collections import Counter


def summarize_entities(entities, duration_seconds=None):
    entities = list(entities or [])
    attributes = [attribute for entity in entities for attribute in entity.get("attributes", [])]
    spans = entities + attributes
    categories = Counter(
        item.get("category") or item.get("Category")
        for item in spans
        if item.get("category") or item.get("Category")
    )
    types = Counter(
        item.get("type") or item.get("Type")
        for item in spans
        if item.get("type") or item.get("Type")
    )
    scores = [
        item.get("score", item.get("Score"))
        for item in spans
        if isinstance(item.get("score", item.get("Score")), (int, float))
    ]
    return {
        "total_entities": len(entities),
        "total_attributes": len(attributes),
        "total_spans": len(spans),
        "categories": dict(categories),
        "unique_categories": len(categories),
        "types": dict(types),
        "average_confidence": sum(scores) / len(scores) if scores else None,
        "duration_seconds": duration_seconds,
    }
