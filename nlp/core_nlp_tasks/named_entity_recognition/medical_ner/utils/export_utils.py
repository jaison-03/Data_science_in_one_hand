"""CSV, JSON, and plain-text exports for returned extraction data."""

import csv
import io
import json


CSV_FIELDS = (
    "text",
    "category",
    "type",
    "display_group",
    "score",
    "begin_offset",
    "end_offset",
    "traits",
    "attributes",
    "concepts",
)


def entities_csv(entities):
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=CSV_FIELDS, extrasaction="ignore")
    writer.writeheader()
    for entity in entities or []:
        row = dict(entity)
        for field in ("traits", "attributes", "concepts"):
            row[field] = json.dumps(row.get(field, []), ensure_ascii=False)
        writer.writerow(row)
    return buffer.getvalue()


def result_json(result):
    exportable = {
        key: value
        for key, value in result.items()
        if key not in {"raw_response"} and key not in {"entities"}
    }
    exportable["entities"] = result.get("entities", [])
    exportable["raw_response"] = result.get("raw_response", {})
    return json.dumps(exportable, ensure_ascii=False, indent=2, default=str)


def result_summary_text(result):
    entities = result.get("entities", [])
    lines = [
        "Medical Named Entity Recognition — extraction summary",
        f"Operation: {result.get('operation', 'DetectEntitiesV2')}",
        f"Region: {result.get('region', 'Not provided')}",
        f"Model version: {result.get('model_version') or 'Not provided'}",
        f"Entities returned: {len(entities)}",
        "",
        "Entities:",
    ]
    for entity in entities:
        score = entity.get("score")
        confidence = f"{score:.4f}" if isinstance(score, (int, float)) else "not returned"
        lines.append(
            f"- {entity.get('text', '')} | AWS category: {entity.get('category')} | "
            f"type: {entity.get('type')} | score: {confidence} | "
            f"offsets: {entity.get('begin_offset')}:{entity.get('end_offset')}"
        )
    return "\n".join(lines) + "\n"
