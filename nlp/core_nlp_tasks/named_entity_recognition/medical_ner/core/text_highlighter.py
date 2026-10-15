"""Safe offset-based HTML highlighting, including overlapping spans."""

from html import escape


COLORS = {
    "Medication": "#d7f3e3",
    "Dosage": "#fff0bf",
    "Medication schedule": "#ffdfbf",
    "Condition / diagnosis": "#f7d9df",
    "Symptom": "#ffe3e8",
    "Sign": "#fbe2d3",
    "Anatomy": "#dce8ff",
    "Test / treatment / procedure": "#e8ddff",
    "Medical test": "#e8ddff",
    "Treatment": "#eee2ff",
    "Procedure": "#ddd0ff",
    "Test value": "#e2ebff",
    "Test unit": "#d7e8ff",
    "Protected health information": "#e6e6e6",
    "Time expression": "#fff3d5",
    "Behavioral / social": "#e5f0d6",
    "Other AWS category": "#eeeeee",
}


def highlight_entities(text, entities):
    """Render escaped source text split at every span boundary."""
    if not text:
        return ""
    valid = []
    for entity in entities or []:
        start, end = entity.get("begin_offset"), entity.get("end_offset")
        if isinstance(start, int) and isinstance(end, int) and 0 <= start < end <= len(text):
            valid.append(entity)
    boundaries = {0, len(text)}
    for entity in valid:
        boundaries.update((entity["begin_offset"], entity["end_offset"]))
    positions = sorted(boundaries)
    output = []
    for start, end in zip(positions, positions[1:]):
        segment = escape(text[start:end])
        active = [
            entity
            for entity in valid
            if entity["begin_offset"] < end and entity["end_offset"] > start
        ]
        if not active:
            output.append(segment)
            continue
        labels = ", ".join(
            f"{entity['text']} ({entity['category']} / {entity['type']})"
            for entity in active
        )
        color = COLORS.get(active[0]["display_group"], "#eeeeee")
        output.append(
            f'<mark style="background-color:{color};padding:1px 3px;border-radius:3px" '
            f'title="{escape(labels, quote=True)}">{segment}</mark>'
        )
    return "".join(output)


def highlight_legend():
    return dict(COLORS)
