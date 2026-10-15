"""Filtered entity results, highlighting, relationships, concepts, and exports."""

import html

import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

from nlp.core_nlp_tasks.medical_ner.core.text_highlighter import (
    COLORS,
    highlight_entities,
)
from nlp.core_nlp_tasks.medical_ner.core.entity_mapper import flatten_entity_spans
from nlp.core_nlp_tasks.medical_ner.services.aws_comprehend_medical import (
    MedicalNerServiceError,
    infer_icd10cm,
    infer_rxnorm,
)
from nlp.core_nlp_tasks.medical_ner.utils.error_handler import user_error_message
from nlp.core_nlp_tasks.medical_ner.utils.export_utils import (
    entities_csv,
    result_json,
    result_summary_text,
)


def _entity_frame(entities):
    rows = []
    for entity in entities:
        rows.append(
            {
                "Entity text": entity.get("text"),
                "AWS category": entity.get("category"),
                "AWS type": entity.get("type"),
                "Display group (app-defined)": entity.get("display_group"),
                "Confidence": entity.get("score"),
                "Offsets": f"{entity.get('begin_offset')}–{entity.get('end_offset')}",
                "Traits returned": ", ".join(
                    trait["name"] for trait in entity.get("traits", []) if trait.get("name")
                ),
                "Concept identifiers returned": "; ".join(
                    f"{concept.get('system')}:{concept.get('code')}"
                    for concept in entity.get("concepts", [])
                    if concept.get("code")
                ),
                "Parent entity": entity.get("parent_entity"),
                "AWS relationship type": entity.get("relationship_type"),
                "_category": entity.get("category"),
            }
        )
    return pd.DataFrame(rows)


def render_relationships(entities, unmapped_attributes=None):
    rows = []
    for entity in entities:
        for attribute in entity.get("attributes", []):
            relation = attribute.get("relationshipType")
            if relation:
                rows.append(
                    {
                        "Parent entity": entity.get("text"),
                        "Returned attribute": attribute.get("text"),
                        "AWS relationship type": relation,
                        "Relationship score": attribute.get("relationshipScore"),
                    }
                )
    if rows:
        st.subheader("Relationships returned by AWS")
        st.dataframe(rows, hide_index=True, use_container_width=True)
    else:
        st.caption("The service did not return linked entity-attribute relationships.")
    if unmapped_attributes:
        st.markdown("**Unmapped attributes returned by AWS**")
        st.dataframe(unmapped_attributes, hide_index=True, use_container_width=True)


def _render_concepts(text):
    st.subheader("Optional concept candidates")
    st.warning(
        "ICD-10-CM and RxNorm results are AWS candidates only—not confirmed diagnoses, "
        "prescriptions, or billing decisions. These operations may incur additional charges."
    )
    left, right = st.columns(2)
    if left.button("Look up ICD-10-CM candidates"):
        with st.spinner("Requesting ICD-10-CM candidates..."):
            try:
                result = infer_icd10cm(text)
            except (MedicalNerServiceError, ValueError) as error:
                st.error(user_error_message(error))
            else:
                st.session_state["medical_ner_icd10_result"] = result
    if right.button("Look up RxNorm medication candidates"):
        with st.spinner("Requesting RxNorm candidates..."):
            try:
                result = infer_rxnorm(text)
            except (MedicalNerServiceError, ValueError) as error:
                st.error(user_error_message(error))
            else:
                st.session_state["medical_ner_rxnorm_result"] = result
    for label, key in (
        ("ICD-10-CM candidates", "medical_ner_icd10_result"),
        ("RxNorm candidates", "medical_ner_rxnorm_result"),
    ):
        result = st.session_state.get(key)
        if result is not None:
            st.markdown(f"**{label}**")
            rows = [
                {
                    "Text": entity.get("text"),
                    "AWS category": entity.get("category"),
                    "AWS type": entity.get("type"),
                    **concept,
                }
                for entity in result.get("entities", [])
                for concept in entity.get("concepts", [])
            ]
            st.dataframe(rows, hide_index=True, use_container_width=True) if rows else st.info(
                "AWS returned no concept candidates."
            )


def render_results(text, result):
    entities = result.get("entities", [])
    spans = flatten_entity_spans(entities)
    st.subheader("Entity extraction results")
    if not entities:
        st.info("AWS returned no entities for this text.")
    else:
        frame = _entity_frame(spans)
        categories = sorted(value for value in frame["_category"].dropna().unique())
        selected_categories = st.multiselect(
            "Filter by AWS category",
            categories,
            default=categories,
        )
        min_confidence = st.slider("Minimum confidence", 0.0, 1.0, 0.0, 0.05)
        filtered = [
            entity
            for entity in spans
            if entity.get("category") in selected_categories
            and (
                entity.get("score") is None
                or entity.get("score", 0) >= min_confidence
            )
        ]
        filtered_frame = _entity_frame(filtered)
        if not filtered_frame.empty:
            st.dataframe(
                filtered_frame.drop(columns=["_category"]),
                hide_index=True,
                use_container_width=True,
            )
        else:
            st.info("No entities match the selected filters.")

    st.subheader("Highlighted source text")
    st.caption("Entity colors use application-defined display groups; category/type labels are from AWS.")
    legend_items = [
        f'<span style="background:{color};padding:3px 7px;border-radius:4px">'
        f"{html.escape(group)}</span>"
        for group, color in COLORS.items()
    ]
    st.markdown(" ".join(legend_items), unsafe_allow_html=True)
    components.html(
        f'<div style="white-space:pre-wrap;font:15px/1.7 sans-serif">'
        f"{highlight_entities(text, spans)}</div>",
        height=max(120, min(600, text.count("\n") * 24 + 140)),
        scrolling=True,
    )

    render_relationships(entities, result.get("unmapped_attributes"))
    _render_concepts(text)
    st.subheader("Export")
    safe_stem = "medical_ner_results"
    col1, col2, col3 = st.columns(3)
    col1.download_button(
        "Download CSV entities",
        entities_csv(spans),
        file_name=f"{safe_stem}.csv",
        mime="text/csv",
        use_container_width=True,
    )
    col2.download_button(
        "Download JSON API report",
        result_json(result),
        file_name=f"{safe_stem}.json",
        mime="application/json",
        use_container_width=True,
    )
    col3.download_button(
        "Download TXT summary",
        result_summary_text(result),
        file_name=f"{safe_stem}.txt",
        mime="text/plain",
        use_container_width=True,
    )
