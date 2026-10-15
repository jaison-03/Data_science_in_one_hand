"""Main Streamlit dashboard and analysis workflow."""

from pathlib import Path

import streamlit as st

from nlp.core_nlp_tasks.medical_ner.config.settings import (
    get_aws_region,
)
from nlp.core_nlp_tasks.medical_ner.core.analytics import summarize_entities
from nlp.core_nlp_tasks.medical_ner.core.validators import validate_comprehend_text
from nlp.core_nlp_tasks.medical_ner.services.aws_comprehend_medical import (
    MedicalNerServiceError,
    check_configuration,
    detect_entities,
    test_connection,
)
from nlp.core_nlp_tasks.medical_ner.ui.results import render_results
from nlp.core_nlp_tasks.medical_ner.ui.visualizations import render_statistics
from nlp.core_nlp_tasks.medical_ner.utils.error_handler import user_error_message


APP_DIR = Path(__file__).resolve().parents[1]
SAMPLE_TEXTS = {
    "Medication and condition": (
        "Synthetic example: The patient was prescribed 500 mg of amoxicillin twice daily "
        "for seven days to treat a throat infection."
    ),
    "Clinical note": (
        "Synthetic example: Patient reports fever and cough for three days. History of "
        "asthma. Continue prescribed inhaler and follow up with the clinic next week."
    ),
}


def _render_home():
    st.title("AI-Powered Medical Named Entity Recognition")
    st.caption("Clinical entity extraction using the AWS Comprehend Medical API.")
    st.markdown(
        "Medical NER identifies clinical mentions such as conditions, symptoms, medications, "
        "dosages, tests, procedures, anatomy, and time expressions. AWS Comprehend Medical "
        "returns the entities and confidence scores; this app does not make clinical decisions."
    )
    status = check_configuration()
    columns = st.columns(3)
    columns[0].metric("AWS region", status["region"])
    columns[1].metric("Credentials detected", "Yes" if status["credentials_available"] else "No")
    columns[2].metric("Client", "Available" if status["client_available"] else "Unavailable")
    if not status["credentials_available"]:
        st.info("Configure an AWS profile, environment credentials, or an IAM role to analyze text.")
    if st.button("Test AWS connection (may incur charges)"):
        with st.spinner("Sending a synthetic sample to AWS Comprehend Medical..."):
            try:
                test_connection()
            except MedicalNerServiceError as error:
                st.error(user_error_message(error))
            else:
                st.success("AWS Comprehend Medical responded to the synthetic test request.")
    st.subheader("How to use")
    st.markdown(
        "1. Open **Analyze Text** and enter a short English clinical note or synthetic sample.\n"
        "2. Confirm the external-service privacy notice before sending it.\n"
        "3. Review returned categories, types, character spans, confidence, traits, and relationships.\n"
        "4. Optionally request ICD-10-CM or RxNorm candidate concepts."
    )
    st.warning(
        "Privacy: Do not submit identifiable patient information unless you have authorization "
        "and have assessed your AWS configuration and applicable privacy obligations. "
        "AWS usage may incur charges. This application is not automatically HIPAA-compliant."
    )


def _render_analyzer():
    st.title("Analyze Medical Text")
    st.caption(
        "The primary DetectEntitiesV2 operation accepts fewer than 20,000 UTF-8 bytes. "
        "The optional concept operations accept at most 10,000 characters."
    )
    col_sample, col_clear = st.columns(2)
    if col_sample.button("Load synthetic sample", use_container_width=True):
        st.session_state["medical_ner_input"] = SAMPLE_TEXTS["Medication and condition"]
    if col_clear.button("Clear text and results", use_container_width=True):
        st.session_state["medical_ner_input"] = ""
        st.session_state.pop("medical_ner_result", None)
        st.session_state.pop("medical_ner_icd10_result", None)
        st.session_state.pop("medical_ner_rxnorm_result", None)
        st.rerun()
    sample_path = APP_DIR / "samples" / "sample_medical_notes.txt"
    if st.button("Load second synthetic sample"):
        st.session_state["medical_ner_input"] = sample_path.read_text(encoding="utf-8")
    text = st.text_area(
        "Clinical text (English)",
        key="medical_ner_input",
        height=220,
        help="Only synthetic or properly authorized text should be sent to AWS.",
    )
    byte_count = len(text.encode("utf-8"))
    st.caption(f"{len(text):,} characters · {byte_count:,} UTF-8 bytes")
    consent = st.checkbox(
        "I understand this text will be sent to AWS Comprehend Medical for processing.",
        key="medical_ner_external_consent",
    )
    if st.button("Analyze with AWS Comprehend Medical", type="primary", use_container_width=True):
        try:
            validate_comprehend_text(text)
        except ValueError as error:
            st.error(str(error))
            return
        if not consent:
            st.error("Confirm the external-service notice before sending text to AWS.")
            return
        with st.spinner("AWS Comprehend Medical is analyzing the text..."):
            try:
                result = detect_entities(text)
            except (MedicalNerServiceError, ValueError) as error:
                st.error(user_error_message(error))
                return
            except (OSError, RuntimeError, TypeError, KeyError, AttributeError):
                st.error("An unexpected error occurred. Clinical text and AWS credentials were not logged.")
                return
        st.session_state["medical_ner_result"] = result
        st.session_state["medical_ner_analyzed_text"] = text
        st.session_state.pop("medical_ner_icd10_result", None)
        st.session_state.pop("medical_ner_rxnorm_result", None)
    result = st.session_state.get("medical_ner_result")
    analyzed_text = st.session_state.get("medical_ner_analyzed_text")
    if result and analyzed_text is not None:
        st.caption(
            f"Operation: {result['operation']} · Region: {result['region']} · "
            f"Model version: {result.get('model_version') or 'not returned'}"
        )
        render_statistics(summarize_entities(result["entities"], result["duration_seconds"]))
        render_results(analyzed_text, result)


def _render_help():
    st.title("Help and About")
    with st.expander("What is Medical NER?", expanded=True):
        st.write(
            "Medical named entity recognition detects spans of clinical text and classifies "
            "them using AWS-provided categories and types. Display groups in this app are "
            "application-defined summaries, not additional AWS predictions."
        )
    with st.expander("AWS service operations"):
        st.markdown(
            "- `DetectEntitiesV2`: primary clinical entity extraction.\n"
            "- `InferICD10CM`: optional condition-to-code candidate matching.\n"
            "- `InferRxNorm`: optional medication concept candidate matching."
        )
        st.write(
            "Amazon Comprehend Medical processes English text. Returned codes and concepts are "
            "candidates only and must be reviewed by a qualified professional."
        )
    with st.expander("Privacy, medical safety, and limitations"):
        st.warning(
            "This educational tool is not for diagnosis, treatment, or medication changes. "
            "It does not make an application HIPAA-compliant. Avoid real patient information "
            "unless appropriately authorized and your AWS controls have been reviewed."
        )
        st.write(
            "Submitted text is sent to AWS. The app does not store clinical notes in a database "
            "or write them to logs. AWS service use can incur charges."
        )
    with st.expander("AWS setup and troubleshooting"):
        st.markdown(
            "1. Configure credentials with `aws configure` or a named AWS profile; never put "
            "access keys in this app's `.env` file.\n"
            "2. Set `AWS_REGION` to a region where Amazon Comprehend Medical is offered.\n"
            "3. Grant only the needed actions: `comprehendmedical:DetectEntitiesV2`, "
            "`comprehendmedical:InferICD10CM`, and `comprehendmedical:InferRxNorm`.\n"
            "4. Check AWS region availability if the service endpoint is unsupported.\n"
            "5. Missing credentials: verify the active profile or IAM role. Access denied: "
            "check IAM actions. Throttling: wait and retry. Validation errors: check language "
            "and documented input limits."
        )
    st.caption(f"Current region: {get_aws_region()}")


def run_app():
    st.set_page_config(
        page_title="Medical NER — AWS Comprehend Medical",
        page_icon="🩺",
        layout="wide",
    )
    st.sidebar.title("Medical NER")
    page = st.sidebar.radio(
        "Navigate",
        ["Home Dashboard", "Analyze Text", "Help and About"],
    )
    if page == "Home Dashboard":
        _render_home()
    elif page == "Analyze Text":
        _render_analyzer()
    else:
        _render_help()
