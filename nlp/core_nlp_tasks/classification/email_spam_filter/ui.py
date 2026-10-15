"""Streamlit pages for the email spam and security intelligence application."""

import json
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from nlp.core_nlp_tasks.email_spam_filter.database.db_manager import (
    clear_history,
    get_analyses,
    get_setting,
    get_summary,
    initialize_database,
    record_analysis,
    set_setting,
)
from nlp.core_nlp_tasks.email_spam_filter.ml.model import (
    CUSTOM_DATASET_PATH,
    MODEL_PATH,
    get_model,
    load_saved_metrics,
    train_and_evaluate,
)
from nlp.core_nlp_tasks.email_spam_filter.ml.preprocessing import validate_dataset
from nlp.core_nlp_tasks.email_spam_filter.services.email_analysis import analyze_email
from nlp.core_nlp_tasks.email_spam_filter.services.email_parser import parse_eml
from nlp.core_nlp_tasks.email_spam_filter.services.groq_service import (
    explain_email,
    groq_is_configured,
)
from nlp.advanced_nlp_and_applications.dialogue_systems.multilingual_question_answering_system.config import get_groq_model


APP_TITLE = "AI-Powered Email Spam Filter & Security Intelligence"
PAGES = [
    "Executive Dashboard",
    "Email Analyzer",
    "Bulk Email Scanner",
    "Model Performance",
    "Analysis History",
    "Settings",
]


def _active_dataset_path():
    if get_setting("dataset_source", "synthetic") == "custom":
        if not CUSTOM_DATASET_PATH.is_file():
            raise FileNotFoundError("Custom dataset is selected but no validated CSV is available.")
        return CUSTOM_DATASET_PATH
    return None


def _store_report(report, groq_used=False):
    return record_analysis(
        subject=report["subject"],
        sender=report["sender"],
        predicted_label=report["predicted_label"],
        risk_score=report["risk_score"],
        ml_probability=report["spam_probability"],
        heuristic_findings=report["heuristic_findings"],
        groq_used=groq_used,
    )


def render_dashboard():
    initialize_database()
    st.title(APP_TITLE)
    st.caption("Local TF-IDF machine learning, transparent phishing heuristics, and optional Groq explanations.")

    retention = int(get_setting("retention_days", "0"))
    if retention > 0:
        clear_history(older_than_days=retention)

    try:
        get_model()
        model_status = "Ready"
        model_status_detail = str(MODEL_PATH)
    except (OSError, ValueError) as error:
        model_status = "Unavailable"
        model_status_detail = str(error)

    summary = get_summary()
    columns = st.columns(4)
    columns[0].metric("Total analyzed", summary["total"])
    columns[1].metric("Spam", summary["spam"])
    columns[2].metric("Ham", summary["ham"])
    columns[3].metric("High risk", summary["high_risk"])

    status_columns = st.columns(2)
    with status_columns[0]:
        st.success(f"ML model: {model_status}") if model_status == "Ready" else st.error(
            f"ML model: {model_status} — {model_status_detail}"
        )
    with status_columns[1]:
        st.success(f"Groq API: configured ({get_groq_model()})") if groq_is_configured() else st.info(
            f"Groq API: optional and not configured ({get_groq_model()})"
        )

    st.subheader("Recent analysis activity")
    recent = get_analyses(limit=10)
    if recent:
        st.dataframe(
            [
                {
                    "Time (UTC)": row["timestamp"],
                    "Subject": row["subject"],
                    "Sender": row["sender"],
                    "Prediction": row["predicted_label"].upper(),
                    "Risk": f'{row["risk_score"]:.0f}%',
                    "ML spam probability": f'{row["ml_probability"]:.1%}',
                }
                for row in recent
            ],
            use_container_width=True,
            hide_index=True,
        )
    else:
        st.info("No messages have been analyzed yet. Open Email Analyzer to begin.")


def render_email_analyzer():
    initialize_database()
    st.title("Single Email Analyzer")
    st.write("Paste an email or safely extract its headers and body from an `.eml` file.")
    upload = st.file_uploader("Import .eml file", type=["eml"], key="analyzer-eml")
    parsed = {"subject": "", "sender": "", "body": "", "html_body": ""}
    if upload is not None:
        try:
            parsed = parse_eml(upload)
            st.success("Email file parsed. Remote images and other external resources were not loaded.")
        except (OSError, TypeError, ValueError) as error:
            st.error(f"Could not parse this email: {error}")
            return

    with st.form("email-analysis-form"):
        subject = st.text_input("Subject", value=parsed["subject"][:500], max_chars=500)
        sender = st.text_input("Sender", value=parsed["sender"][:500], max_chars=500)
        body = st.text_area("Email body", value=parsed["body"], height=240, max_chars=100000)
        threshold = st.slider(
            "Classify as spam at or above this ML probability",
            min_value=0.05,
            max_value=0.95,
            value=float(get_setting("spam_threshold", "0.5")),
            step=0.05,
        )
        use_groq = st.checkbox("Include an optional Groq security explanation", value=False)
        submit = st.form_submit_button("Analyze email", type="primary")

    if submit:
        if not body.strip() and not subject.strip():
            st.error("Enter a subject or email body before analysis.")
            return
        try:
            report = analyze_email(
                subject=subject,
                sender=sender,
                body=body,
                html_body=parsed["html_body"],
                threshold=threshold,
            )
        except (OSError, ValueError) as error:
            st.error(f"Local analysis failed: {error}")
            return
        explanation_result = None
        if use_groq:
            explanation_result = explain_email(
                subject,
                body,
                report["heuristic_findings"],
                report["predicted_label"],
                report["spam_probability"],
            )
            if explanation_result["error"]:
                st.warning(explanation_result["error"])
        report["groq_explanation"] = (
            explanation_result["explanation"] if explanation_result and explanation_result["used"] else None
        )
        _store_report(report, groq_used=bool(explanation_result and explanation_result["used"]))
        st.session_state["email_spam_filter_latest_report"] = report
        st.rerun()

    report = st.session_state.get("email_spam_filter_latest_report")
    if not report:
        return
    st.subheader("Analysis result")
    result_columns = st.columns(4)
    result_columns[0].metric("Prediction", report["predicted_label"].upper())
    result_columns[1].metric("ML spam probability", f'{report["spam_probability"]:.1%}')
    result_columns[2].metric("Overall security risk", f'{report["risk_score"]:.0f}%')
    result_columns[3].metric("Risk level", report["risk_level"])
    st.progress(min(1.0, max(0.0, report["spam_probability"])))
    if report["heuristic_findings"]:
        st.warning("Security signals were detected. Verify senders and destinations independently.")
        st.dataframe(
            [
                {
                    "Severity": item["severity"].upper(),
                    "Check": item["category"].replace("_", " ").title(),
                    "Finding": item["message"],
                    "Evidence": item["evidence"],
                }
                for item in report["heuristic_findings"]
            ],
            use_container_width=True,
            hide_index=True,
        )
    else:
        st.success("No configured heuristic warning matched this message.")
    if report["groq_explanation"]:
        st.subheader("Groq security explanation")
        st.write(report["groq_explanation"]["summary"])
        st.write("Phishing flags:", report["groq_explanation"]["phishing_flags"])
        st.write("Urgency detected:", report["groq_explanation"]["urgency_detected"])
        st.write("Recommended actions:", report["groq_explanation"]["recommended_actions"])

    json_report = json.dumps(report, indent=2, ensure_ascii=True)
    csv_report = pd.DataFrame(
        [
            {
                "subject": report["subject"],
                "sender": report["sender"],
                "predicted_label": report["predicted_label"],
                "spam_probability": report["spam_probability"],
                "risk_score": report["risk_score"],
                "risk_level": report["risk_level"],
                "heuristic_findings": json.dumps(report["heuristic_findings"]),
                "groq_explanation": json.dumps(report["groq_explanation"]),
            }
        ]
    ).to_csv(index=False)
    download_columns = st.columns(2)
    download_columns[0].download_button(
        "Download JSON report",
        data=json_report,
        file_name="email_security_report.json",
        mime="application/json",
    )
    download_columns[1].download_button(
        "Download CSV report",
        data=csv_report,
        file_name="email_security_report.csv",
        mime="text/csv",
    )


def render_bulk_scanner():
    initialize_database()
    st.title("Bulk Email Scanner")
    st.write("Upload a CSV with a `body`, `text`, `message`, or `email` column. Subject and sender are optional.")
    upload = st.file_uploader("Email dataset (CSV)", type=["csv"], key="bulk-email-csv")
    if upload is None:
        return
    try:
        frame = pd.read_csv(upload)
    except (pd.errors.ParserError, UnicodeDecodeError, ValueError) as error:
        st.error(f"Could not read the uploaded CSV: {error}")
        return
    if frame.empty:
        st.error("The uploaded CSV contains no email rows.")
        return
    if len(frame) > 10000:
        st.error("The scanner accepts at most 10,000 rows per upload.")
        return
    columns = {str(column).strip().lower(): column for column in frame.columns}
    body_column = next((columns[name] for name in ("body", "text", "message", "email") if name in columns), None)
    if body_column is None:
        st.error("CSV needs one of these columns: body, text, message, or email.")
        return
    st.caption(f"Loaded {len(frame):,} rows.")
    if not st.button("Scan all emails", type="primary", key="run-bulk-scan"):
        st.dataframe(frame.head(20), use_container_width=True)
        return

    subject_column = columns.get("subject")
    sender_column = columns.get("sender") or columns.get("from")
    threshold = float(get_setting("spam_threshold", "0.5"))
    reports = []
    try:
        model = get_model()
        for row in frame.to_dict(orient="records"):
            subject_value = row.get(subject_column) if subject_column else ""
            sender_value = row.get(sender_column) if sender_column else ""
            body_value = row.get(body_column, "")
            report = analyze_email(
                subject="" if pd.isna(subject_value) else str(subject_value),
                sender="" if pd.isna(sender_value) else str(sender_value),
                body="" if pd.isna(body_value) else str(body_value),
                threshold=threshold,
                model=model,
            )
            reports.append(report)
            _store_report(report)
    except (OSError, ValueError) as error:
        st.error(f"Bulk analysis failed: {error}")
        return

    results = pd.DataFrame(
        [
            {
                "subject": report["subject"],
                "sender": report["sender"],
                "predicted_label": report["predicted_label"],
                "spam_probability": round(report["spam_probability"], 4),
                "risk_score": report["risk_score"],
                "risk_level": report["risk_level"],
                "heuristic_findings": json.dumps(report["heuristic_findings"]),
            }
            for report in reports
        ]
    )
    st.session_state["email_spam_filter_bulk_report"] = results
    st.success(f"Scanned {len(results):,} emails.")
    st.dataframe(results, use_container_width=True, hide_index=True)
    st.download_button(
        "Download bulk risk report",
        data=results.to_csv(index=False),
        file_name="bulk_email_risk_report.csv",
        mime="text/csv",
    )


def render_model_performance():
    initialize_database()
    st.title("Model Performance & Training")
    try:
        dataset_path = _active_dataset_path()
    except FileNotFoundError as error:
        st.error(str(error))
        return
    dataset_name = "Built-in synthetic demonstration dataset" if dataset_path is None else str(dataset_path)
    st.write(f"Training dataset: **{dataset_name}**")
    metrics = load_saved_metrics()
    if st.button("Train and compare models", type="primary"):
        try:
            _, metrics = train_and_evaluate(dataset_path=dataset_path)
            st.success(f'Best model trained and saved: {metrics["best_model"]}')
        except (OSError, ValueError, pd.errors.ParserError) as error:
            st.error(f"Model training failed: {error}")
            return
    if metrics is None:
        try:
            get_model()
            metrics = load_saved_metrics()
        except (OSError, ValueError) as error:
            st.error(f"Model is not available: {error}")
            return
    if metrics is None:
        st.error("Training completed without evaluation metrics.")
        return

    st.subheader(f'Best model: {metrics["best_model"]}')
    st.caption(
        f'{metrics["training_samples"]} training examples; '
        f'{metrics["best_model_metrics"]["test_samples"]} held-out test examples.'
    )
    best = metrics["best_model_metrics"]
    metric_columns = st.columns(5)
    for column, (label, key) in zip(
        metric_columns,
        [
            ("Accuracy", "accuracy"),
            ("Precision", "precision"),
            ("Recall", "recall"),
            ("F1-score", "f1_score"),
            ("ROC-AUC", "roc_auc"),
        ],
    ):
        column.metric(label, f'{best[key]:.1%}')

    matrix = best["confusion_matrix"]
    labels = best["confusion_labels"]
    confusion_figure = go.Figure(
        data=go.Heatmap(
            z=matrix,
            x=[f"Predicted {label}" for label in labels],
            y=[f"Actual {label}" for label in labels],
            text=matrix,
            texttemplate="%{text}",
            colorscale="Blues",
            showscale=True,
        )
    )
    confusion_figure.update_layout(title="Confusion matrix", height=400)
    roc = best["roc_curve"]
    roc_figure = go.Figure()
    roc_figure.add_trace(
        go.Scatter(
            x=roc["false_positive_rate"],
            y=roc["true_positive_rate"],
            mode="lines",
            name=f'Model ROC (AUC {best["roc_auc"]:.3f})',
        )
    )
    roc_figure.add_trace(
        go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="Random baseline", line={"dash": "dash"})
    )
    roc_figure.update_layout(
        title="Receiver operating characteristic",
        xaxis_title="False positive rate",
        yaxis_title="True positive rate",
        height=400,
    )
    chart_columns = st.columns(2)
    chart_columns[0].plotly_chart(confusion_figure, use_container_width=True)
    chart_columns[1].plotly_chart(roc_figure, use_container_width=True)

    comparisons = metrics["model_comparison"]
    comparison = pd.DataFrame(
        [
            {"Model": name, **{key: values[key] for key in ("accuracy", "precision", "recall", "f1_score", "roc_auc")}}
            for name, values in comparisons.items()
        ]
    ).set_index("Model")
    st.subheader("Classifier comparison")
    st.plotly_chart(
        go.Figure(
            [
                go.Bar(name=metric, x=comparison.index, y=comparison[metric], texttemplate="%{y:.1%}")
                for metric in ("accuracy", "precision", "recall", "f1_score", "roc_auc")
            ]
        ).update_layout(barmode="group", yaxis={"range": [0, 1]}, height=420),
        use_container_width=True,
    )
    st.dataframe(comparison.style.format("{:.1%}"), use_container_width=True)


def render_analysis_history():
    initialize_database()
    st.title("Analysis History")
    filters = st.columns([2, 1, 1])
    search = filters[0].text_input("Search subject or sender")
    label = filters[1].selectbox("Prediction", ["All", "Spam", "Ham"])
    min_risk = filters[2].slider("Minimum risk", 0, 100, 0)
    rows = get_analyses(
        search=search,
        label=label.lower() if label != "All" else None,
        min_risk=min_risk,
    )
    st.caption(f"{len(rows)} matching records (up to 5,000).")
    if rows:
        display = pd.DataFrame(
            [
                {
                    "ID": row["id"],
                    "Timestamp (UTC)": row["timestamp"],
                    "Subject": row["subject"],
                    "Sender": row["sender"],
                    "Prediction": row["predicted_label"].upper(),
                    "Risk": row["risk_score"],
                    "ML probability": row["ml_probability"],
                    "Heuristic findings": len(row["heuristic_findings"]),
                    "Groq used": row["groq_used"],
                }
                for row in rows
            ]
        )
        st.dataframe(display, use_container_width=True, hide_index=True)
        st.download_button(
            "Download filtered history CSV",
            data=display.to_csv(index=False),
            file_name="email_analysis_history.csv",
            mime="text/csv",
        )
    else:
        st.info("No history entries match these filters.")
    st.divider()
    st.subheader("Clear analysis history")
    confirm = st.checkbox("I understand that clearing history permanently deletes the selected records.")
    if st.button("Clear all history", disabled=not confirm):
        deleted = clear_history()
        st.success(f"Deleted {deleted} analysis records.")
        st.rerun()


def render_settings():
    initialize_database()
    st.title("Settings")
    st.subheader("Classification threshold")
    current_threshold = float(get_setting("spam_threshold", "0.5"))
    threshold = st.slider("Default spam probability threshold", 0.05, 0.95, current_threshold, 0.05)
    if st.button("Save threshold"):
        set_setting("spam_threshold", threshold)
        st.success(f"Default spam threshold saved at {threshold:.0%}.")

    st.subheader("Analysis data retention")
    current_retention = int(get_setting("retention_days", "0"))
    retention_choices = [0, 30, 90, 180, 365]
    selected_index = retention_choices.index(current_retention) if current_retention in retention_choices else 0
    retention = st.selectbox(
        "Delete records older than (0 keeps history indefinitely)",
        retention_choices,
        index=selected_index,
        format_func=lambda value: "Never" if value == 0 else f"{value} days",
    )
    if st.button("Save retention policy"):
        set_setting("retention_days", retention)
        st.success("Retention policy saved.")

    st.subheader("Training dataset")
    source = st.radio(
        "Select dataset source",
        ["synthetic", "custom"],
        index=0 if get_setting("dataset_source", "synthetic") == "synthetic" else 1,
        format_func=lambda value: "Built-in synthetic demonstration data" if value == "synthetic" else "Uploaded custom CSV",
        horizontal=True,
    )
    custom_upload = st.file_uploader("Custom labeled training CSV", type=["csv"], key="training-csv")
    if custom_upload is not None and st.button("Validate and save custom dataset"):
        try:
            custom = pd.read_csv(custom_upload)
            validated = validate_dataset(custom)
            CUSTOM_DATASET_PATH.parent.mkdir(parents=True, exist_ok=True)
            validated.to_csv(CUSTOM_DATASET_PATH, index=False)
            st.success(
                f"Validated and saved {len(validated)} labeled messages. "
                "Select the custom source and retrain the model to use it."
            )
        except (OSError, pd.errors.ParserError, ValueError, UnicodeDecodeError) as error:
            st.error(f"Custom dataset was not saved: {error}")
    if st.button("Save dataset selection"):
        if source == "custom" and not CUSTOM_DATASET_PATH.is_file():
            st.error("Upload and validate a custom dataset before selecting it.")
        else:
            set_setting("dataset_source", source)
            st.success("Dataset selection saved. Retrain the model on the Model Performance page to apply it.")

    st.subheader("Optional Groq API")
    if groq_is_configured():
        st.success("GROQ_API_KEY is configured. Its value is never shown in this panel.")
        if st.button("Test Groq connection"):
            result = explain_email(
                "Connection test",
                "This is a harmless configuration test, not an email analysis.",
                [],
                "ham",
                0.01,
            )
            if result["used"]:
                st.success("Groq connection succeeded.")
                st.json(result["explanation"])
            else:
                st.error(result["error"] or "Groq connection failed.")
    else:
        st.info("Groq is optional. Set GROQ_API_KEY in the environment or a local .env file to enable it.")


def render_app(embed=False):
    """Render the complete application within the host project or on its own."""
    initialize_database()
    if embed:
        st.subheader(APP_TITLE)
        default_page = "Executive Dashboard"
    else:
        st.set_page_config(page_title="Email Spam Filter & Security Intelligence", page_icon="🛡️", layout="wide")
        default_page = "Executive Dashboard"
    selected_page = st.sidebar.radio(
        "Email Security",
        PAGES,
        index=PAGES.index(default_page),
        key="email-spam-filter-navigation",
    )
    render_page(selected_page)


def render_page(page):
    """Render a named email security page for both embedded and standalone use."""
    page_renderers = {
        "Executive Dashboard": render_dashboard,
        "Email Analyzer": render_email_analyzer,
        "Bulk Email Scanner": render_bulk_scanner,
        "Model Performance": render_model_performance,
        "Analysis History": render_analysis_history,
        "Settings": render_settings,
    }
    renderer = page_renderers.get(page)
    if renderer is None:
        raise ValueError(f"Unknown email security page: {page}")
    renderer()
