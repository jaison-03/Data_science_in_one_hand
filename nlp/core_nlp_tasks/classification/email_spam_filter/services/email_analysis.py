"""Shared orchestration for machine-learning and heuristic email analysis."""

from nlp.core_nlp_tasks.email_spam_filter.ml.model import predict_email
from nlp.core_nlp_tasks.email_spam_filter.services.security_service import inspect_email


def analyze_email(subject, body, sender="", html_body="", threshold=0.5, model=None):
    """Combine local model and transparent security signals into a risk report."""
    content = f"{subject}\n{body}".strip()
    prediction = predict_email(content, threshold=threshold, model=model)
    security = inspect_email(body, html_body=html_body)
    risk_score = round(
        min(100.0, 0.55 * prediction["spam_probability"] * 100 + 0.45 * security["risk_score"]),
        2,
    )
    return {
        "subject": str(subject),
        "sender": str(sender),
        "body": str(body),
        "predicted_label": prediction["label"],
        "spam_probability": prediction["spam_probability"],
        "ham_probability": prediction["ham_probability"],
        "threshold": prediction["threshold"],
        "heuristic_findings": security["findings"],
        "heuristic_risk_score": security["risk_score"],
        "risk_score": risk_score,
        "risk_level": "High" if risk_score >= 70 else "Medium" if risk_score >= 40 else "Low",
    }
