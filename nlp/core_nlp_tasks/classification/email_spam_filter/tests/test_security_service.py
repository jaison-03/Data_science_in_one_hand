"""Tests for URL and social-engineering checks."""

from nlp.core_nlp_tasks.email_spam_filter.services.security_service import inspect_email


def test_flags_ip_link_and_shortener():
    result = inspect_email("Please visit http://192.0.2.55/login and https://bit.ly/example")
    categories = {finding["category"] for finding in result["findings"]}
    assert "ip_based_link" in categories
    assert "url_shortener" in categories
    assert result["risk_score"] > 0


def test_flags_mismatched_anchor_host():
    result = inspect_email(
        "Sign in",
        '<a href="https://attacker.example/login">https://trusted.example/account</a>',
    )
    assert "mismatched_anchor_text" in {item["category"] for item in result["findings"]}


def test_flags_urgent_payment_credentials_and_fake_warning():
    result = inspect_email(
        "URGENT: pay today or your account will be suspended. "
        "Verify your password and send gift cards immediately."
    )
    categories = {finding["category"] for finding in result["findings"]}
    assert {
        "urgent_language",
        "urgent_payment_demand",
        "credential_request",
        "fake_account_warning",
    } <= categories
    assert result["risk_score"] <= 100


def test_safe_message_has_no_security_findings():
    result = inspect_email("Thanks for the notes. See you at lunch tomorrow.")
    assert result == {"findings": [], "risk_score": 0}
