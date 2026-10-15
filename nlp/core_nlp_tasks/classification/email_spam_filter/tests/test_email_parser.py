"""Tests for safe message and .eml parsing."""

from io import BytesIO

from nlp.core_nlp_tasks.email_spam_filter.services.email_parser import parse_eml


def test_parser_extracts_headers_plain_body_and_html_without_fetching_resources():
    message = b"""From: sender@example.test
To: user@example.test
Subject: Suspicious link
MIME-Version: 1.0
Content-Type: multipart/alternative; boundary="mail"

--mail
Content-Type: text/plain; charset="utf-8"

Check this link.
--mail
Content-Type: text/html; charset="utf-8"

<html><img src="https://remote.example/pixel"><a href="https://bad.example">click</a></html>
--mail--
"""
    result = parse_eml(BytesIO(message))
    assert result["subject"] == "Suspicious link"
    assert result["sender"] == "sender@example.test"
    assert result["body"] == "Check this link."
    assert "https://remote.example/pixel" in result["html_body"]


def test_html_only_message_returns_plain_text_without_loading_resources():
    result = parse_eml(
        b"Subject: HTML\r\nContent-Type: text/html; charset=utf-8\r\n\r\n"
        b"<img src='https://remote.example/pixel'><p>Hello there</p>"
    )
    assert result["body"] == "Hello there"
    assert result["html_body"].startswith("<img")
