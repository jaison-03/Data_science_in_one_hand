"""Deterministic URL and social-engineering checks for email content."""

import ipaddress
import re
from urllib.parse import urlparse

from bs4 import BeautifulSoup


URL_PATTERN = re.compile(r"(?i)\b(?:https?://|www\.)[^\s<>'\"]+")
SHORTENER_DOMAINS = {
    "bit.ly",
    "t.co",
    "tinyurl.com",
    "ow.ly",
    "is.gd",
    "buff.ly",
    "rebrand.ly",
    "cutt.ly",
    "shorturl.at",
}
URL_REPLACEMENT = re.compile(r"(?i)\b(?:https?://|www\.)[^\s<>'\"]+")


def _host(url):
    candidate = url if "://" in url else f"http://{url}"
    parsed = urlparse(candidate)
    return (parsed.hostname or "").lower().rstrip(".")


def _is_ip_host(host):
    try:
        ipaddress.ip_address(host)
        return True
    except ValueError:
        return False


def _append(finding_list, category, severity, evidence, message, points):
    finding_list.append(
        {
            "category": category,
            "severity": severity,
            "evidence": evidence[:240],
            "message": message,
            "risk_points": points,
        }
    )


def inspect_email(body, html_body=""):
    """Return transparent findings and a capped heuristic risk score."""
    findings = []
    urls = list(dict.fromkeys(URL_PATTERN.findall(f"{body}\n{html_body}")))
    for raw_url in urls:
        cleaned_url = raw_url.rstrip(".,);!?]")
        host = _host(cleaned_url)
        if not host:
            continue
        if _is_ip_host(host):
            _append(
                findings,
                "ip_based_link",
                "high",
                cleaned_url,
                "Link points directly to an IP address instead of a named domain.",
                28,
            )
        if host in SHORTENER_DOMAINS:
            _append(
                findings,
                "url_shortener",
                "medium",
                cleaned_url,
                "Link uses a URL-shortening service that hides its destination.",
                16,
            )

    for expression, category, severity, message, points in (
        (
            r"\b(?:urgent|immediately|act now|final warning|within \d+\s*(?:minutes?|hours?|days?))\b",
            "urgent_language",
            "medium",
            "Message uses urgent or pressure-based language.",
            12,
        ),
        (
            r"\b(?:wire|transfer|send|pay|payment|gift cards?|crypto(?:currency)?)\b.{0,70}"
            r"\b(?:immediately|urgent|today|now|confirm|verify|account|fee|funds?)\b"
            r"|\b(?:urgent|immediate|final warning)\b.{0,70}\b(?:payment|pay|wire|transfer|gift cards?)\b",
            "urgent_payment_demand",
            "high",
            "Message pressures the recipient to make or arrange a payment.",
            26,
        ),
        (
            r"\b(?:password|passcode|login credentials?|bank details|credit card|card number|"
            r"social security number|one[- ]time code|verification code)\b",
            "credential_request",
            "high",
            "Message requests credentials, financial details, or a verification code.",
            28,
        ),
        (
            r"\b(?:account (?:will be|has been|is being) (?:suspended|closed|deleted|locked)|"
            r"unusual activity|security alert|verify your account|restore access)\b",
            "fake_account_warning",
            "medium",
            "Message contains an account or security warning commonly used to create panic.",
            18,
        ),
    ):
        match = re.search(expression, body, flags=re.IGNORECASE | re.DOTALL)
        if match:
            _append(findings, category, severity, match.group(0), message, points)

    if html_body:
        soup = BeautifulSoup(html_body, "html.parser")
        for anchor in soup.find_all("a", href=True):
            visible_text = " ".join(anchor.get_text(" ", strip=True).split())
            href = str(anchor.get("href", "")).strip()
            visible_urls = URL_PATTERN.findall(visible_text)
            if not href or not visible_urls:
                continue
            shown_host = _host(visible_urls[0])
            destination_host = _host(href)
            if shown_host and destination_host and shown_host != destination_host:
                _append(
                    findings,
                    "mismatched_anchor_text",
                    "high",
                    f"Visible host: {shown_host}; destination host: {destination_host}",
                    "The link text advertises a different host than the actual destination.",
                    30,
                )

    unique_findings = []
    seen = set()
    for finding in findings:
        key = (finding["category"], finding["evidence"])
        if key not in seen:
            seen.add(key)
            unique_findings.append(finding)
    risk_score = min(100, sum(finding["risk_points"] for finding in unique_findings))
    return {"findings": unique_findings, "risk_score": risk_score}
