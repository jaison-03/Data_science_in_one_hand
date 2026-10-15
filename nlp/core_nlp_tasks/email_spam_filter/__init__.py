"""Backward-compatible import path for the relocated email spam filter."""

from pathlib import Path

__path__ = [
    str(Path(__file__).resolve().parents[1] / "classification" / "email_spam_filter")
]
