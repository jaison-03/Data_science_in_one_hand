"""Shared Groq and application configuration."""

import os
from pathlib import Path

from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_GROQ_MODEL = "openai/gpt-oss-20b"
PLACEHOLDER_KEYS = {
    "",
    "your_actual_groq_api_key_here",
    "your_groq_api_key_here",
    "your_api_key_here",
}


def load_project_environment():
    """Load the project-root .env without overriding process-level settings."""
    load_dotenv(PROJECT_ROOT / ".env", override=False)


def get_groq_api_key():
    """Return a configured key, treating sample placeholders as missing."""
    load_project_environment()
    value = os.getenv("GROQ_API_KEY", "").strip()
    if value.lower() in PLACEHOLDER_KEYS:
        return ""
    return value


def get_groq_model():
    """Return the configured Groq model or the supported project default."""
    load_project_environment()
    return os.getenv("GROQ_MODEL", "").strip() or DEFAULT_GROQ_MODEL


def get_database_path():
    """Resolve DATABASE_PATH relative to the project root unless absolute."""
    load_project_environment()
    value = os.getenv("DATABASE_PATH", "database/multilingual_qa.db").strip()
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path
