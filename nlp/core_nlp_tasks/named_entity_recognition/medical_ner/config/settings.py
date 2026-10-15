"""Application configuration and safe AWS credential discovery."""

import os
from pathlib import Path

import boto3
from botocore.exceptions import BotoCoreError
from dotenv import load_dotenv


PROJECT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_REGION = "us-east-1"
MAX_DETECT_BYTES = 19_999
MAX_CONCEPT_CHARACTERS = 10_000


def load_settings():
    load_dotenv(PROJECT_DIR / ".env", override=False)


def get_aws_region():
    load_settings()
    return os.getenv("AWS_REGION", DEFAULT_REGION).strip() or DEFAULT_REGION


def get_aws_profile():
    load_settings()
    return os.getenv("AWS_PROFILE", "").strip() or None


def get_session():
    """Create an SDK session using standard AWS profile/environment/role lookup."""
    load_settings()
    return boto3.Session(profile_name=get_aws_profile(), region_name=get_aws_region())


def has_aws_credentials():
    """Check credential availability without returning or displaying credentials."""
    try:
        return get_session().get_credentials() is not None
    except (BotoCoreError, ValueError):
        return False
