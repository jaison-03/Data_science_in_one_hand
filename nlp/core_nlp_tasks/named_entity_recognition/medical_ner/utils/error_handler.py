"""Translate known AWS SDK failures into safe user-facing messages."""

from botocore.exceptions import BotoCoreError, ClientError

from nlp.core_nlp_tasks.medical_ner.services.aws_comprehend_medical import (
    MedicalNerServiceError,
)


def user_error_message(error):
    if isinstance(error, MedicalNerServiceError):
        return str(error)
    if isinstance(error, ValueError):
        return str(error)
    if isinstance(error, ClientError):
        code = error.response.get("Error", {}).get("Code")
        return f"AWS request failed ({code or 'service error'}). Check AWS setup and permissions."
    if isinstance(error, BotoCoreError):
        return "AWS SDK configuration or network error. Check your AWS profile and selected region."
    return "An unexpected error occurred. No clinical text or credentials were logged."
