"""AWS service wrapper tests that never contact AWS."""

from unittest.mock import Mock

import pytest
from botocore.exceptions import ClientError, NoCredentialsError

from nlp.core_nlp_tasks.medical_ner.services import (
    aws_comprehend_medical as service,
)


def test_detect_entities_calls_correct_operation_and_maps_response(monkeypatch):
    client = Mock()
    client.detect_entities_v2.return_value = {
        "Entities": [{"Text": "fever", "Category": "MEDICAL_CONDITION", "Type": "SYMPTOM"}],
        "ModelVersion": "1.0",
    }
    monkeypatch.setattr(service, "create_client", lambda: client)

    result = service.detect_entities("The patient has fever.")

    client.detect_entities_v2.assert_called_once_with(Text="The patient has fever.")
    assert result["operation"] == "DetectEntitiesV2"
    assert result["entities"][0]["text"] == "fever"


def test_missing_credentials_are_sanitized(monkeypatch):
    client = Mock()
    client.detect_entities_v2.side_effect = NoCredentialsError()
    monkeypatch.setattr(service, "create_client", lambda: client)

    with pytest.raises(service.MedicalNerServiceError, match="credentials were not found"):
        service.detect_entities("Synthetic sample.")


def test_access_denied_is_reported_without_raw_request(monkeypatch):
    client = Mock()
    client.detect_entities_v2.side_effect = ClientError(
        {"Error": {"Code": "AccessDeniedException", "Message": "request secret"}},
        "DetectEntitiesV2",
    )
    monkeypatch.setattr(service, "create_client", lambda: client)

    with pytest.raises(service.MedicalNerServiceError, match="lacks permission") as error:
        service.detect_entities("Synthetic patient text.")
    assert "request secret" not in str(error.value)
