"""AWS Comprehend Medical client and operation wrappers."""

from time import perf_counter

from botocore.config import Config
from botocore.exceptions import (
    BotoCoreError,
    ClientError,
    ConnectTimeoutError,
    EndpointConnectionError,
    NoCredentialsError,
    PartialCredentialsError,
    ReadTimeoutError,
)

from nlp.core_nlp_tasks.medical_ner.config.settings import (
    get_aws_region,
    get_session,
)
from nlp.core_nlp_tasks.medical_ner.core.entity_mapper import (
    map_concept_response,
    map_entities_response,
)
from nlp.core_nlp_tasks.medical_ner.core.validators import (
    validate_comprehend_text,
    validate_concept_text,
)


class MedicalNerServiceError(RuntimeError):
    """Service error with safe user-facing text and an optional AWS error code."""

    def __init__(self, message, code=None):
        super().__init__(message)
        self.code = code


def create_client(session=None, region=None):
    session = session or get_session()
    selected_region = region or get_aws_region()
    try:
        return session.client(
            "comprehendmedical",
            region_name=selected_region,
            config=Config(
                connect_timeout=5,
                read_timeout=30,
                retries={"max_attempts": 3, "mode": "standard"},
                user_agent_extra="medical-ner-streamlit/1.0",
            ),
        )
    except (BotoCoreError, ValueError):
        raise MedicalNerServiceError(
            "AWS could not create a Comprehend Medical client for this region. "
            "Check the region and service availability."
        ) from None


def check_configuration():
    """Return non-secret AWS status without invoking a billable operation."""
    region = get_aws_region()
    try:
        session = get_session()
        credentials_available = session.get_credentials() is not None
        client = create_client(session=session, region=region)
        operations = set(client.meta.service_model.operation_names)
    except (BotoCoreError, ValueError, MedicalNerServiceError):
        return {
            "region": region,
            "credentials_available": False,
            "client_available": False,
            "operations_available": [],
        }
    required = {"DetectEntitiesV2", "InferICD10CM", "InferRxNorm"}
    return {
        "region": region,
        "credentials_available": credentials_available,
        "client_available": True,
        "operations_available": sorted(operations & required),
    }


def _safe_error(error):
    if isinstance(error, (NoCredentialsError, PartialCredentialsError)):
        return MedicalNerServiceError(
            "AWS credentials were not found. Configure an AWS profile, environment "
            "credentials, or an IAM role."
        )
    if isinstance(error, (EndpointConnectionError, ConnectTimeoutError, ReadTimeoutError)):
        return MedicalNerServiceError(
            "Could not reach AWS Comprehend Medical. Check your network and selected region."
        )
    if isinstance(error, ClientError):
        code = error.response.get("Error", {}).get("Code", "AWSServiceError")
        messages = {
            "AccessDeniedException": "The configured AWS identity lacks permission for this operation.",
            "UnrecognizedClientException": "AWS rejected the configured credentials. Check the AWS profile.",
            "InvalidSignatureException": "AWS could not authenticate this request. Check the AWS profile and system clock.",
            "ThrottlingException": "AWS is throttling requests. Wait briefly and try again.",
            "TooManyRequestsException": "AWS is throttling requests. Wait briefly and try again.",
            "ValidationException": "AWS rejected the request. Check the text and operation input limits.",
            "ResourceNotFoundException": "Comprehend Medical is unavailable for the selected region or operation.",
            "InternalServerException": "AWS reported a temporary service error. Please try again.",
            "ServiceUnavailableException": "AWS Comprehend Medical is temporarily unavailable.",
        }
        return MedicalNerServiceError(
            messages.get(code, "AWS Comprehend Medical could not process the request."),
            code=code,
        )
    if isinstance(error, BotoCoreError):
        return MedicalNerServiceError(
            "The AWS request could not be completed. Check the local AWS configuration."
        )
    return MedicalNerServiceError("Medical entity extraction failed. Please try again.")


def _call(operation, text):
    client = create_client()
    try:
        return getattr(client, operation)(Text=text)
    except (ClientError, BotoCoreError, ValueError) as error:
        raise _safe_error(error) from None


def detect_entities(text):
    """Call DetectEntitiesV2 and return normalized entities and duration."""
    normalized = validate_comprehend_text(text)
    started = perf_counter()
    response = _call("detect_entities_v2", normalized)
    duration = perf_counter() - started
    return {
        "operation": "DetectEntitiesV2",
        "region": get_aws_region(),
        "model_version": response.get("ModelVersion"),
        "entities": map_entities_response(response),
        "unmapped_attributes": response.get("UnmappedAttributes", []),
        "raw_response": response,
        "duration_seconds": duration,
    }


def infer_icd10cm(text):
    normalized = validate_concept_text(text)
    response = _call("infer_icd10_cm", normalized)
    return map_concept_response(response, "ICD10CMConcepts")


def infer_rxnorm(text):
    normalized = validate_concept_text(text)
    response = _call("infer_rx_norm", normalized)
    return map_concept_response(response, "RxNormConcepts")


def test_connection():
    """Run an explicit synthetic API request; AWS usage charges may apply."""
    return detect_entities("The patient has diabetes.")
