"""Safe, centralized access to Groq chat completions."""

from groq import (
    APIConnectionError,
    APIStatusError,
    APITimeoutError,
    AuthenticationError,
    BadRequestError,
    Groq,
    GroqError,
    RateLimitError,
)

from nlp.advanced_nlp_and_applications.dialogue_systems.multilingual_question_answering_system.config import (
    get_groq_api_key,
    get_groq_model,
)


class GroqServiceError(RuntimeError):
    """An API failure with a user-safe message that contains no request data."""


def groq_is_configured():
    """Return whether a non-placeholder Groq key is available."""
    return bool(get_groq_api_key())


def create_chat_completion(messages, temperature=0.2, max_output_tokens=1024):
    """Request a completion using validated settings and sanitized API errors."""
    if not isinstance(temperature, (int, float)) or not 0 <= temperature <= 2:
        raise ValueError("Temperature must be between 0 and 2.")
    if not isinstance(max_output_tokens, int) or not 1 <= max_output_tokens <= 8192:
        raise ValueError("Maximum output tokens must be between 1 and 8192.")

    api_key = get_groq_api_key()
    if not api_key:
        raise GroqServiceError(
            "Groq API key is not configured. Add GROQ_API_KEY to the project .env file."
        )

    try:
        client = Groq(api_key=api_key, timeout=45.0, max_retries=1)
        response = client.chat.completions.create(
            model=get_groq_model(),
            messages=messages,
            temperature=float(temperature),
            max_completion_tokens=max_output_tokens,
        )
    except AuthenticationError:
        raise GroqServiceError(
            "Groq authentication failed. Check GROQ_API_KEY in the project .env file."
        ) from None
    except RateLimitError:
        raise GroqServiceError(
            "Groq is temporarily rate-limiting requests. Please wait and try again."
        ) from None
    except APITimeoutError:
        raise GroqServiceError("Groq did not respond in time. Please try again.") from None
    except APIConnectionError:
        raise GroqServiceError(
            "Could not connect to Groq. Check your network connection and try again."
        ) from None
    except (BadRequestError, APIStatusError):
        raise GroqServiceError(
            "Groq could not process this request. Check the configured model and try again."
        ) from None
    except GroqError:
        raise GroqServiceError("The Groq request failed. Please try again.") from None

    if not response.choices or not response.choices[0].message.content:
        raise GroqServiceError("Groq returned an empty response. Please try again.")
    return response.choices[0].message.content.strip()
