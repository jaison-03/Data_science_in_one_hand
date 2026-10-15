"""Unit tests for multilingual Q&A services without external API calls."""

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from nlp.advanced_nlp_and_applications.dialogue_systems.multilingual_question_answering_system import (
    config,
)
from nlp.advanced_nlp_and_applications.dialogue_systems.multilingual_question_answering_system.database import (
    db_manager,
)
from nlp.advanced_nlp_and_applications.dialogue_systems.multilingual_question_answering_system.services import (
    groq_service,
    language_service,
    llm_tasks,
)


def test_project_environment_loads_from_project_root(monkeypatch):
    loaded_paths = []
    monkeypatch.setattr(config, "load_dotenv", lambda path, override: loaded_paths.append(path))
    monkeypatch.setenv("GROQ_API_KEY", "test-key")

    assert config.get_groq_api_key() == "test-key"
    assert loaded_paths == [Path(config.__file__).resolve().parent / ".env"]


def test_groq_completion_uses_environment_model_and_secret_not_in_errors(monkeypatch):
    result = SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content="Answer text"))]
    )
    create = Mock(return_value=result)
    client = SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=create))
    )
    constructor = Mock(return_value=client)
    monkeypatch.setattr(groq_service, "get_groq_api_key", lambda: "private-test-key")
    monkeypatch.setattr(groq_service, "get_groq_model", lambda: "openai/gpt-oss-20b")
    monkeypatch.setattr(groq_service, "Groq", constructor)

    answer = groq_service.create_chat_completion(
        [{"role": "user", "content": "Hello"}], temperature=0.2, max_output_tokens=512
    )

    assert answer == "Answer text"
    assert constructor.call_args.kwargs["api_key"] == "private-test-key"
    assert create.call_args.kwargs["model"] == "openai/gpt-oss-20b"
    assert create.call_args.kwargs["max_completion_tokens"] == 512


def test_missing_key_is_safe_and_no_api_request(monkeypatch):
    constructor = Mock()
    monkeypatch.setattr(groq_service, "get_groq_api_key", lambda: "")
    monkeypatch.setattr(groq_service, "Groq", constructor)

    with pytest.raises(groq_service.GroqServiceError, match="GROQ_API_KEY") as error:
        groq_service.create_chat_completion([{"role": "user", "content": "Hi"}])

    assert "private-test-key" not in str(error.value)
    constructor.assert_not_called()


def test_language_detection_and_short_input():
    assert language_service.detect_language("What is machine learning?")["name"] == "English"
    assert language_service.detect_language("Hi")["confidence"] is None


def test_context_only_prompt_and_validation(monkeypatch):
    completion = Mock(return_value="From context.")
    monkeypatch.setattr(llm_tasks, "create_chat_completion", completion)

    result = llm_tasks.answer_question(
        question="What does Earth orbit?",
        answer_language="English",
        mode="Context-Only",
        context="The Earth revolves around the Sun.",
    )

    assert result == "From context."
    prompt = completion.call_args.args[0][-1]["content"]
    assert "Answer exclusively from the quoted context" in prompt
    assert "<context>" in prompt
    with pytest.raises(ValueError, match="Add context"):
        llm_tasks.answer_question(
            question="What is this?", answer_language="English", mode="Context-Only"
        )


def test_translation_and_summary_prompts(monkeypatch):
    completion = Mock(side_effect=["Hola", "A short summary."])
    monkeypatch.setattr(llm_tasks, "create_chat_completion", completion)

    assert llm_tasks.translate_text("Hello", "English", "Spanish") == "Hola"
    assert llm_tasks.summarize_text(
        "A longer passage.", "Short", "English", "Two sentences"
    ) == "A short summary."
    assert "Spanish" in completion.call_args_list[0].args[0][0]["content"]
    assert "Two sentences" in completion.call_args_list[1].args[0][0]["content"]


def test_database_persistence_and_delete(tmp_path):
    path = tmp_path / "qa.db"
    conversation_id = db_manager.create_conversation("Question", path)
    db_manager.save_exchange(
        conversation_id,
        "Question",
        "Answer",
        "English",
        "English",
        "Tamil",
        db_path=path,
    )

    messages = db_manager.get_messages(conversation_id, db_path=path)
    assert [message["content"] for message in messages] == ["Question", "Answer"]
    assert db_manager.get_dashboard_summary(path) == {"conversations": 1, "questions": 1}
    assert db_manager.delete_conversation(conversation_id, path)
    assert db_manager.get_messages(conversation_id, db_path=path) == []


def test_database_search_escapes_wildcards(tmp_path):
    path = tmp_path / "qa.db"
    conversation_id = db_manager.create_conversation("100% question", path)
    db_manager.save_exchange(
        conversation_id,
        "100% question",
        "Answer",
        "English",
        "English",
        "English",
        db_path=path,
    )

    assert len(db_manager.list_conversations(search="100%", db_path=path)) == 1
    assert db_manager.list_conversations(search="100_", db_path=path) == []
