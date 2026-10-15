from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate

from core.llm_service import LLMService
from core.prompts import build_rag_prompt


class FakeChatClient:
    def __init__(self):
        self.messages = None

    def invoke(self, messages):
        self.messages = messages
        return type("Response", (), {"content": "Grounded answer"})()


def test_generate_passes_concrete_rag_messages_to_client():
    service = LLMService.__new__(LLMService)
    service.client = FakeChatClient()
    prompt = build_rag_prompt("What is documented?", "The answer is 42.")

    answer = service.generate(prompt)

    assert answer == "Grounded answer"
    assert len(service.client.messages) == 2
    assert isinstance(service.client.messages[0], SystemMessage)
    assert isinstance(service.client.messages[1], HumanMessage)
    assert "The answer is 42." in service.client.messages[1].content


def test_generate_formats_chat_prompt_template_before_invoking_client():
    service = LLMService.__new__(LLMService)
    service.client = FakeChatClient()
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a careful assistant."),
        ("human", "Answer this question."),
    ])

    answer = service.generate(prompt)

    assert answer == "Grounded answer"
    assert isinstance(service.client.messages[0], SystemMessage)
    assert isinstance(service.client.messages[1], HumanMessage)
