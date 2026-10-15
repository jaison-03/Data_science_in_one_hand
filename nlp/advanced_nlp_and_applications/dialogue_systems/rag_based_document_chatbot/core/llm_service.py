from __future__ import annotations

from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_groq import ChatGroq

from config.settings import settings


class LLMService:
    def __init__(self, api_key: str | None = None, model_name: str | None = None):
        self.api_key = api_key or settings.groq_api_key
        self.model_name = model_name or settings.groq_model
        if not self.api_key:
            raise ValueError("GROQ_API_KEY is not set. Add it to your .env file before generating answers.")

        self.client = ChatGroq(model=self.model_name, api_key=self.api_key, temperature=0.1)

    def generate(self, prompt: str | list[Any] | Any) -> str:
        if isinstance(prompt, str):
            messages = [
                SystemMessage(content="You are a careful document-grounded assistant."),
                HumanMessage(content=prompt),
            ]
        elif hasattr(prompt, "format_messages"):
            messages = prompt.format_messages()
        elif hasattr(prompt, "to_messages"):
            messages = prompt.to_messages()
        elif hasattr(prompt, "messages"):
            messages = prompt.messages
        else:
            messages = prompt
        response = self.client.invoke(messages)
        return response.content.strip() if hasattr(response, "content") else str(response)
