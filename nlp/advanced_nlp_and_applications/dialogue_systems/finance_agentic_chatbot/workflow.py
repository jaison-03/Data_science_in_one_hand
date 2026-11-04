"""LangGraph workflow for routing finance-chat requests to a specialist."""

from collections.abc import Callable
from typing import Literal, TypedDict

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_groq import ChatGroq
from langgraph.graph import END, StateGraph

from nlp.advanced_nlp_and_applications.dialogue_systems.finance_agentic_chatbot.agents.finance_specialist import (
    answer_finance_question,
)
from nlp.advanced_nlp_and_applications.dialogue_systems.finance_agentic_chatbot.agents.router_agent import (
    FinanceRouteDecision,
    route_finance_request,
)
from nlp.advanced_nlp_and_applications.dialogue_systems.multilingual_question_answering_system.config import (
    get_groq_api_key,
    get_groq_model,
)


class FinanceWorkflowState(TypedDict, total=False):
    messages: list[BaseMessage]
    user_input: str
    route: Literal["finance", "out_of_scope"]
    topic: str | None
    response: str


Router = Callable[[str], FinanceRouteDecision]
Specialist = Callable[[list[BaseMessage], str], str]


def _create_chat_model() -> BaseChatModel:
    api_key = get_groq_api_key()
    if not api_key:
        raise RuntimeError(
            "GROQ_API_KEY is not configured. Add it to the project-root .env before chatting."
        )
    return ChatGroq(api_key=api_key, model=get_groq_model(), temperature=0.2)


def build_finance_workflow(
    *,
    router: Router | None = None,
    specialist: Specialist | None = None,
    model: BaseChatModel | None = None,
):
    """Compile the router → specialist LangGraph; injectable nodes support offline tests."""
    configured_model = model
    if router is None or specialist is None:
        configured_model = configured_model or _create_chat_model()
    if router is None:
        if configured_model is None:
            raise RuntimeError("A model is required when no router is provided.")
        router = lambda text: route_finance_request(text, configured_model)
    if specialist is None:
        if configured_model is None:
            raise RuntimeError("A model is required when no specialist is provided.")
        specialist = lambda messages, topic: answer_finance_question(
            messages, topic, configured_model
        )
    route_request = router
    answer_request = specialist

    def router_node(state: FinanceWorkflowState) -> dict[str, str | None]:
        decision = route_request(state["user_input"])
        return {"route": decision.route, "topic": decision.topic}

    def finance_specialist_node(state: FinanceWorkflowState) -> dict[str, str]:
        conversation = [*state.get("messages", []), HumanMessage(content=state["user_input"])]
        topic = state.get("topic") or "financial_literacy"
        return {"response": answer_request(conversation, topic)}

    def out_of_scope_node(_: FinanceWorkflowState) -> dict[str, str]:
        return {
            "response": (
                "I’m a finance-focused assistant and can help with topics like budgeting, "
                "saving, debt, credit, investing concepts, taxes, and retirement. I can’t "
                "help with that request, but you can ask me a finance-related question."
            )
        }

    def choose_path(state: FinanceWorkflowState) -> str:
        return "finance_specialist" if state["route"] == "finance" else "out_of_scope"

    graph = StateGraph(FinanceWorkflowState)
    graph.add_node("router_agent", router_node)
    graph.add_node("finance_specialist", finance_specialist_node)
    graph.add_node("out_of_scope", out_of_scope_node)
    graph.set_entry_point("router_agent")
    graph.add_conditional_edges(
        "router_agent",
        choose_path,
        {
            "finance_specialist": "finance_specialist",
            "out_of_scope": "out_of_scope",
        },
    )
    graph.add_edge("finance_specialist", END)
    graph.add_edge("out_of_scope", END)
    return graph.compile()
