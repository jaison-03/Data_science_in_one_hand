from langchain_core.messages import HumanMessage

from nlp.advanced_nlp_and_applications.dialogue_systems.finance_agentic_chatbot.agents.router_agent import (
    FinanceRouteDecision,
    route_finance_request,
)
from nlp.advanced_nlp_and_applications.dialogue_systems.finance_agentic_chatbot.workflow import (
    build_finance_workflow,
)


def test_router_returns_structured_finance_decision():
    expected = FinanceRouteDecision(route="finance", topic="debt")

    class FakeStructuredModel:
        def with_structured_output(self, schema):
            assert schema is FinanceRouteDecision
            return self

        def invoke(self, messages):
            assert messages[-1].content == "How should I prioritize my debt?"
            return expected

    assert route_finance_request("How should I prioritize my debt?", FakeStructuredModel()) == expected


def test_finance_question_routes_to_specialist_with_context():
    specialist_calls = []

    def specialist(messages, topic):
        specialist_calls.append((messages, topic))
        return "Start by listing your monthly income and essential expenses."

    workflow = build_finance_workflow(
        router=lambda _: FinanceRouteDecision(route="finance", topic="budgeting"),
        specialist=specialist,
    )
    result = workflow.invoke(
        {
            "messages": [HumanMessage(content="I get paid monthly.")],
            "user_input": "How can I build a budget?",
        }
    )

    assert result["route"] == "finance"
    assert result["topic"] == "budgeting"
    assert result["response"] == "Start by listing your monthly income and essential expenses."
    assert specialist_calls[0][0][-1].content == "How can I build a budget?"
    assert specialist_calls[0][1] == "budgeting"


def test_out_of_scope_question_does_not_call_finance_specialist():
    specialist_calls = []
    workflow = build_finance_workflow(
        router=lambda _: FinanceRouteDecision(route="out_of_scope"),
        specialist=lambda messages, topic: specialist_calls.append((messages, topic)) or "unexpected",
    )

    result = workflow.invoke({"messages": [], "user_input": "Write me a poem."})

    assert result["route"] == "out_of_scope"
    assert specialist_calls == []
    assert "finance-focused assistant" in result["response"]
