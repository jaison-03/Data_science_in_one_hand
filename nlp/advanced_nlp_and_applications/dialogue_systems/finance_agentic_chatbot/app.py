"""Streamlit interface for the finance agentic chatbot."""

from pathlib import Path
import sys

import streamlit as st
from groq import GroqError
from langchain_core.messages import AIMessage, HumanMessage

ROOT = Path(__file__).resolve().parents[4]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from nlp.advanced_nlp_and_applications.dialogue_systems.finance_agentic_chatbot.workflow import (
    build_finance_workflow,
)


@st.cache_resource
def _workflow():
    return build_finance_workflow()


def render_app(embed: bool = False) -> None:
    if not embed:
        st.set_page_config(page_title="Finance Agentic AI Chatbot", page_icon="💰")
    st.title("Finance Agentic AI Chatbot")
    st.caption(
        "A LangGraph router directs finance questions to a dedicated financial-education specialist."
    )
    st.info(
        "Educational information only, not individualized financial, investment, tax, or legal advice. "
        "Do not share account numbers, passwords, card details, or tax IDs."
    )

    if "finance_chat_history" not in st.session_state:
        st.session_state["finance_chat_history"] = []

    if st.sidebar.button(
        "Clear conversation",
        disabled=not st.session_state["finance_chat_history"],
    ):
        st.session_state["finance_chat_history"] = []
        st.rerun()

    for turn in st.session_state["finance_chat_history"]:
        with st.chat_message(turn["role"]):
            st.markdown(turn["content"])

    st.markdown("**Try asking:** How can I build a monthly budget?")
    user_input = st.chat_input("Ask a finance question")

    if user_input:
        history = st.session_state["finance_chat_history"]
        previous_messages = [
            HumanMessage(content=turn["content"])
            if turn["role"] == "user"
            else AIMessage(content=turn["content"])
            for turn in history
        ]
        with st.chat_message("user"):
            st.markdown(user_input)
        history.append({"role": "user", "content": user_input})

        try:
            with st.spinner("Routing your question to the finance specialist..."):
                result = _workflow().invoke(
                    {"messages": previous_messages, "user_input": user_input}
                )
        except RuntimeError as error:
            st.error(str(error))
        except GroqError:
            st.error(
                "The finance assistant could not reach Groq. "
                "Check your connection and model configuration."
            )
        else:
            response = result["response"]
            history.append({"role": "assistant", "content": response})
            with st.chat_message("assistant"):
                st.markdown(response)
                st.caption(f"Routed to: {result['topic'] or 'finance scope check'}")


if __name__ == "__main__":
    render_app()
