"""Standalone Groq LLM playground with configurable system instructions."""
from pathlib import Path
import os
import streamlit as st
from dotenv import load_dotenv
from groq import APIError, Groq

ROOT=Path(__file__).resolve().parents[4]
load_dotenv(ROOT/".env",override=False)
MODEL=os.getenv("GROQ_MODEL","").strip() or "openai/gpt-oss-20b"
if not st.session_state.get("_embedded_ai_project"):
    st.set_page_config(page_title="Groq LLM Playground",page_icon="✨",layout="wide")
st.markdown("<style>.block-container{max-width:1100px;padding-top:2rem}.stChatMessage{border-radius:12px}</style>",unsafe_allow_html=True)
st.title("Large Language Model Playground")
st.caption(f"Chat with the configured Groq model · `{MODEL}`")
st.caption("Messages are sent to the Groq API for processing.")
key=os.getenv("GROQ_API_KEY","").strip()
if not key:
    st.error("GROQ_API_KEY is not configured in the root .env file.")
    st.stop()
system=st.sidebar.text_area("System prompt","You are a clear, accurate assistant. State uncertainty and ask clarifying questions when needed.",height=180)
temperature=st.sidebar.slider("Temperature",0.0,1.5,0.6,0.1)
max_tokens=st.sidebar.slider("Maximum response tokens",128,4096,1024,128)
if st.sidebar.button("Clear conversation"):
    st.session_state.pop("groq_playground_messages",None)
history=st.session_state.setdefault("groq_playground_messages",[])
for message in history:
    with st.chat_message(message["role"]): st.markdown(message["content"])
prompt=st.chat_input("Ask the model")
if prompt:
    if len(prompt)>30000:
        st.error("Message is too long (30,000 character limit).")
    else:
        with st.chat_message("user"): st.markdown(prompt)
        try:
            with st.spinner("Waiting for Groq..."):
                response=Groq(api_key=key).chat.completions.create(model=MODEL,messages=[{"role":"system","content":system},*history[-20:],{"role":"user","content":prompt}],temperature=temperature,max_tokens=max_tokens)
            answer=response.choices[0].message.content
            if not answer: raise RuntimeError("The model returned an empty response.")
            history.extend([{"role":"user","content":prompt},{"role":"assistant","content":answer}])
            del history[:-20]
            with st.chat_message("assistant"): st.markdown(answer)
        except (APIError,RuntimeError) as error:
            st.error(f"Groq request failed: {error}")
