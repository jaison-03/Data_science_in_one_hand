from pathlib import Path
import sys
import os

import streamlit as st
from dotenv import load_dotenv
from groq import APIError, Groq

ROOT = Path(__file__).resolve().parents[4]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

load_dotenv(ROOT / ".env", override=False)
MODEL = os.getenv("GROQ_MODEL", "").strip() or "openai/gpt-oss-20b"
MAX_TEXT_LENGTH = 50_000

st.set_page_config(page_title="News Categorization", page_icon="🗞️", layout="centered")

st.title("🗞️ Automated News Categorization Engine")
st.write(
    "Enter a news headline, sentence, or text snippet and the AI will classify "
    "it into the closest matching domain."
)

api_key = os.getenv("GROQ_API_KEY", "").strip()
if not api_key:
    try:
        api_key = st.secrets.get("GROQ_API_KEY", "").strip()
    except (AttributeError, KeyError, FileNotFoundError):
        api_key = ""

if not api_key:
    st.warning(
        "GROQ_API_KEY was not found in the project-root .env or Streamlit secrets."
    )
    api_key = st.text_input("Groq API key", type="password").strip()


def categorize_news(sentence: str) -> str:
    """Classify a news sentence using the configured Groq chat model."""
    if not api_key:
        raise RuntimeError("Configure GROQ_API_KEY before categorizing news.")

    system_prompt = (
        "You are an expert news classifier. Your task is to analyze the given sentence "
        "and categorize it strictly into one of the following standard news categories:\n"
        "- Politics\n"
        "- Technology\n"
        "- Business & Finance\n"
        "- Sports\n"
        "- Entertainment\n"
        "- Health & Science\n"
        "- World News\n"
        "- Crime & Law\n\n"
        "Rules:\n"
        "1. Output ONLY the category name. Do not include extra punctuation, explanations, or full sentences.\n"
        "2. If it spans multiple categories or doesn't fit neatly, pick the closest matching primary category."
    )

    client = Groq(api_key=api_key)
    completion = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Sentence: {sentence}"},
        ],
        temperature=0.1,
        max_tokens=20
    )
    category = completion.choices[0].message.content
    if not category:
        raise RuntimeError("Groq returned an empty category.")
    return category.strip()

with st.form("categorization_form"):
    user_input = st.text_area(
        "News Text / Headline",
        placeholder="For example: Apple announces a new chip architecture focused on on-device AI.",
        height=120,
        max_chars=MAX_TEXT_LENGTH,
    )
    submit_btn = st.form_submit_button("Categorize News")

if submit_btn:
    if not api_key:
        st.error("Configure GROQ_API_KEY to run categorization.")
    elif not user_input.strip():
        st.warning("Please enter a sentence or headline to categorize.")
    else:
        with st.spinner(f"Classifying with {MODEL}..."):
            try:
                category = categorize_news(user_input)
                st.success("Categorization Complete!")
                col1, col2 = st.columns([1, 2])
                with col1:
                    st.metric(label="Predicted Category", value=category)
                with col2:
                    st.info(f'**Analyzed Text:**\n\n"{user_input}"')
            except (APIError, RuntimeError) as error:
                st.error(f"Groq categorization failed: {error}")