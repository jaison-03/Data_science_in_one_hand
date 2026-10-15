"""Master launcher for the NLP project collection."""

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import streamlit as st

from nlp.common.apps import render_app


PROJECTS = {
    "News Categorization": "news_categorization",
    "Resume Parser": "resume_parser",
    "Contract Entity Extractor": "contract_extractor",
    "Medical NER": "medical_ner",
    "Abstractive Summarizer": "abstractive_summarizer",
    "Extractive Summarizer": "extractive_text_summarizer",
    "Speech-to-Text Transcriber": "speech_to_text_transcriber",
    "Emotion Detection from Voice": "emotion_detection_from_voice",
    "FAQ Chatbot": "faq_chatbot",
    "Information Retrieval Engine": "information_retrieval_engine",
    "Domain-Specific Chatbot": "domain_chatbot",
    "English to Spanish Translator": "english_to_spanish_translator",
    "Grammar Corrector": "grammar_corrector",
    "Context-Aware Chatbot": "context_aware_chatbot",
    "Simple Python Code Generator": "code_generator",
    "Blog Post Idea Generator": "blog_post_idea_generator",
    "Poetry Writing Assistant": "poetry_writing_assistant",
    "Text Normalizer": "text_normalizer",
    "Custom Tokenizer": "custom_tokenizer",
    "POS Tagger": "pos_tagger",
    "Keyword Extractor": "keyword_extractor",
    "TF-IDF Classifier": "tf_idf_classifier",
    "Word Vector Explorer": "word2vec_explorer",
    "N-gram Generator": "n_gram_generator",
    "Sentiment with Word Vectors": "sentiment_with_word_vectors",
    "Email Spam Filter (existing)": "existing_email_spam_filter",
    "Multilingual Question Answering (existing)": "existing_multilingual_qa",
    "RAG Document Chatbot (existing)": "existing_rag_chatbot",
}

st.set_page_config(page_title="NLP Project Launcher", page_icon="🧠", layout="wide")
st.title("NLP Project Launcher")
st.caption("Explore NLP applications powered by Groq. Apps use GROQ_API_KEY and GROQ_MODEL from the project-root .env.")
selection = st.selectbox("Choose an NLP project", list(PROJECTS))
profile = PROJECTS[selection]
if profile == "existing_email_spam_filter":
    from nlp.core_nlp_tasks.classification.email_spam_filter.ui import render_app as render_spam_filter

    render_spam_filter(embed=True)
elif profile == "existing_multilingual_qa":
    from nlp.advanced_nlp_and_applications.dialogue_systems.multilingual_question_answering_system.ui import (
        render_app as render_multilingual_qa,
    )

    render_multilingual_qa(embed=True)
elif profile == "existing_rag_chatbot":
    rag_root = (
        Path(__file__).resolve().parent
        / "advanced_nlp_and_applications"
        / "dialogue_systems"
        / "rag_based_document_chatbot"
    )
    if str(rag_root) not in sys.path:
        sys.path.insert(0, str(rag_root))
    from nlp.advanced_nlp_and_applications.dialogue_systems.rag_based_document_chatbot.app import (
        render_app as render_rag_chatbot,
    )

    render_rag_chatbot(embed=True)
else:
    render_app(profile)
