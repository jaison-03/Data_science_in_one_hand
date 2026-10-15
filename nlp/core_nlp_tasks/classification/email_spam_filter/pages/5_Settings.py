"""Streamlit entry point for model and security settings."""

import streamlit as st

from nlp.core_nlp_tasks.email_spam_filter.ui import render_settings


st.set_page_config(page_title="Email Security Settings", page_icon="⚙️", layout="wide")
render_settings()
