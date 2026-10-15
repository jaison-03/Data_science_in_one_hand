"""Streamlit entry point for single-email analysis."""

import streamlit as st

from nlp.core_nlp_tasks.email_spam_filter.ui import render_email_analyzer


st.set_page_config(page_title="Email Analyzer", page_icon="✉️", layout="wide")
render_email_analyzer()
