"""Streamlit entry point for searchable analysis history."""

import streamlit as st

from nlp.core_nlp_tasks.email_spam_filter.ui import render_analysis_history


st.set_page_config(page_title="Analysis History", page_icon="🗂️", layout="wide")
render_analysis_history()
