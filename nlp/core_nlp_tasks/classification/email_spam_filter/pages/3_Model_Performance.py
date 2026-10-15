"""Streamlit entry point for training and evaluation."""

import streamlit as st

from nlp.core_nlp_tasks.email_spam_filter.ui import render_model_performance


st.set_page_config(page_title="Model Performance", page_icon="📊", layout="wide")
render_model_performance()
