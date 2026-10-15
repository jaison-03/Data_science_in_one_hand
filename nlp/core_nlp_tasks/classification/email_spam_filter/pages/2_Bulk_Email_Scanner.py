"""Streamlit entry point for CSV batch scanning."""

import streamlit as st

from nlp.core_nlp_tasks.email_spam_filter.ui import render_bulk_scanner


st.set_page_config(page_title="Bulk Email Scanner", page_icon="📨", layout="wide")
render_bulk_scanner()
