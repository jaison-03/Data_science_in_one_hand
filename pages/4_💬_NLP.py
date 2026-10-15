import streamlit as st

from nlp_projects import render_nlp_workspace

st.set_page_config(page_title="NLP Mind Map", page_icon="💬", layout="wide")
render_nlp_workspace(in_dashboard=False)