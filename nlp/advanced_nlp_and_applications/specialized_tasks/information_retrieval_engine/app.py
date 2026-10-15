from pathlib import Path
import sys

import streamlit as st

ROOT = Path(__file__).resolve().parents[4]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from nlp.common.apps import render_app

st.set_page_config(page_title="Information Retrieval Engine", page_icon="🔎")
render_app("information_retrieval_engine")
