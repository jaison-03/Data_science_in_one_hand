"""Standalone Streamlit entry point for the multilingual Q&A application."""

import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[4]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from nlp.advanced_nlp_and_applications.dialogue_systems.multilingual_question_answering_system.ui import render_app


render_app()
