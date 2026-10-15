"""Standalone Streamlit entry point for the email security application."""

import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from nlp.core_nlp_tasks.email_spam_filter.ui import render_app


render_app()
