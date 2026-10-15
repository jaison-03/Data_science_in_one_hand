"""SQLite CRUD and SQL-parameterization tests."""

import sqlite3

from nlp.core_nlp_tasks.email_spam_filter.database.db_manager import (
    clear_history,
    get_analyses,
    get_setting,
    get_summary,
    initialize_database,
    record_analysis,
    set_setting,
)


def test_database_analysis_crud_settings_and_retention(tmp_path):
    db_path = tmp_path / "history.db"
    initialize_database(db_path)
    row_id = record_analysis(
        subject="Invoice",
        sender="sender@example.test",
        predicted_label="spam",
        risk_score=91,
        ml_probability=0.97,
        heuristic_findings=[{"category": "urgent_payment_demand"}],
        groq_used=True,
        db_path=db_path,
    )

    rows = get_analyses(search="Invoice", label="spam", db_path=db_path)
    assert rows[0]["id"] == row_id
    assert rows[0]["heuristic_findings"][0]["category"] == "urgent_payment_demand"
    assert rows[0]["groq_used"] is True
    assert get_summary(db_path) == {"total": 1, "spam": 1, "ham": 0, "high_risk": 1, "groq_used": 1}

    set_setting("spam_threshold", 0.7, db_path)
    assert get_setting("spam_threshold", db_path=db_path) == "0.7"
    assert clear_history(db_path=db_path) == 1
    assert get_summary(db_path)["total"] == 0


def test_database_queries_bind_search_input(tmp_path):
    db_path = tmp_path / "parameterized.db"
    initialize_database(db_path)
    injection = "' OR 1=1 --"
    record_analysis("ordinary", "sender", "ham", 2, 0.02, [], db_path=db_path)
    assert get_analyses(search=injection, db_path=db_path) == []
    with sqlite3.connect(db_path) as connection:
        assert connection.execute("SELECT COUNT(*) FROM analyses").fetchone()[0] == 1
