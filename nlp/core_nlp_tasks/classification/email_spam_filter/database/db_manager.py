"""SQLite persistence for analyzed emails and application configuration."""

import json
import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parents[1]
DB_PATH = PROJECT_DIR / "database" / "email_spam_filter.db"


def _connect(db_path=None):
    path = Path(db_path) if db_path is not None else DB_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(path)
    connection.row_factory = sqlite3.Row
    return connection


def initialize_database(db_path=None):
    """Create all application tables and indexes if they do not exist."""
    with _connect(db_path) as connection:
        connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS analyses (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                subject TEXT NOT NULL,
                sender TEXT NOT NULL,
                predicted_label TEXT NOT NULL CHECK (predicted_label IN ('spam', 'ham')),
                risk_score REAL NOT NULL CHECK (risk_score >= 0 AND risk_score <= 100),
                ml_probability REAL NOT NULL CHECK (ml_probability >= 0 AND ml_probability <= 1),
                heuristic_findings TEXT NOT NULL,
                groq_used INTEGER NOT NULL DEFAULT 0 CHECK (groq_used IN (0, 1))
            );
            CREATE INDEX IF NOT EXISTS idx_analyses_timestamp ON analyses(timestamp DESC);
            CREATE INDEX IF NOT EXISTS idx_analyses_label ON analyses(predicted_label);
            CREATE TABLE IF NOT EXISTS settings (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );
            """
        )


def record_analysis(
    subject,
    sender,
    predicted_label,
    risk_score,
    ml_probability,
    heuristic_findings,
    groq_used=False,
    db_path=None,
):
    """Store an analysis using bound SQL parameters and return its row id."""
    if predicted_label not in {"spam", "ham"}:
        raise ValueError("predicted_label must be 'spam' or 'ham'.")
    if not 0 <= float(risk_score) <= 100:
        raise ValueError("risk_score must be between 0 and 100.")
    if not 0 <= float(ml_probability) <= 1:
        raise ValueError("ml_probability must be between 0 and 1.")
    initialize_database(db_path)
    timestamp = datetime.now(timezone.utc).isoformat(timespec="seconds")
    findings_json = json.dumps(heuristic_findings, ensure_ascii=True)
    with _connect(db_path) as connection:
        cursor = connection.execute(
            """
            INSERT INTO analyses
                (timestamp, subject, sender, predicted_label, risk_score,
                 ml_probability, heuristic_findings, groq_used)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                timestamp,
                str(subject),
                str(sender),
                predicted_label,
                float(risk_score),
                float(ml_probability),
                findings_json,
                int(bool(groq_used)),
            ),
        )
        return int(cursor.lastrowid)


def get_analyses(search="", label=None, min_risk=0, limit=500, db_path=None):
    """Return newest matching analyses, safely filtering user-supplied terms."""
    initialize_database(db_path)
    clauses = ["risk_score >= ?"]
    parameters = [float(min_risk)]
    if label in {"spam", "ham"}:
        clauses.append("predicted_label = ?")
        parameters.append(label)
    if search:
        clauses.append("(subject LIKE ? ESCAPE '\\' OR sender LIKE ? ESCAPE '\\')")
        escaped_search = str(search).replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
        parameters.extend([f"%{escaped_search}%", f"%{escaped_search}%"])
    parameters.append(max(1, min(int(limit), 5000)))
    query = (
        "SELECT id, timestamp, subject, sender, predicted_label, risk_score, "
        "ml_probability, heuristic_findings, groq_used FROM analyses WHERE "
        + " AND ".join(clauses)
        + " ORDER BY timestamp DESC, id DESC LIMIT ?"
    )
    with _connect(db_path) as connection:
        rows = connection.execute(query, parameters).fetchall()
    results = [dict(row) for row in rows]
    for row in results:
        row["heuristic_findings"] = json.loads(row["heuristic_findings"])
        row["groq_used"] = bool(row["groq_used"])
    return results


def get_summary(db_path=None):
    """Return aggregate dashboard counts for all persisted analyses."""
    initialize_database(db_path)
    with _connect(db_path) as connection:
        row = connection.execute(
            """
            SELECT COUNT(*) AS total,
                   SUM(CASE WHEN predicted_label = 'spam' THEN 1 ELSE 0 END) AS spam,
                   SUM(CASE WHEN predicted_label = 'ham' THEN 1 ELSE 0 END) AS ham,
                   SUM(CASE WHEN risk_score >= 70 THEN 1 ELSE 0 END) AS high_risk,
                   SUM(groq_used) AS groq_used
            FROM analyses
            """
        ).fetchone()
    return {key: int(row[key] or 0) for key in ("total", "spam", "ham", "high_risk", "groq_used")}


def clear_history(older_than_days=None, db_path=None):
    """Delete all records, or only records older than the requested age."""
    initialize_database(db_path)
    with _connect(db_path) as connection:
        if older_than_days is None:
            cursor = connection.execute("DELETE FROM analyses")
        else:
            cutoff = (datetime.now(timezone.utc) - timedelta(days=int(older_than_days))).isoformat(
                timespec="seconds"
            )
            cursor = connection.execute("DELETE FROM analyses WHERE timestamp < ?", (cutoff,))
        return int(cursor.rowcount)


def get_setting(key, default=None, db_path=None):
    """Read one application setting."""
    initialize_database(db_path)
    with _connect(db_path) as connection:
        row = connection.execute("SELECT value FROM settings WHERE key = ?", (str(key),)).fetchone()
    return row["value"] if row is not None else default


def set_setting(key, value, db_path=None):
    """Insert or replace one application setting."""
    initialize_database(db_path)
    timestamp = datetime.now(timezone.utc).isoformat(timespec="seconds")
    with _connect(db_path) as connection:
        connection.execute(
            """
            INSERT INTO settings (key, value, updated_at) VALUES (?, ?, ?)
            ON CONFLICT(key) DO UPDATE SET value = excluded.value, updated_at = excluded.updated_at
            """,
            (str(key), str(value), timestamp),
        )
