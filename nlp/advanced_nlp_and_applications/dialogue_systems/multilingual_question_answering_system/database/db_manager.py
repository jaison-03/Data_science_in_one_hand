"""SQLite storage for conversations and feedback."""

import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path

from nlp.advanced_nlp_and_applications.dialogue_systems.multilingual_question_answering_system.config import (
    get_database_path,
)


def _connect(db_path=None):
    path = Path(db_path) if db_path is not None else get_database_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(path, timeout=10)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys = ON")
    return connection


def initialize_database(db_path=None):
    with _connect(db_path) as connection:
        connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS conversations (
                id TEXT PRIMARY KEY,
                title TEXT NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                conversation_id TEXT NOT NULL REFERENCES conversations(id) ON DELETE CASCADE,
                role TEXT NOT NULL CHECK (role IN ('user', 'assistant')),
                content TEXT NOT NULL,
                detected_language TEXT,
                input_language TEXT,
                answer_language TEXT,
                language_confidence REAL,
                feedback INTEGER CHECK (feedback IN (-1, 1)),
                created_at TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_conversations_updated
                ON conversations(updated_at DESC);
            CREATE INDEX IF NOT EXISTS idx_messages_conversation
                ON messages(conversation_id, id);
            """
        )
        columns = {
            row["name"] for row in connection.execute("PRAGMA table_info(messages)")
        }
        if "input_language" not in columns:
            connection.execute("ALTER TABLE messages ADD COLUMN input_language TEXT")
        if "language_confidence" not in columns:
            connection.execute("ALTER TABLE messages ADD COLUMN language_confidence REAL")


def create_conversation(title, db_path=None):
    initialize_database(db_path)
    conversation_id = str(uuid.uuid4())
    timestamp = datetime.now(timezone.utc).isoformat(timespec="seconds")
    with _connect(db_path) as connection:
        connection.execute(
            "INSERT INTO conversations (id, title, created_at, updated_at) VALUES (?, ?, ?, ?)",
            (conversation_id, str(title)[:120], timestamp, timestamp),
        )
    return conversation_id


def save_exchange(
    conversation_id,
    question,
    answer,
    detected_language,
    input_language,
    answer_language,
    language_confidence=None,
    db_path=None,
):
    initialize_database(db_path)
    timestamp = datetime.now(timezone.utc).isoformat(timespec="seconds")
    with _connect(db_path) as connection:
        exists = connection.execute(
            "SELECT 1 FROM conversations WHERE id = ?", (str(conversation_id),)
        ).fetchone()
        if exists is None:
            raise ValueError("Conversation does not exist.")
        connection.executemany(
            """
            INSERT INTO messages
                (conversation_id, role, content, detected_language, input_language,
                 answer_language, language_confidence, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                (
                    conversation_id,
                    "user",
                    str(question),
                    detected_language,
                    input_language,
                    answer_language,
                    language_confidence,
                    timestamp,
                ),
                (
                    conversation_id,
                    "assistant",
                    str(answer),
                    detected_language,
                    input_language,
                    answer_language,
                    language_confidence,
                    timestamp,
                ),
            ],
        )
        connection.execute(
            "UPDATE conversations SET updated_at = ? WHERE id = ?",
            (timestamp, conversation_id),
        )


def list_conversations(search="", limit=100, db_path=None):
    initialize_database(db_path)
    query = (
        "SELECT DISTINCT c.id, c.title, c.created_at, c.updated_at "
        "FROM conversations c LEFT JOIN messages m ON m.conversation_id = c.id"
    )
    parameters = []
    if search:
        escaped = str(search).replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
        query += " WHERE c.title LIKE ? ESCAPE '\\' OR m.content LIKE ? ESCAPE '\\'"
        parameters.extend((f"%{escaped}%", f"%{escaped}%"))
    query += " ORDER BY c.updated_at DESC LIMIT ?"
    parameters.append(max(1, min(int(limit), 500)))
    with _connect(db_path) as connection:
        rows = connection.execute(query, parameters).fetchall()
    return [dict(row) for row in rows]


def get_messages(conversation_id, limit=200, db_path=None):
    initialize_database(db_path)
    max_messages = max(2, min(int(limit), 1000))
    with _connect(db_path) as connection:
        rows = connection.execute(
            """
            SELECT id, conversation_id, role, content, detected_language, input_language,
                   answer_language, language_confidence, feedback, created_at
            FROM messages WHERE conversation_id = ?
            ORDER BY id DESC LIMIT ?
            """,
            (str(conversation_id), max_messages),
        ).fetchall()
    return [dict(row) for row in reversed(rows)]


def set_feedback(message_id, helpful, db_path=None):
    initialize_database(db_path)
    with _connect(db_path) as connection:
        cursor = connection.execute(
            "UPDATE messages SET feedback = ? WHERE id = ? AND role = 'assistant'",
            (1 if helpful else -1, int(message_id)),
        )
        if cursor.rowcount != 1:
            raise ValueError("Assistant message was not found.")


def delete_conversation(conversation_id, db_path=None):
    initialize_database(db_path)
    with _connect(db_path) as connection:
        cursor = connection.execute(
            "DELETE FROM conversations WHERE id = ?", (str(conversation_id),)
        )
        return cursor.rowcount == 1


def clear_history(db_path=None):
    initialize_database(db_path)
    with _connect(db_path) as connection:
        cursor = connection.execute("DELETE FROM conversations")
        return cursor.rowcount


def get_dashboard_summary(db_path=None):
    initialize_database(db_path)
    with _connect(db_path) as connection:
        row = connection.execute(
            """
            SELECT COUNT(DISTINCT c.id) AS conversations,
                   SUM(CASE WHEN m.role = 'user' THEN 1 ELSE 0 END) AS questions
            FROM conversations c LEFT JOIN messages m ON m.conversation_id = c.id
            """
        ).fetchone()
    return {
        "conversations": int(row["conversations"] or 0),
        "questions": int((row["questions"] or 0) if row is not None else 0),
    }
