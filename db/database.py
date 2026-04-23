"""
SQLite persistent storage for uploaded documents.

Database file is stored at: agexplain/data/agexplain.db
Tables:
    - documents: stores doc_id, filename, extracted text, parser type, and content hash
"""

import os
import sqlite3
from typing import Optional, Dict
from services.json_logger import log_event

# Path: agexplain/backend/db/ -> ../../data/agexplain.db -> agexplain/data/agexplain.db
_DB_PATH = os.path.normpath(
    os.path.join(os.path.dirname(__file__), "..", "..", "data", "agexplain.db")
)


def _get_connection() -> sqlite3.Connection:
    os.makedirs(os.path.dirname(_DB_PATH), exist_ok=True)
    conn = sqlite3.connect(_DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    """Create tables if they do not already exist."""
    with _get_connection() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS documents (
                doc_id   TEXT PRIMARY KEY,
                filename TEXT NOT NULL,
                text     TEXT NOT NULL,
                parser   TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        # Backward-compatible migration for existing SQLite files.
        cols = {row["name"] for row in conn.execute("PRAGMA table_info(documents)").fetchall()}
        if "content_hash" not in cols:
            conn.execute("ALTER TABLE documents ADD COLUMN content_hash TEXT")

        conn.execute("CREATE INDEX IF NOT EXISTS idx_documents_content_hash ON documents(content_hash)")
        conn.commit()
    log_event("db.init", path=_DB_PATH)


def save_document(doc_id: str, filename: str, text: str, parser: str, content_hash: Optional[str] = None) -> None:
    """Insert or replace a document record."""
    with _get_connection() as conn:
        conn.execute(
            "INSERT OR REPLACE INTO documents (doc_id, filename, text, parser, content_hash) VALUES (?, ?, ?, ?, ?)",
            (doc_id, filename, text, parser, content_hash),
        )
        conn.commit()


def get_document(doc_id: str) -> Optional[Dict[str, str]]:
    """Return document dict or None if not found."""
    with _get_connection() as conn:
        row = conn.execute(
            "SELECT filename, text, parser FROM documents WHERE doc_id = ?",
            (doc_id,),
        ).fetchone()
    if row is None:
        return None
    return {"filename": row["filename"], "text": row["text"], "parser": row["parser"]}


def get_document_by_content_hash(content_hash: str) -> Optional[Dict[str, str]]:
    """Return the most recent document with this content hash, or None."""
    with _get_connection() as conn:
        row = conn.execute(
            "SELECT doc_id, filename, text, parser FROM documents WHERE content_hash = ? ORDER BY created_at DESC LIMIT 1",
            (content_hash,),
        ).fetchone()
    if row is None:
        return None
    return {
        "doc_id": row["doc_id"],
        "filename": row["filename"],
        "text": row["text"],
        "parser": row["parser"],
    }


def get_all_documents() -> Dict[str, Dict[str, str]]:
    """Return all documents as {doc_id: {filename, text, parser, content_hash}}."""
    with _get_connection() as conn:
        rows = conn.execute(
            "SELECT doc_id, filename, text, parser, content_hash FROM documents"
        ).fetchall()
    return {
        row["doc_id"]: {
            "filename": row["filename"],
            "text": row["text"],
            "parser": row["parser"],
            "content_hash": row["content_hash"],
        }
        for row in rows
    }
