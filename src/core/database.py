# ============================================================
# 轻量数据库 — 仅存元数据，向量走 Qdrant，关键词走 ES
# ============================================================
from __future__ import annotations

import sqlite3
import json
from datetime import datetime
from pathlib import Path
from threading import Lock

from src.core.config import get_settings


class MetadataStore:
    """轻量元数据存储 (SQLite)。

    为什么不用 PostgreSQL？
    - 向量检索 → Qdrant（专业向量库，比 pgvector 性能好）
    - 关键词检索 → Elasticsearch（成熟的全文检索引擎）
    - 元数据 → SQLite（零运维，开发环境即开即用）
    各司其职，不把 PostgreSQL 既当向量库又当全文引擎又当关系库。
    """

    def __init__(self):
        settings = get_settings()
        db_path = settings.project_root / "data" / "metadata.db"
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self.db_path = str(db_path)
        self._lock = Lock()
        self._init_tables()

    def _get_conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA foreign_keys=ON")
        return conn

    def _init_tables(self):
        with self._lock:
            conn = self._get_conn()
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS documents (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    title TEXT NOT NULL,
                    file_type TEXT NOT NULL,
                    file_path TEXT,
                    file_size_bytes INTEGER,
                    source TEXT DEFAULT 'upload',
                    parse_status TEXT DEFAULT 'pending',
                    parse_error TEXT,
                    chunk_count INTEGER DEFAULT 0,
                    metadata_json TEXT DEFAULT '{}',
                    created_at TEXT NOT NULL DEFAULT (datetime('now')),
                    updated_at TEXT NOT NULL DEFAULT (datetime('now'))
                );

                CREATE TABLE IF NOT EXISTS chunks (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    doc_id INTEGER NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
                    chunk_type TEXT NOT NULL DEFAULT 'child',
                    chunk_index INTEGER NOT NULL,
                    parent_chunk_id INTEGER,
                    content TEXT NOT NULL,
                    heading TEXT,
                    page_range TEXT,
                    qdrant_point_id TEXT,
                    metadata_json TEXT DEFAULT '{}',
                    created_at TEXT NOT NULL DEFAULT (datetime('now'))
                );

                CREATE TABLE IF NOT EXISTS query_logs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    request_id TEXT NOT NULL UNIQUE,
                    original_query TEXT NOT NULL,
                    rewritten_queries TEXT,
                    intent_type TEXT,
                    retrieval_rounds INTEGER DEFAULT 1,
                    sufficiency_score REAL,
                    conflict_detected INTEGER DEFAULT 0,
                    hallucination_rate REAL,
                    user_feedback TEXT,
                    latency_ms_json TEXT DEFAULT '{}',
                    created_at TEXT NOT NULL DEFAULT (datetime('now'))
                );

                CREATE INDEX IF NOT EXISTS idx_chunks_doc ON chunks(doc_id);
                CREATE INDEX IF NOT EXISTS idx_chunks_parent ON chunks(parent_chunk_id);
                CREATE INDEX IF NOT EXISTS idx_docs_status ON documents(parse_status);
                CREATE INDEX IF NOT EXISTS idx_query_logs_req ON query_logs(request_id);
            """)
            conn.commit()
            conn.close()

    # --- 文档操作 ---
    def insert_document(self, **kwargs) -> int:
        kwargs.setdefault("created_at", datetime.now().isoformat())
        kwargs.setdefault("updated_at", datetime.now().isoformat())
        columns = ", ".join(kwargs.keys())
        placeholders = ", ".join("?" for _ in kwargs)
        with self._lock:
            conn = self._get_conn()
            cursor = conn.execute(
                f"INSERT INTO documents ({columns}) VALUES ({placeholders})",
                list(kwargs.values()),
            )
            conn.commit()
            doc_id = cursor.lastrowid
            conn.close()
        return doc_id

    def get_document(self, doc_id: int) -> dict | None:
        conn = self._get_conn()
        row = conn.execute("SELECT * FROM documents WHERE id = ?", (doc_id,)).fetchone()
        conn.close()
        return dict(row) if row else None

    def list_documents(self, status: str | None = None, limit: int = 50, offset: int = 0) -> list[dict]:
        conn = self._get_conn()
        if status:
            rows = conn.execute(
                "SELECT * FROM documents WHERE parse_status = ? ORDER BY created_at DESC LIMIT ? OFFSET ?",
                (status, limit, offset),
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT * FROM documents ORDER BY created_at DESC LIMIT ? OFFSET ?",
                (limit, offset),
            ).fetchall()
        conn.close()
        return [dict(r) for r in rows]

    def count_documents(self) -> int:
        conn = self._get_conn()
        count = conn.execute("SELECT COUNT(*) FROM documents").fetchone()[0]
        conn.close()
        return count

    def update_document(self, doc_id: int, **kwargs):
        kwargs["updated_at"] = datetime.now().isoformat()
        sets = ", ".join(f"{k} = ?" for k in kwargs)
        with self._lock:
            conn = self._get_conn()
            conn.execute(
                f"UPDATE documents SET {sets} WHERE id = ?",
                list(kwargs.values()) + [doc_id],
            )
            conn.commit()
            conn.close()

    def delete_document(self, doc_id: int):
        with self._lock:
            conn = self._get_conn()
            conn.execute("DELETE FROM documents WHERE id = ?", (doc_id,))
            conn.commit()
            conn.close()

    # --- Chunk 操作 ---
    def insert_chunks(self, chunks: list[dict]):
        with self._lock:
            conn = self._get_conn()
            for c in chunks:
                c.setdefault("created_at", datetime.now().isoformat())
                c.setdefault("metadata_json", "{}")
                conn.execute(
                    """INSERT INTO chunks (doc_id, chunk_type, chunk_index, parent_chunk_id,
                       content, heading, page_range, qdrant_point_id, metadata_json, created_at)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (c["doc_id"], c.get("chunk_type", "child"), c["chunk_index"],
                     c.get("parent_chunk_id"), c["content"], c.get("heading"),
                     c.get("page_range"), c.get("qdrant_point_id"),
                     c.get("metadata_json", "{}"), c.get("created_at", datetime.now().isoformat())),
                )
            conn.commit()
            conn.close()

    def get_chunks_by_doc(self, doc_id: int) -> list[dict]:
        conn = self._get_conn()
        rows = conn.execute(
            "SELECT * FROM chunks WHERE doc_id = ? ORDER BY chunk_index", (doc_id,)
        ).fetchall()
        conn.close()
        return [dict(r) for r in rows]

    def get_chunk(self, chunk_id: int) -> dict | None:
        conn = self._get_conn()
        row = conn.execute("SELECT * FROM chunks WHERE id = ?", (chunk_id,)).fetchone()
        conn.close()
        return dict(row) if row else None

    # --- 查询日志 ---
    def insert_query_log(self, **kwargs) -> int:
        kwargs.setdefault("created_at", datetime.now().isoformat())
        kwargs.setdefault("latency_ms_json", "{}")
        columns = ", ".join(kwargs.keys())
        placeholders = ", ".join("?" for _ in kwargs)
        with self._lock:
            conn = self._get_conn()
            cursor = conn.execute(
                f"INSERT INTO query_logs ({columns}) VALUES ({placeholders})",
                list(kwargs.values()),
            )
            conn.commit()
            log_id = cursor.lastrowid
            conn.close()
        return log_id

    def update_query_log(self, request_id: str, **kwargs):
        sets = ", ".join(f"{k} = ?" for k in kwargs)
        with self._lock:
            conn = self._get_conn()
            conn.execute(
                f"UPDATE query_logs SET {sets} WHERE request_id = ?",
                list(kwargs.values()) + [request_id],
            )
            conn.commit()
            conn.close()


# 单例
store = MetadataStore()
