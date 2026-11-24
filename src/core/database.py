# ============================================================
# 元数据存储 — SQLite（零运维，开发/演示即开即用）
# 向量走 Qdrant，关键词走 ES，元数据/权限/可信度走这里
# ============================================================
from __future__ import annotations

import sqlite3
import json
import hashlib
from datetime import datetime
from pathlib import Path
from threading import Lock

from src.core.config import get_settings


class MetadataStore:
    """轻量元数据存储 (SQLite)。

    职责边界：
    - Qdrant  → 向量语义检索
    - ES      → BM25 关键词检索
    - SQLite  → 文档元数据 / 权限 / 可信度权重 / 查询日志
    """

    def __init__(self, db_path: str | None = None):
        if db_path is not None:
            self.db_path = db_path
        else:
            settings = get_settings()
            db_path = settings.project_root / "data" / "metadata.db"
            db_path.parent.mkdir(parents=True, exist_ok=True)
            self.db_path = str(db_path)
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
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
                    version TEXT,                     -- 版本号 "v2.0"（冲突消解：新旧）
                    department TEXT,                  -- 发布部门（冲突消解：层级）
                    authority_score REAL DEFAULT 0.5, -- 可信度权重（反馈回流修正）
                    checksum TEXT,                    -- SHA-256（增量更新去重）
                    permission_groups TEXT DEFAULT '*', -- 权限组（* = 全员可见）
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
                    top_doc_ids TEXT,
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

        self._migrate()

    def _migrate(self):
        """轻量 schema 迁移：为旧库补充新增列（幂等）。"""
        with self._lock:
            conn = self._get_conn()
            for table, cols in {
                "documents": {
                    "version": "TEXT",
                    "department": "TEXT",
                    "authority_score": "REAL DEFAULT 0.5",
                    "checksum": "TEXT",
                    "permission_groups": "TEXT DEFAULT '*'",
                },
                "query_logs": {
                    "top_doc_ids": "TEXT",
                },
            }.items():
                existing = {row[1] for row in conn.execute(f"PRAGMA table_info({table})")}
                for col, ddl in cols.items():
                    if col not in existing:
                        conn.execute(f"ALTER TABLE {table} ADD COLUMN {col} {ddl}")
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_docs_checksum ON documents(checksum)"
            )
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

    def find_by_checksum(self, checksum: str) -> dict | None:
        conn = self._get_conn()
        row = conn.execute(
            "SELECT * FROM documents WHERE checksum = ?", (checksum,)
        ).fetchone()
        conn.close()
        return dict(row) if row else None

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

    # --- 可信度权重（反馈回流） ---
    def apply_feedback(self, doc_id: int, feedback: str, step: float = 0.05):
        """点赞升权 / 点踩降权，形成可迭代可信度模型。"""
        delta = step if feedback == "positive" else -step
        with self._lock:
            conn = self._get_conn()
            conn.execute(
                "UPDATE documents SET authority_score = MIN(1.0, MAX(0.0, authority_score + ?)), updated_at = ? WHERE id = ?",
                (delta, datetime.now().isoformat(), doc_id),
            )
            conn.commit()
            conn.close()

    def get_documents_meta(self, doc_ids: list[int]) -> dict[int, dict]:
        """批量取文档权威性元数据（检索后给冲突消解用）。"""
        if not doc_ids:
            return {}
        conn = self._get_conn()
        q = ",".join("?" for _ in doc_ids)
        rows = conn.execute(
            f"SELECT id, version, department, authority_score, updated_at FROM documents WHERE id IN ({q})",
            list(doc_ids),
        ).fetchall()
        conn.close()
        return {r["id"]: dict(r) for r in rows}

    # --- Chunk 操作 ---
    def insert_chunks(self, chunks: list[dict]) -> list[int]:
        """插入 chunks，返回自增 id 列表（顺序与入参一致）。"""
        ids = []
        with self._lock:
            conn = self._get_conn()
            for c in chunks:
                c.setdefault("created_at", datetime.now().isoformat())
                c.setdefault("metadata_json", "{}")
                cursor = conn.execute(
                    """INSERT INTO chunks (doc_id, chunk_type, chunk_index, parent_chunk_id,
                       content, heading, page_range, qdrant_point_id, metadata_json, created_at)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (c["doc_id"], c.get("chunk_type", "child"), c["chunk_index"],
                     c.get("parent_chunk_id"), c["content"], c.get("heading"),
                     c.get("page_range"), c.get("qdrant_point_id"),
                     c.get("metadata_json", "{}"), c.get("created_at", datetime.now().isoformat())),
                )
                ids.append(cursor.lastrowid)
            conn.commit()
            conn.close()
        return ids

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

    def get_parent_chunks(self, doc_id: int, parent_ids: list[int]) -> dict[int, dict]:
        """Small-to-Big 回溯：取父块完整上下文。"""
        if not parent_ids:
            return {}
        conn = self._get_conn()
        q = ",".join("?" for _ in parent_ids)
        rows = conn.execute(
            f"SELECT chunk_index, content, heading FROM chunks "
            f"WHERE doc_id = ? AND chunk_type = 'parent' AND chunk_index IN ({q})",
            [doc_id] + list(parent_ids),
        ).fetchall()
        conn.close()
        return {r["chunk_index"]: dict(r) for r in rows}

    def get_chunk_parent(self, doc_id: int, chunk_id: int) -> dict | None:
        """单个子块回溯父块（生成阶段用）。"""
        conn = self._get_conn()
        row = conn.execute(
            "SELECT parent_chunk_id FROM chunks WHERE id = ?", (chunk_id,)
        ).fetchone()
        if not row or not row["parent_chunk_id"]:
            conn.close()
            return None
        p = conn.execute(
            "SELECT id, content, heading FROM chunks "
            "WHERE doc_id = ? AND chunk_type = 'parent' AND chunk_index = ?",
            (doc_id, row["parent_chunk_id"]),
        ).fetchone()
        conn.close()
        return dict(p) if p else None

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

    def get_query_log(self, request_id: str) -> dict | None:
        conn = self._get_conn()
        row = conn.execute(
            "SELECT * FROM query_logs WHERE request_id = ?", (request_id,)
        ).fetchone()
        conn.close()
        return dict(row) if row else None

    def feedback_stats(self) -> dict:
        """点踩率 / 点赞率统计（在线监控指标）。"""
        conn = self._get_conn()
        total = conn.execute(
            "SELECT COUNT(*) FROM query_logs WHERE user_feedback IS NOT NULL"
        ).fetchone()[0]
        pos = conn.execute(
            "SELECT COUNT(*) FROM query_logs WHERE user_feedback = 'positive'"
        ).fetchone()[0]
        neg = conn.execute(
            "SELECT COUNT(*) FROM query_logs WHERE user_feedback = 'negative'"
        ).fetchone()[0]
        conn.close()
        return {
            "total_feedback": total,
            "positive": pos,
            "negative": neg,
            "dislike_rate": (neg / total) if total else 0.0,
        }


# 单例
store = MetadataStore()


def compute_checksum(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()
