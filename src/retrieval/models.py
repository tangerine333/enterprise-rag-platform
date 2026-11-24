# ============================================================
# 检索数据模型
# ============================================================
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class SearchResult:
    """单条检索结果（统一 Qdrant / ES 两种来源）。"""
    chunk_id: int
    doc_id: int
    content: str
    score: float
    heading: str | None = None
    source: str = ""          # "qdrant" | "elasticsearch"
    chunk_type: str = "child"  # child | parent | table_row | table_summary
    table_summary: str | None = None
    # --- 冲突消解 · 信源权威性元数据（检索后从 SQLite 回填） ---
    version: str | None = None
    department: str | None = None
    updated_at: str | None = None
    authority_score: float = 0.5


@dataclass
class AgenticResult:
    """Agentic RAG 主循环结果。"""
    chunks: list[SearchResult]
    confidence: float
    rounds_used: int
    intent: str
    rewritten_queries: list[str]
    conflicts: list[dict] = field(default_factory=list)
    warning: str | None = None
    trace: list[dict] = field(default_factory=list)  # 检索链路（可观测性）
