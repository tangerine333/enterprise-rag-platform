# ============================================================
# 系统监控 API — 健康检查 / 统计 / 链路追踪
# ============================================================
from __future__ import annotations

import time

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from src.core.database import store

router = APIRouter()
_start_time = time.time()


class HealthResponse(BaseModel):
    status: str = "healthy"
    version: str
    uptime_seconds: float


class StatsResponse(BaseModel):
    total_documents: int
    total_chunks: int = 0
    total_queries: int = 0
    avg_confidence: float = 0.0
    dislike_rate: float = 0.0
    pending_index_tasks: int = 0


@router.get("/health", response_model=HealthResponse)
async def health():
    from src.core.config import get_settings
    return HealthResponse(
        version=get_settings().app_version,
        uptime_seconds=round(time.time() - _start_time, 1),
    )


@router.get("/stats", response_model=StatsResponse)
async def stats():
    conn = store._get_conn()
    total_chunks = conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
    total_queries = conn.execute("SELECT COUNT(*) FROM query_logs").fetchone()[0]
    avg_conf = conn.execute(
        "SELECT AVG(sufficiency_score) FROM query_logs WHERE sufficiency_score IS NOT NULL"
    ).fetchone()[0]
    conn.close()

    feedback = store.feedback_stats()
    return StatsResponse(
        total_documents=store.count_documents(),
        total_chunks=total_chunks,
        total_queries=total_queries,
        avg_confidence=round(avg_conf or 0.0, 4),
        dislike_rate=round(feedback["dislike_rate"], 4),
        pending_index_tasks=store.count_pending_sync_tasks(),
    )


@router.post("/reconcile")
async def reconcile_index():
    """重试文档删除后 Qdrant/ES 清理失败的补偿任务。"""
    from src.api.documents import reconcile_index_sync
    return await reconcile_index_sync()


@router.get("/trace/{request_id}")
async def trace(request_id: str):
    """查询单次检索的完整链路。"""
    log = store.get_query_log(request_id)
    if not log:
        raise HTTPException(404, "请求记录不存在")
    return log
