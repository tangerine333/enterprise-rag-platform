# ============================================================
# 系统监控 API
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


@router.get("/health", response_model=HealthResponse)
async def health():
    from src.core.config import get_settings
    return HealthResponse(
        version=get_settings().app_version,
        uptime_seconds=round(time.time() - _start_time, 1),
    )


@router.get("/stats", response_model=StatsResponse)
async def stats():
    return StatsResponse(
        total_documents=store.count_documents(),
    )


@router.get("/trace/{request_id}")
async def trace(request_id: str):
    """查询单次检索的完整链路。"""
    conn = store._get_conn()
    row = conn.execute(
        "SELECT * FROM query_logs WHERE request_id = ?", (request_id,)
    ).fetchone()
    conn.close()

    if not row:
        raise HTTPException(404, "请求记录不存在")

    return dict(row)
