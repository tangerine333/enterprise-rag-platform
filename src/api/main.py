# ============================================================
# FastAPI 主应用 — Agentic RAG 智能问答平台
# ============================================================
from __future__ import annotations

import time
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.core.config import get_settings

settings = get_settings()
_start_time = time.time()


@asynccontextmanager
async def lifespan(app: FastAPI):
    print(f"[{settings.app_name} v{settings.app_version}] 启动中...")
    yield
    print(f"[{settings.app_name}] 已关闭")


app = FastAPI(
    title="Agentic RAG 企业多源异构文档智能问答平台",
    description="LLM驱动的自适应检索 — 意图分类→多路召回→充分性判断→迭代改写→冲突消解→生成回答",
    version=settings.app_version,
    lifespan=lifespan,
    docs_url=f"{settings.api_prefix}/docs",
    openapi_url=f"{settings.api_prefix}/openapi.json",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

_prefix = settings.api_prefix

from src.api import documents, search, system

app.include_router(documents.router, prefix=f"{_prefix}/documents", tags=["文档管理"])
app.include_router(search.router, prefix=f"{_prefix}/search", tags=["智能问答"])
app.include_router(system.router, prefix=f"{_prefix}/system", tags=["系统监控"])


@app.get(f"{_prefix}/health")
async def health():
    return {"status": "healthy", "version": settings.app_version,
            "uptime_seconds": round(time.time() - _start_time, 1)}
