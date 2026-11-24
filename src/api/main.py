# ============================================================
# FastAPI 主应用 — Agentic RAG 智能问答平台
# 含：request_id 贯穿日志、统一异常处理、Prometheus /metrics
# ============================================================
from __future__ import annotations

import time
import uuid
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from src.core.config import get_settings
from src.core.logging_config import logger, request_id_var

settings = get_settings()
_start_time = time.time()


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info(f"{settings.app_name} v{settings.app_version} 启动")
    yield
    logger.info(f"{settings.app_name} 关闭")


app = FastAPI(
    title="一文通 · 多文档智能问答系统",
    description="LLM驱动的自适应检索 — 意图分类→工具注册表多路召回→充分性判断→迭代改写→冲突消解→生成回答",
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


# ------------------------------------------------------------------
# 请求上下文中间件：request_id 贯穿 + 访问日志
# ------------------------------------------------------------------
@app.middleware("http")
async def request_context(request: Request, call_next):
    request_id = f"req_{uuid.uuid4().hex[:12]}"
    token = request_id_var.set(request_id)
    start = time.perf_counter()
    try:
        response = await call_next(request)
    except Exception:
        logger.exception(f"{request.method} {request.url.path} 异常")
        response = JSONResponse(
            status_code=500,
            content={"detail": "服务器内部错误", "request_id": request_id},
        )
    finally:
        request_id_var.reset(token)

    duration = (time.perf_counter() - start) * 1000
    logger.info(f"{request.method} {request.url.path} -> {response.status_code} ({duration:.1f}ms)")
    response.headers["X-Request-ID"] = request_id
    return response


# ------------------------------------------------------------------
# 统一异常处理
# ------------------------------------------------------------------
@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception):
    logger.exception(f"未处理异常: {exc}")
    return JSONResponse(
        status_code=500,
        content={"detail": "服务器内部错误", "request_id": request_id_var.get()},
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


@app.get("/metrics")
async def metrics():
    """Prometheus 抓取端点。"""
    from prometheus_client import CONTENT_TYPE_LATEST, generate_latest
    from fastapi import Response
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)
