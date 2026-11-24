# ============================================================
# 智能问答 API — Agentic RAG 检索 + 生成 + 冲突消解 + 反馈回流
# ============================================================
from __future__ import annotations

import json
import time
import uuid
from pathlib import Path

from fastapi import APIRouter, Depends
from pydantic import BaseModel, Field

from src.core.auth import get_user_groups
from src.core.config import get_settings
from src.core.database import store
from src.core.logging_config import logger
from src.core.metrics import FEEDBACK_TOTAL, QUERY_LATENCY, QUERY_TOTAL
from src.retrieval import (
    agentic_engine,
    es_retriever,
    qdrant_retriever,
    reranker,
    retrieval_cache,
)

router = APIRouter()


# ------------------------------------------------------------------
# 请求/响应
# ------------------------------------------------------------------
class SearchRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=2000)
    top_k: int = Field(10, ge=1, le=50)
    enable_agentic: bool = True
    enable_cache: bool = True
    enable_conflict_detection: bool = True


class ChunkItem(BaseModel):
    chunk_id: int
    doc_id: int
    score: float
    content: str
    heading: str | None = None
    source: str = ""
    chunk_type: str = "child"


class SearchResponse(BaseModel):
    request_id: str
    query: str
    intent: str
    rewritten_queries: list[str]
    retrieval_rounds: int
    confidence: float
    chunks: list[ChunkItem]
    conflicts: list[dict] = []
    conflict_resolution: str = ""
    warning: str | None = None
    latency_ms: dict[str, float] = {}
    trace: list[dict] = []
    cached: bool = False


class AnswerRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=2000)
    top_k: int = 10


class AnswerResponse(BaseModel):
    request_id: str
    query: str
    answer: str
    intent: str
    confidence: float
    retrieval_rounds: int
    citations: list[dict]
    hallucination_rate: float | None = None
    conflicts: list[dict] = []
    conflict_resolution: str = ""
    latency_breakdown_ms: dict[str, float] = {}
    cached: bool = False


class FeedbackRequest(BaseModel):
    request_id: str
    feedback: str  # "positive" | "negative"


# ------------------------------------------------------------------
# POST /search — 纯检索（不生成答案）
# ------------------------------------------------------------------
@router.post("/search", response_model=SearchResponse)
async def search(req: SearchRequest, user_groups: list[str] = Depends(get_user_groups)):
    t_start = time.perf_counter()
    request_id = f"req_{uuid.uuid4().hex[:12]}"
    latencies = {}

    if req.enable_cache:
        t_cache = time.perf_counter()
        cache_key = f"{req.top_k}|{req.query}"
        cached = await retrieval_cache.get(cache_key, user_groups)
        latencies["cache_check_ms"] = (time.perf_counter() - t_cache) * 1000
        if cached:
            cached["request_id"] = request_id
            cached["cached"] = True
            return cached

    t_retrieve = time.perf_counter()
    if req.enable_agentic:
        result = await agentic_engine.retrieve(
            req.query, qdrant_retriever, es_retriever, reranker,
            top_k=req.top_k, filter_groups=user_groups,
        )
    else:
        result = await _baseline_retrieve(req.query, req.top_k, user_groups)
    latencies["retrieval_ms"] = (time.perf_counter() - t_retrieve) * 1000
    logger.info(f"search intent={result.intent} rounds={result.rounds_used} "
                f"chunks={len(result.chunks)} conf={result.confidence:.2f}")

    conflicts = result.conflicts if req.enable_conflict_detection else []
    conflict_resolution = ""
    if conflicts:
        conflict_resolution = await agentic_engine.conflict_resolver.resolve(
            conflicts, result.chunks
        )

    store.insert_query_log(
        request_id=request_id,
        original_query=req.query,
        rewritten_queries=json.dumps(result.rewritten_queries, ensure_ascii=False),
        intent_type=result.intent,
        retrieval_rounds=result.rounds_used,
        sufficiency_score=result.confidence,
        conflict_detected=1 if conflicts else 0,
        top_doc_ids=json.dumps([c.doc_id for c in result.chunks[:10]]),
        latency_ms_json=json.dumps(latencies),
    )

    chunks_out = [
        ChunkItem(
            chunk_id=c.chunk_id, doc_id=c.doc_id, score=round(c.score, 4),
            content=c.content[:500], heading=c.heading, source=c.source,
            chunk_type=c.chunk_type,
        )
        for c in result.chunks
    ]

    resp = SearchResponse(
        request_id=request_id,
        query=req.query,
        intent=result.intent,
        rewritten_queries=result.rewritten_queries,
        retrieval_rounds=result.rounds_used,
        confidence=round(result.confidence, 4),
        chunks=chunks_out,
        conflicts=conflicts,
        conflict_resolution=conflict_resolution,
        warning=result.warning,
        latency_ms=latencies,
        trace=result.trace,
    )

    if req.enable_cache:
        await retrieval_cache.set(f"{req.top_k}|{req.query}", resp.model_dump(), user_groups)

    QUERY_TOTAL.labels(endpoint="search").inc()
    QUERY_LATENCY.labels(endpoint="search").observe(time.perf_counter() - t_start)
    return resp


async def _baseline_retrieve(query: str, top_k: int, user_groups: list[str]):
    """非 Agentic 基线：单轮混合检索 + RRF 融合（用于对比评估）。"""
    from src.retrieval import AgenticResult, dedupe, reciprocal_rank_fusion

    emb = await qdrant_retriever.encode([query])
    dense = await qdrant_retriever.search(emb[0], top_k=100, filter_groups=user_groups)
    sparse = await es_retriever.search(query, top_k=50, filter_groups=user_groups)
    fused = reciprocal_rank_fusion([dense, sparse])
    fused = dedupe(fused)
    reranked = await reranker.rerank(query, fused, top_k=top_k)
    return AgenticResult(
        chunks=reranked, confidence=0.7, rounds_used=1,
        intent="hybrid", rewritten_queries=[query], trace=[],
    )


# ------------------------------------------------------------------
# POST /answer — 检索 + LLM 生成答案
# ------------------------------------------------------------------
@router.post("/answer", response_model=AnswerResponse)
async def answer(req: AnswerRequest, user_groups: list[str] = Depends(get_user_groups)):
    t_start = time.perf_counter()
    request_id = f"req_{uuid.uuid4().hex[:12]}"
    latencies = {}

    cache_key = f"{req.top_k}|{req.query}"
    cached = await retrieval_cache.get(cache_key, user_groups, namespace="answer")
    if cached:
        cached["request_id"] = request_id
        cached["cached"] = True
        return cached

    t_retrieve = time.perf_counter()
    result = await agentic_engine.retrieve(
        req.query, qdrant_retriever, es_retriever, reranker,
        top_k=req.top_k, filter_groups=user_groups,
    )
    latencies["retrieval_ms"] = (time.perf_counter() - t_retrieve) * 1000

    conflicts = result.conflicts
    conflict_resolution = ""
    if conflicts:
        conflict_resolution = await agentic_engine.conflict_resolver.resolve(
            conflicts, result.chunks
        )

    t_gen = time.perf_counter()
    answer_text, citations, hall_rate = await _generate_answer(
        req.query, result.chunks, conflicts, conflict_resolution
    )
    latencies["generation_ms"] = (time.perf_counter() - t_gen) * 1000
    latencies["total_ms"] = (time.perf_counter() - t_start) * 1000

    store.insert_query_log(
        request_id=request_id,
        original_query=req.query,
        rewritten_queries=json.dumps(result.rewritten_queries, ensure_ascii=False),
        intent_type=result.intent,
        retrieval_rounds=result.rounds_used,
        sufficiency_score=result.confidence,
        conflict_detected=1 if conflicts else 0,
        hallucination_rate=hall_rate,
        top_doc_ids=json.dumps([c.doc_id for c in result.chunks[:10]]),
        latency_ms_json=json.dumps(latencies),
    )

    resp = AnswerResponse(
        request_id=request_id,
        query=req.query,
        answer=answer_text,
        intent=result.intent,
        confidence=round(result.confidence, 4),
        retrieval_rounds=result.rounds_used,
        citations=citations,
        hallucination_rate=round(hall_rate, 4) if hall_rate is not None else None,
        conflicts=conflicts,
        conflict_resolution=conflict_resolution,
        latency_breakdown_ms=latencies,
    )
    await retrieval_cache.set(cache_key, resp.model_dump(), user_groups, namespace="answer")
    QUERY_TOTAL.labels(endpoint="answer").inc()
    QUERY_LATENCY.labels(endpoint="answer").observe(time.perf_counter() - t_start)
    return resp


async def _generate_answer(
    query: str, chunks: list, conflicts: list, conflict_resolution: str,
) -> tuple[str, list[dict], float | None]:
    """LLM 生成回答 + 引用标注 + Small-to-Big 父块回溯 + 幻觉率估算。"""
    if not chunks:
        return "根据当前知识库中的资料，无法回答此问题。", [], None

    # Small-to-Big：子块命中后回溯父块，取完整上下文参与生成
    context_parts = []
    chunks_ctx = []
    citations = []
    for i, c in enumerate(chunks[:10], 1):
        context_text = c.content
        if c.chunk_type == "child":
            parent = store.get_chunk_parent(c.doc_id, c.chunk_id)
            if parent and parent.get("content"):
                context_text = parent["content"]
        context_parts.append(f"[{i}] (文档{c.doc_id}) {c.heading or ''}\n{context_text}")
        chunks_ctx.append({
            "doc_title": f"文档{c.doc_id}",
            "section": c.heading or "",
            "content": context_text,
        })
        citations.append({"index": i, "doc_id": c.doc_id, "chunk_id": c.chunk_id,
                          "heading": c.heading, "score": round(c.score, 4)})

    context = "\n\n---\n\n".join(context_parts)
    conflict_note = f"\n\n[信息冲突提示]\n{conflict_resolution}" if conflict_resolution else ""

    prompt = _load_prompt_template("answer_generation.md", {
        "query": query, "chunks": chunks_ctx, "context": context, "conflict_note": conflict_note,
    })

    try:
        llm = await agentic_engine._get_llm()
        if llm is None:
            raise RuntimeError("no llm")
        resp = await llm.ainvoke(prompt)
        answer_text = resp.content if hasattr(resp, "content") else str(resp)
    except Exception:
        answer_text = f"基于检索到的 {len(chunks)} 个文档片段，相关内容如下：\n\n"
        for i, c in enumerate(chunks[:5], 1):
            answer_text += f"[citation:{i}] {c.content[:300]}...\n"

    # 幻觉率估算（引用覆盖度：回答中实际引用的 citation 占比）
    import re
    cited_indices = {int(m.group(1)) for m in re.finditer(r"\[citation:(\d+)\]", answer_text)}
    hall_rate = 0.0 if not citations else 1.0 - (len(cited_indices) / len(citations))
    hall_rate = min(max(hall_rate, 0.0), 1.0)

    return answer_text, citations, hall_rate


def _load_prompt_template(name: str, ctx: dict) -> str:
    """加载 configs/prompts 下的 Jinja2 模板（answer_generation / hallucination_check / query_rewrite）。"""
    path = get_settings().project_root / "configs" / "prompts" / name
    if path.exists():
        try:
            from jinja2 import Template
            tmpl = Template(path.read_text(encoding="utf-8"))
            rendered = tmpl.render(**ctx)
            # 取 system 之后的正文
            body = rendered.split("system: |", 1)[-1].strip()
            return body if body else rendered
        except Exception:
            pass
    # 兜底：内联 prompt
    return (
        f"基于以下文档片段回答问题，每句话标注 [citation:X]，不要编造：\n\n"
        f"{ctx.get('context', '')}{ctx.get('conflict_note', '')}\n\n问题: {ctx.get('query', '')}\n\n回答:"
    )


# ------------------------------------------------------------------
# POST /feedback — 用户反馈（线上闭环 + 可信度权重回流）
# ------------------------------------------------------------------
@router.post("/feedback")
async def feedback(req: FeedbackRequest):
    store.update_query_log(req.request_id, user_feedback=req.feedback)
    FEEDBACK_TOTAL.labels(feedback=req.feedback).inc()

    # 反馈回流：点踩降权 / 点赞升权（可迭代可信度模型）
    log = store.get_query_log(req.request_id)
    if log and log.get("top_doc_ids"):
        try:
            doc_ids = json.loads(log["top_doc_ids"])
            for doc_id in doc_ids:
                store.apply_feedback(int(doc_id), req.feedback)
        except (ValueError, TypeError, json.JSONDecodeError):
            pass

    return {"message": f"反馈已记录并回流到可信度模型: {req.feedback}"}
