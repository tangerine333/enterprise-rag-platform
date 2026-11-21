# ============================================================
# 智能问答 API — Agentic RAG 检索 + 生成 + 冲突消解
# ============================================================
from __future__ import annotations

import json
import time
import uuid

from fastapi import APIRouter
from pydantic import BaseModel, Field

from src.core.database import store
from src.retrieval import (
    agentic_engine,
    es_retriever,
    qdrant_retriever,
    reranker,
    retrieval_cache,
)

router = APIRouter()


# --- 请求/响应 ---
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


class FeedbackRequest(BaseModel):
    request_id: str
    feedback: str  # "positive" | "negative"


# ------------------------------------------------------------------
# POST /search — 纯检索（不生成答案）
# ------------------------------------------------------------------
@router.post("/search", response_model=SearchResponse)
async def search(req: SearchRequest):
    t_start = time.perf_counter()
    request_id = f"req_{uuid.uuid4().hex[:12]}"
    latencies = {}

    # 缓存检查
    if req.enable_cache:
        t_cache = time.perf_counter()
        cached = await retrieval_cache.get(req.query)
        latencies["cache_check_ms"] = (time.perf_counter() - t_cache) * 1000
        if cached:
            cached["request_id"] = request_id
            cached["cached"] = True
            return cached

    # Agentic RAG 检索
    t_retrieve = time.perf_counter()
    if req.enable_agentic:
        result = await agentic_engine.retrieve(
            req.query, qdrant_retriever, es_retriever, reranker, top_k=req.top_k,
        )
    else:
        # 非Agentic模式：单轮混合检索
        emb = await qdrant_retriever.encode([req.query])
        dense = await qdrant_retriever.search(emb[0], top_k=100)
        sparse = await es_retriever.search(req.query, top_k=50)
        all_results = dense + sparse
        all_results.sort(key=lambda x: x.score, reverse=True)
        # 去重
        seen = set()
        unique = []
        for r in all_results:
            if r.chunk_id not in seen:
                seen.add(r.chunk_id)
                unique.append(r)
        from src.retrieval import AgenticResult
        result = AgenticResult(
            chunks=await reranker.rerank(req.query, unique, top_k=req.top_k),
            confidence=0.7,
            rounds_used=1,
            intent="hybrid",
            rewritten_queries=[req.query],
        )
    latencies["retrieval_ms"] = (time.perf_counter() - t_retrieve) * 1000

    # 冲突检测
    conflicts = result.conflicts if req.enable_conflict_detection else []
    conflict_resolution = ""
    if conflicts:
        conflict_resolution = await agentic_engine.resolve_conflicts(conflicts, result.chunks)

    # 记录日志
    store.insert_query_log(
        request_id=request_id,
        original_query=req.query,
        rewritten_queries=json.dumps(result.rewritten_queries, ensure_ascii=False),
        intent_type=result.intent,
        retrieval_rounds=result.rounds_used,
        sufficiency_score=result.confidence,
        conflict_detected=1 if conflicts else 0,
        latency_ms_json=json.dumps(latencies),
    )

    chunks_out = [
        ChunkItem(
            chunk_id=c.chunk_id, doc_id=c.doc_id, score=round(c.score, 4),
            content=c.content[:500], heading=c.heading, source=c.source,
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
    )

    # 写入缓存
    if req.enable_cache:
        await retrieval_cache.set(req.query, resp.model_dump())

    return resp


# ------------------------------------------------------------------
# POST /answer — 检索 + LLM 生成答案
# ------------------------------------------------------------------
@router.post("/answer", response_model=AnswerResponse)
async def answer(req: AnswerRequest):
    t_start = time.perf_counter()
    request_id = f"req_{uuid.uuid4().hex[:12]}"
    latencies = {}

    # Step 1: Agentic RAG 检索
    t_retrieve = time.perf_counter()
    result = await agentic_engine.retrieve(
        req.query, qdrant_retriever, es_retriever, reranker, top_k=req.top_k,
    )
    latencies["retrieval_ms"] = (time.perf_counter() - t_retrieve) * 1000

    # Step 2: 冲突检测 + 消解
    conflicts = result.conflicts
    conflict_resolution = ""
    if conflicts:
        conflict_resolution = await agentic_engine.resolve_conflicts(conflicts, result.chunks)

    # Step 3: 上下文组装 + LLM 生成
    t_gen = time.perf_counter()
    answer_text, citations, hall_rate = await _generate_answer(
        req.query, result.chunks, conflicts, conflict_resolution
    )
    latencies["generation_ms"] = (time.perf_counter() - t_gen) * 1000

    latencies["total_ms"] = (time.perf_counter() - t_start) * 1000

    # 日志
    store.insert_query_log(
        request_id=request_id,
        original_query=req.query,
        rewritten_queries=json.dumps(result.rewritten_queries, ensure_ascii=False),
        intent_type=result.intent,
        retrieval_rounds=result.rounds_used,
        sufficiency_score=result.confidence,
        conflict_detected=1 if conflicts else 0,
        hallucination_rate=hall_rate,
        latency_ms_json=json.dumps(latencies),
    )

    return AnswerResponse(
        request_id=request_id,
        query=req.query,
        answer=answer_text,
        intent=result.intent,
        confidence=round(result.confidence, 4),
        retrieval_rounds=result.rounds_used,
        citations=citations,
        hallucination_rate=round(hall_rate, 4) if hall_rate else None,
        conflicts=conflicts,
        conflict_resolution=conflict_resolution,
        latency_breakdown_ms=latencies,
    )


async def _generate_answer(
    query: str, chunks: list, conflicts: list, conflict_resolution: str,
) -> tuple[str, list[dict], float | None]:
    """LLM 生成回答 + 引用标注 + 幻觉自检。"""
    if not chunks:
        return "根据当前知识库中的资料，无法回答此问题。", [], None

    # 上下文组装
    context_parts = []
    citations = []
    for i, c in enumerate(chunks[:10], 1):
        context_parts.append(f"[{i}] (文档{c.doc_id}) {c.heading or ''}\n{c.content}")
        citations.append({"index": i, "doc_id": c.doc_id, "chunk_id": c.chunk_id,
                          "heading": c.heading, "score": round(c.score, 4)})

    context = "\n\n---\n\n".join(context_parts)
    conflict_note = f"\n\n[信息冲突提示]\n{conflict_resolution}" if conflict_resolution else ""

    # LLM 生成
    prompt = f"""基于以下文档片段回答问题。每句话必须标注引用来源 [citation:X]。

## 规则
1. 只使用提供的文档片段，不要编造
2. 每句话标注 [citation:X]
3. 不足以回答时明确说明
4. 如果答案涉及多个文档且存在矛盾，同时呈现多方说法并给出推荐结论

## 文档片段
{context}
{conflict_note}

## 问题
{query}

## 回答:"""

    try:
        llm = await agentic_engine._get_llm()
        resp = await llm.ainvoke(prompt)
        answer_text = resp.content if hasattr(resp, "content") else str(resp)
    except Exception:
        # LLM 不可用时的降级策略：返回检索片段拼接
        answer_text = f"基于检索到的 {len(chunks)} 个文档片段，相关内容如下：\n\n"
        for i, c in enumerate(chunks[:5], 1):
            answer_text += f"[citation:{i}] {c.content[:300]}...\n"

    # 简易幻觉率估算（基于引用覆盖度）
    import re
    cited_indices = set()
    for m in re.finditer(r'\[citation:(\d+)\]', answer_text):
        cited_indices.add(int(m.group(1)))
    hall_rate = 0.0 if not citations else 1.0 - (len(cited_indices) / len(citations))
    hall_rate = min(hall_rate, 1.0)

    return answer_text, citations, hall_rate


# ------------------------------------------------------------------
# POST /feedback — 用户反馈
# ------------------------------------------------------------------
@router.post("/feedback")
async def feedback(req: FeedbackRequest):
    store.update_query_log(req.request_id, user_feedback=req.feedback)
    return {"message": f"反馈已记录: {req.feedback}"}
