# ============================================================
# Agentic RAG 核心引擎 — LangGraph 有状态检索决策回路
#
# classify(意图分类) → retrieve(工具注册表多路召回+RRF融合)
#   → judge(充分性判断) ──不足──> rewrite(查询改写) ──循环──> retrieve
#                         └─充分/达上限─> finalize(冲突消解+精排)
# 最多 3 轮，新增文档净增量 < 20% 提前终止。
#
# 并发安全：所有请求级状态（top_k/filter_groups/retrievers）都放进
# LangGraph state，随单次 ainvoke 流转，不再挂在共享单例 self 上。
# ============================================================
from __future__ import annotations

import hashlib
import re
import time
from typing import Any, TypedDict

from langgraph.graph import END, StateGraph

from src.core.config import get_settings, load_yaml_config
from src.core.database import store
from src.retrieval.conflict import ConflictResolver
from src.retrieval.fusion import dedupe, reciprocal_rank_fusion
from src.retrieval.intent import IntentClassifier
from src.retrieval.models import AgenticResult, SearchResult
from src.retrieval.tools import registry

# 各检索工具的召回条数（按工具语义配置）
TOOL_TOP_K = {"keyword": 50, "structured": 50, "semantic": 100, "multi_query": 30}


class RAGState(TypedDict):
    query: str
    intent: str
    retrieval_strategy: str
    confidence: float
    search_queries: list[str]
    round_num: int
    accumulated: list[SearchResult]
    round_new: int
    best_score: float
    best_chunks: list[SearchResult]
    rewritten_history: list[str]
    sufficiency: float
    trace: list[dict]
    result: AgenticResult | None
    # --- 请求级运行时状态（注入进 state，规避共享单例竞态） ---
    top_k: int
    filter_groups: list[str]
    qdrant: Any
    es: Any
    reranker: Any


class AgenticRAGEngine:
    """LangGraph 编排的 Agentic RAG 引擎（无请求级共享可变状态）。"""

    def __init__(self, llm=None):
        self.settings = get_settings()
        self.retrieval_config = load_yaml_config("retrieval.yaml")
        self.agentic_config = self.retrieval_config.get("agentic", {})
        self.intent_routing = self.retrieval_config.get("intent_routing", {})
        self.max_rounds = int(self.agentic_config.get("max_retrieval_rounds", 3))
        self.early_stop = float(self.agentic_config.get("early_stop_threshold", 0.8))
        self.min_gain = float(self.agentic_config.get("min_incremental_gain", 0.2))

        self._llm = llm
        self.intent_classifier = IntentClassifier()
        self.conflict_resolver = ConflictResolver()

        self.graph = self._build_graph()

    async def _get_llm(self):
        if self._llm is None:
            from src.core.llm import create_llm
            self._llm = create_llm(model=self.settings.llm_model, temperature=0.1)
        return self._llm

    # ------------------------------------------------------------------
    # 构建 LangGraph 状态图
    # ------------------------------------------------------------------
    def _build_graph(self):
        g = StateGraph(RAGState)
        g.add_node("classify", self._node_classify)
        g.add_node("retrieve", self._node_retrieve)
        g.add_node("judge", self._node_judge)
        g.add_node("rewrite", self._node_rewrite)
        g.add_node("finalize", self._node_finalize)

        g.set_entry_point("classify")
        g.add_edge("classify", "retrieve")
        g.add_edge("retrieve", "judge")
        g.add_conditional_edges(
            "judge",
            self._route_after_judge,
            {"rewrite": "rewrite", "finalize": "finalize"},
        )
        g.add_edge("rewrite", "retrieve")
        g.add_edge("finalize", END)
        return g.compile()

    # ------------------------------------------------------------------
    # 节点：意图分类 + 路由
    # ------------------------------------------------------------------
    async def _node_classify(self, state: RAGState) -> RAGState:
        t0 = time.perf_counter()
        analysis = await self.intent_classifier.classify(state["query"])
        state["intent"] = analysis.get("intent", "conceptual")
        state["retrieval_strategy"] = analysis.get("retrieval_strategy", "semantic")
        state["confidence"] = float(analysis.get("confidence", 0.6))
        state["search_queries"] = analysis.get("rewritten_queries") or [state["query"]]
        state["rewritten_history"] = [state["query"]]
        state["round_num"] = 1
        state["accumulated"] = []
        state["best_score"] = 0.0
        state["best_chunks"] = []
        state["trace"] = [{
            "step": "意图分类",
            "intent": state["intent"],
            "tools": self.intent_routing.get(state["intent"], {}).get("tools", []),
            "ms": round((time.perf_counter() - t0) * 1000, 1),
        }]
        return state

    # ------------------------------------------------------------------
    # 节点：工具注册表多路召回 + RRF 融合
    # ------------------------------------------------------------------
    async def _node_retrieve(self, state: RAGState) -> RAGState:
        t0 = time.perf_counter()
        intent = state["intent"]
        query = state["search_queries"][0] if state["search_queries"] else state["query"]

        # 配置驱动：意图 → 工具列表（来自 retrieval.yaml，单一事实源）
        tool_names = self.intent_routing.get(intent, {}).get("tools", ["semantic", "keyword"])

        result_sets: list[list[SearchResult]] = []
        for name in tool_names:
            tool = registry.get(name)
            if tool is None:
                continue
            kwargs: dict[str, Any] = {}
            if name == "multi_query":
                kwargs["extra_queries"] = state["search_queries"][1:3]
            res = await tool.func(
                query=query,
                top_k=TOOL_TOP_K.get(name, 50),
                filter_groups=state["filter_groups"],
                qdrant=state["qdrant"],
                es=state["es"],
                **kwargs,
            )
            if res:
                result_sets.append(res)

        fused = reciprocal_rank_fusion(result_sets)
        before = len(state["accumulated"])
        state["accumulated"] = dedupe(state["accumulated"] + fused)
        state["round_new"] = len(state["accumulated"]) - before

        state["trace"].append({
            "step": f"多路召回(R{state['round_num']})",
            "tools": tool_names,
            "query": query,
            "recalled": len(fused),
            "new": state["round_new"],
            "total": len(state["accumulated"]),
            "ms": round((time.perf_counter() - t0) * 1000, 1),
        })
        return state

    # ------------------------------------------------------------------
    # 节点：充分性判断
    # ------------------------------------------------------------------
    async def _node_judge(self, state: RAGState) -> RAGState:
        t0 = time.perf_counter()
        current = sorted(state["accumulated"], key=lambda x: x.score, reverse=True)
        sufficiency = await self._judge_sufficiency(state["query"], current[:10])
        state["sufficiency"] = sufficiency

        if sufficiency > state["best_score"]:
            state["best_score"] = sufficiency
            state["best_chunks"] = current[: state["top_k"]]

        state["trace"].append({
            "step": "充分性判断",
            "sufficiency": round(sufficiency, 3),
            "best_score": round(state["best_score"], 3),
            "ms": round((time.perf_counter() - t0) * 1000, 1),
        })
        return state

    async def _route_after_judge(self, state: RAGState) -> str:
        if state["sufficiency"] >= self.early_stop:
            return "finalize"
        if state["round_num"] >= self.max_rounds:
            return "finalize"
        if state["round_num"] > 1 and state["accumulated"]:
            gain = state["round_new"] / len(state["accumulated"])
            if gain < self.min_gain:
                return "finalize"
        return "rewrite"

    # ------------------------------------------------------------------
    # 节点：查询改写（不充分时）
    # ------------------------------------------------------------------
    async def _node_rewrite(self, state: RAGState) -> RAGState:
        t0 = time.perf_counter()
        current = sorted(state["accumulated"], key=lambda x: x.score, reverse=True)
        new_query = await self._rewrite_query(state["query"], current[:5], state["round_num"])
        if new_query and new_query != state["query"]:
            state["rewritten_history"].append(new_query)
            state["search_queries"] = [new_query]
        state["round_num"] += 1
        state["trace"].append({
            "step": "查询改写",
            "rewritten": new_query,
            "ms": round((time.perf_counter() - t0) * 1000, 1),
        })
        return state

    # ------------------------------------------------------------------
    # 节点：最终化 — 冲突消解 + 权威性回填 + 精排
    # ------------------------------------------------------------------
    async def _node_finalize(self, state: RAGState) -> RAGState:
        t0 = time.perf_counter()
        all_list = sorted(state["accumulated"], key=lambda x: x.score, reverse=True)

        self._enrich_authority(all_list)
        conflicts = await self.conflict_resolver.detect(all_list[:10])

        candidates = state["best_chunks"] or all_list[: state["top_k"]]
        reranked = await state["reranker"].rerank(state["query"], candidates, top_k=state["top_k"])

        warning = None
        if state["best_score"] < 0.5:
            warning = "检索结果置信度较低，以下为当前最佳匹配"

        state["result"] = AgenticResult(
            chunks=reranked,
            confidence=state["best_score"],
            rounds_used=state["round_num"],
            intent=state["intent"],
            rewritten_queries=state["rewritten_history"],
            conflicts=conflicts,
            warning=warning,
            trace=state["trace"] + [{
                "step": "冲突检测+精排",
                "conflicts": len(conflicts),
                "final_chunks": len(reranked),
                "ms": round((time.perf_counter() - t0) * 1000, 1),
            }],
        )
        return state

    # ------------------------------------------------------------------
    # 主入口
    # ------------------------------------------------------------------
    async def retrieve(
        self,
        query: str,
        qdrant_retriever,
        es_retriever,
        reranker,
        top_k: int = 10,
        filter_groups: list[str] | None = None,
    ) -> AgenticResult:
        initial: RAGState = {
            "query": query,
            "intent": "",
            "retrieval_strategy": "semantic",
            "confidence": 0.0,
            "search_queries": [query],
            "round_num": 1,
            "accumulated": [],
            "round_new": 0,
            "best_score": 0.0,
            "best_chunks": [],
            "rewritten_history": [query],
            "sufficiency": 0.0,
            "trace": [],
            "result": None,
            # 请求级状态注入 state，规避共享单例竞态
            "top_k": top_k,
            "filter_groups": filter_groups or ["*"],
            "qdrant": qdrant_retriever,
            "es": es_retriever,
            "reranker": reranker,
        }
        final = await self.graph.ainvoke(initial)
        return final["result"]

    # ------------------------------------------------------------------
    # LLM 充分性判断 / 查询改写
    # ------------------------------------------------------------------
    async def _judge_sufficiency(self, query: str, chunks: list[SearchResult]) -> float:
        if not chunks:
            return 0.0
        llm = await self._get_llm()
        if llm is None:
            avg = sum(c.score for c in chunks[:10]) / min(len(chunks), 10)
            return min(avg * 3, 1.0)
        snippets = "\n---\n".join(f"[{i+1}] {c.content[:300]}" for i, c in enumerate(chunks[:10]))
        prompt = f"""判断检索结果是否足以回答用户问题，只输出 0.0-1.0 分数。
1.0=完全覆盖；0.8=基本充分；0.5=部分覆盖；0.2=几乎无关。

用户问题: {query}
检索结果:
{snippets}

充分性分数 (0.0-1.0):"""
        try:
            resp = await llm.ainvoke(prompt)
            text = resp.content if hasattr(resp, "content") else str(resp)
            m = re.search(r"(\d+\.?\d*)", text)
            if m:
                return min(max(float(m.group(1)), 0.0), 1.0)
        except Exception:
            pass
        avg = sum(c.score for c in chunks[:10]) / min(len(chunks), 10)
        return min(avg * 3, 1.0)

    async def _rewrite_query(self, query: str, chunks: list[SearchResult], round_num: int) -> str:
        llm = await self._get_llm()
        if llm is None:
            return query
        strategies = [
            "同义词替换、上下位词泛化/特化",
            "提取已检索到的实体扩充查询",
            "从检索结果中发现的线索（如'参见文档B'）追加检索",
        ]
        snippets = "\n".join(c.content[:150] for c in chunks[:5])

        # 改写结果缓存：同一查询 + 轮次 + 已知上下文 → 复用，降低重复 LLM 调用
        from src.retrieval import retrieval_cache
        cache_key = f"{query}|r{round_num}|{hashlib.md5(snippets.encode('utf-8')).hexdigest()}"
        cached = await retrieval_cache.get(cache_key, ["*"], namespace="rewrite")
        if cached and cached.get("query"):
            return cached["query"]

        prompt = self._load_prompt("query_rewrite.md", {
            "query": query, "snippets": snippets, "round_num": round_num,
            "strategy": strategies[min(round_num - 1, len(strategies) - 1)],
        })
        try:
            resp = await llm.ainvoke(prompt)
            rewritten = (resp.content if hasattr(resp, "content") else str(resp)).strip()
            if rewritten:
                await retrieval_cache.set(
                    cache_key, {"query": rewritten}, ["*"], namespace="rewrite"
                )
            return rewritten
        except Exception:
            return query

    @staticmethod
    def _load_prompt(name: str, ctx: dict) -> str:
        from pathlib import Path
        path = Path(__file__).parent.parent.parent / "configs" / "prompts" / name
        if path.exists():
            try:
                from jinja2 import Template
                rendered = Template(path.read_text(encoding="utf-8")).render(**ctx)
                body = rendered.split("system: |", 1)[-1].strip()
                if body:
                    return body
            except Exception:
                pass
        return (
            f"当前检索结果不足以回答用户问题，请改写查询以获取更全面信息。\n\n"
            f"原始查询: {ctx.get('query', '')}\n已检索内容:\n{ctx.get('snippets', '')}\n\n"
            f"第{ctx.get('round_num', 1)}轮改写策略: {ctx.get('strategy', '')}\n"
            f"请只输出改写后的查询（1-2句话，不加前缀）:"
        )

    def _enrich_authority(self, chunks: list[SearchResult]):
        doc_ids = {c.doc_id for c in chunks}
        meta = store.get_documents_meta(list(doc_ids))
        for c in chunks:
            m = meta.get(c.doc_id)
            if m:
                c.version = m.get("version")
                c.department = m.get("department")
                c.updated_at = m.get("updated_at")
                c.authority_score = float(m.get("authority_score") or 0.5)
