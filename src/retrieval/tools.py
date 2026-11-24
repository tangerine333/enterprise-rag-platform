# ============================================================
# 检索工具注册表 — 把"硬编码检索分支"升级为可扩展的 Tool Registry
# 新增一种检索方式 = 注册一个 tool + 在 retrieval.yaml 里加一行，
# 主流程（agentic 状态图）无需改动。
# ============================================================
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Awaitable, Callable

from src.retrieval.models import SearchResult

# 统一工具签名：async (query, top_k, filter_groups, qdrant, es, **kwargs) -> list[SearchResult]
ToolFunc = Callable[..., Awaitable[list[SearchResult]]]


@dataclass
class RetrievalTool:
    name: str
    description: str
    func: ToolFunc


# ------------------------------------------------------------------
# 工具实现（依赖注入检索器，注册表本身无状态、可并发复用）
# ------------------------------------------------------------------
async def _keyword_search(query: str, top_k: int, filter_groups, qdrant, es,
                          structured: bool = False, **_):
    return await es.search(query, top_k=top_k, filter_groups=filter_groups, structured=structured)


async def _semantic_search(query: str, top_k: int, filter_groups, qdrant, es, **_):
    emb = await qdrant.encode([query])
    return await qdrant.search(emb[0], top_k=top_k, filter_groups=filter_groups)


async def _multi_query_search(query: str, top_k: int, filter_groups, qdrant, es,
                              extra_queries: list[str] | None = None, **_):
    """对比型：多查询并行语义召回。"""
    results: list[SearchResult] = []
    for q in [query] + (extra_queries or []):
        emb = await qdrant.encode([q])
        results.extend(await qdrant.search(emb[0], top_k=top_k, filter_groups=filter_groups))
    return results


async def _structured_search(query: str, top_k: int, filter_groups, qdrant, es, **_):
    """操作型：标题/步骤结构化过滤。"""
    return await es.search(query, top_k=top_k, filter_groups=filter_groups, structured=True)


# ------------------------------------------------------------------
# 注册表
# ------------------------------------------------------------------
class RetrievalToolRegistry:
    def __init__(self):
        self._tools: dict[str, RetrievalTool] = {}

    def register(self, tool: RetrievalTool):
        self._tools[tool.name] = tool

    def get(self, name: str) -> RetrievalTool | None:
        return self._tools.get(name)

    def names(self) -> list[str]:
        return list(self._tools.keys())

    def describe(self) -> list[dict]:
        """导出工具 Schema（供未来 LLM Planner 自主选工具读取）。"""
        return [{"name": t.name, "description": t.description} for t in self._tools.values()]


# 全局单例：注册内置检索工具
registry = RetrievalToolRegistry()
registry.register(RetrievalTool("keyword", "BM25 关键词检索，适合事实型精确匹配（合同编号/金额等）", _keyword_search))
registry.register(RetrievalTool("semantic", "BGE-M3 向量语义检索，适合概念型（定义/原理）", _semantic_search))
registry.register(RetrievalTool("multi_query", "多查询并行语义检索，适合对比型（A 与 B 的区别）", _multi_query_search))
registry.register(RetrievalTool("structured", "标题/步骤结构化过滤，适合操作型（如何/流程）", _structured_search))
