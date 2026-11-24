# ============================================================
# BGE-Reranker Cross-Encoder 精排 + MMR 多样性去重
# 模型加载/推理是 CPU 密集，丢 asyncio.to_thread。
# ============================================================
from __future__ import annotations

import asyncio

from src.core.config import get_settings
from src.retrieval.models import SearchResult


class Reranker:
    """BGE-Reranker-v2-m3 Cross-Encoder 精排。"""

    def __init__(self):
        self.settings = get_settings()
        self._model = None
        self._load_attempted = False

    async def _lazy_load(self):
        """首次调用时加载模型（丢线程池）。"""
        if self._load_attempted:
            return
        self._load_attempted = True

        def _load():
            try:
                from sentence_transformers import CrossEncoder
                return CrossEncoder(self.settings.reranker_model, device=self.settings.reranker_device)
            except Exception:
                return None

        self._model = await asyncio.to_thread(_load)

    async def rerank(self, query: str, candidates: list[SearchResult],
                     top_k: int = 10) -> list[SearchResult]:
        if not candidates:
            return []

        await self._lazy_load()

        # Cross-Encoder 精排（CPU 密集，丢线程池）
        if self._model is not None:
            def _predict():
                pairs = [[query, c.content[:512]] for c in candidates]
                return self._model.predict(pairs)
            try:
                scores = await asyncio.to_thread(_predict)
                for c, s in zip(candidates, scores):
                    c.score = float(s)
            except Exception:
                pass  # 精排失败降级为原分数排序

        candidates.sort(key=lambda x: x.score, reverse=True)

        # MMR 多样性去重
        selected = [candidates[0]]
        remaining = candidates[1:]
        while len(selected) < top_k and remaining:
            best, best_mmr = None, float("-inf")
            for c in remaining:
                relevance = c.score
                max_dup = max(self._content_overlap(c.content, s.content) for s in selected)
                mmr = 0.7 * relevance - 0.3 * max_dup
                if mmr > best_mmr:
                    best_mmr, best = mmr, c
            if best is not None:
                selected.append(best)
                remaining.remove(best)
            else:
                break

        return selected[:top_k]

    @staticmethod
    def _content_overlap(a: str, b: str) -> float:
        a_set = set(a[:300])
        b_set = set(b[:300])
        if not a_set or not b_set:
            return 0.0
        return len(a_set & b_set) / len(a_set | b_set)
