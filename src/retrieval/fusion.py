# ============================================================
# 多路召回融合 — Reciprocal Rank Fusion (RRF)
# 修复：之前直接按 score 排序混排 ES BM25 与 Qdrant cosine
# （两者分数尺度不同，不可直接比较）。RRF 只关心排名，天然尺度无关。
# ============================================================
from __future__ import annotations

from src.retrieval.models import SearchResult


def reciprocal_rank_fusion(
    result_sets: list[list[SearchResult]], k: int = 60
) -> list[SearchResult]:
    """RRF 融合多个召回列表（每个列表内部按分数降序已排好）。"""
    if not result_sets:
        return []

    fused: dict[tuple[int, int, str], tuple[SearchResult, float]] = {}

    for lst in result_sets:
        # 列表内按分数降序（此时分数只在同源内可比）
        ranked = sorted(lst, key=lambda x: x.score, reverse=True)
        for rank, item in enumerate(ranked, start=1):
            key = (item.doc_id, item.chunk_id, item.chunk_type)
            rrf_score = 1.0 / (k + rank)
            if key in fused:
                prev_item, prev_score = fused[key]
                # 同 key 多源命中，累加 RRF 分数，保留高分来源
                if item.score > prev_item.score:
                    fused[key] = (item, prev_score + rrf_score)
                else:
                    fused[key] = (prev_item, prev_score + rrf_score)
            else:
                fused[key] = (item, rrf_score)

    # 按融合分数降序
    merged = [item for item, _ in fused.values()]
    merged.sort(key=lambda x: fused[(x.doc_id, x.chunk_id, x.chunk_type)][1], reverse=True)
    return merged


def dedupe(results: list[SearchResult]) -> list[SearchResult]:
    """按 (doc_id, chunk_id, chunk_type) 去重，保留首见。"""
    seen: set[tuple[int, int, str]] = set()
    out = []
    for r in results:
        key = (r.doc_id, r.chunk_id, r.chunk_type)
        if key not in seen:
            seen.add(key)
            out.append(r)
    return out
