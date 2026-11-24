"""RRF 融合 + 去重。"""
from src.retrieval.fusion import dedupe, reciprocal_rank_fusion
from src.retrieval.models import SearchResult


def _r(doc_id, chunk_id, score, source="qdrant", chunk_type="child"):
    return SearchResult(chunk_id=chunk_id, doc_id=doc_id, content=f"d{doc_id}", score=score, source=source, chunk_type=chunk_type)


def test_rrf_merges_rankings():
    # 两个召回列表，分数尺度不同（ES BM25 vs Qdrant cosine）
    es = [_r(1, 1, 30.0, source="elasticsearch"), _r(2, 2, 20.0, source="elasticsearch")]
    qd = [_r(2, 2, 0.9, source="qdrant"), _r(3, 3, 0.8, source="qdrant")]

    fused = reciprocal_rank_fusion([es, qd])
    ids = [(r.doc_id, r.chunk_id) for r in fused]
    # 同 key 多源命中应去重
    assert len(ids) == 3
    assert (2, 2) in ids  # 双方都命中的排前
    # 多源命中的排在首位
    assert ids[0] == (2, 2)


def test_rrf_empty():
    assert reciprocal_rank_fusion([]) == []
    assert reciprocal_rank_fusion([[], []]) == []


def test_dedupe_by_key():
    items = [_r(1, 1, 0.9), _r(1, 1, 0.8), _r(1, 2, 0.7)]
    out = dedupe(items)
    assert len(out) == 2


def test_dedupe_respects_chunk_type():
    # 同 doc/chunk 但不同类型（child vs table_row）不应互相去重
    a = _r(1, 1, 0.9, chunk_type="child")
    b = _r(1, 1, 0.8, chunk_type="table_row")
    out = dedupe([a, b])
    assert len(out) == 2
