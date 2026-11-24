"""评测口径：要点覆盖、分类子集、badcase 归因。"""
from eval.generate_golden_set import _category, _key_points
from eval.run_eval import (
    answer_completeness,
    attribute_failures,
    category_breakdown,
    point_covered,
)


def test_key_points_split():
    assert _key_points("旧版7天，新版15天，以新版为准") == ["旧版7天", "新版15天", "以新版为准"]
    assert _key_points("10000") == ["10000"]


def test_category_assignment():
    assert _category("comparative", "旧版7天，新版15天", [10, 11]) == "conflict"
    assert _category("factual", "10000", [1]) == "table"
    assert _category("conceptual", "分层图算法", [2]) == "general"


def test_point_covered():
    assert point_covered("15天", "退款时效为15天。")
    assert not point_covered("15天", "退款时效为30天。")
    assert point_covered("10000", "最大QPS为10000")


def test_answer_completeness():
    assert answer_completeness("旧版7天，新版15天", ["旧版7天", "新版15天"]) == 1.0
    assert answer_completeness("旧版7天", ["旧版7天", "新版15天"]) == 0.5


def test_category_breakdown():
    per_item = [
        {"category": "table", "hit": True, "redundancy": 0.1},
        {"category": "table", "hit": False, "redundancy": 0.3},
        {"category": "conflict", "hit": True, "redundancy": 0.0},
    ]
    out = category_breakdown(per_item)
    assert out["table"]["count"] == 2
    assert out["table"]["recall_at_10"] == 0.5
    assert abs(out["table"]["avg_redundancy"] - 0.2) < 1e-9


class _FakeConn:
    def __init__(self, total_chunks: int):
        self._total = total_chunks

    def execute(self, *args, **kwargs):
        return self

    def fetchone(self):
        return (self._total,)

    def close(self):
        pass


class _FakeStore:
    def __init__(self, total_chunks: int, doc_chunks: bool):
        self._total = total_chunks
        self._doc_chunks = doc_chunks

    def _get_conn(self):
        return _FakeConn(self._total)

    def get_chunks_by_doc(self, doc_id):
        return [{"id": 1}] if self._doc_chunks else []


def test_attribution_unavailable_without_knowledge_base(monkeypatch):
    import src.core.database as db
    monkeypatch.setattr(db, "store", _FakeStore(total_chunks=0, doc_chunks=False))

    out = attribute_failures([{"hit": False, "evidence": [1], "retrieved_docs": []}])

    assert out["available"] is False


def test_attribution_parse_missing(monkeypatch):
    import src.core.database as db
    monkeypatch.setattr(db, "store", _FakeStore(total_chunks=10, doc_chunks=False))

    out = attribute_failures([{"hit": False, "evidence": [1], "retrieved_docs": [2]}])

    assert out["available"] is True
    assert out["distribution"]["parse_missing"] == 1
