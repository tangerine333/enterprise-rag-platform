"""冲突消解：版本/部门/时间/反馈的权威性加权。"""
from src.retrieval.conflict import (
    compute_authority,
    department_level,
    parse_version,
    recency_score,
)
from src.retrieval.models import SearchResult


def _doc(doc_id, version=None, department=None, updated_at=None, authority_score=0.5):
    return SearchResult(chunk_id=1, doc_id=doc_id, content="x", score=0.9,
                        version=version, department=department,
                        updated_at=updated_at, authority_score=authority_score)


def test_parse_version():
    assert parse_version("v2.0") == 2.0
    assert parse_version("3.2") == 3.2
    assert parse_version("2024.1") == 2024.1
    assert parse_version(None) == 0.0
    assert parse_version("无版本") == 0.0


def test_department_level():
    assert department_level("总部") == 3
    assert department_level("部门") == 2
    assert department_level("小组") == 1
    assert department_level(None) == 1  # 默认层级


def test_newer_version_wins():
    old = _doc(1, version="v1.0", department="客服部", updated_at="2023-01-01")
    new = _doc(2, version="v3.0", department="客服部", updated_at="2023-01-01")
    assert compute_authority(new)["authority_score"] > compute_authority(old)["authority_score"]


def test_higher_department_wins():
    low = _doc(1, version="v1.0", department="小组", updated_at="2024-01-01")
    high = _doc(2, version="v1.0", department="总部", updated_at="2024-01-01")
    assert compute_authority(high)["authority_score"] > compute_authority(low)["authority_score"]


def test_feedback_weight_affects_score():
    disliked = _doc(1, version="v3.0", department="总部", authority_score=0.1)
    liked = _doc(2, version="v3.0", department="总部", authority_score=0.9)
    assert compute_authority(liked)["authority_score"] > compute_authority(disliked)["authority_score"]


def test_recency_bounds():
    assert 0.0 <= recency_score("2024-06-01") <= 1.0
    assert recency_score(None) == 0.5
