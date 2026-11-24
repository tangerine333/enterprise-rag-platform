"""API 集成测试 — 无外部服务（Qdrant/ES/Redis/LLM 全部降级）也能跑通。"""
import time
import uuid

from fastapi.testclient import TestClient

from src.api.main import app

client = TestClient(app)


def _upload_md():
    unique = uuid.uuid4().hex[:8]
    md = (f"# 退款政策 {unique}\n\n## 退款时效\n\n退款时效为15天。\n\n"
          f"## 退款条件\n\n| 条件 | 说明 | 时效 |\n| --- | --- | --- |\n"
          f"| 未使用 | 全额退款 | 3天 |\n| 已使用 | 按比例退款 | 15天 |\n")
    r = client.post(
        "/api/v1/documents/upload",
        files={"file": (f"policy_{unique}.md", md.encode("utf-8"), "text/markdown")},
        data={"version": "v3.0", "department": "客服部"},
    )
    assert r.status_code == 200
    return r.json()


def _poll_status(doc_id: int, timeout: float = 10.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        r = client.get(f"/api/v1/documents/{doc_id}/status")
        assert r.status_code == 200
        if r.json()["parse_status"] in ("completed", "failed"):
            return r.json()
        time.sleep(0.1)
    raise TimeoutError("文档解析超时")


def test_health():
    r = client.get("/api/v1/health")
    assert r.status_code == 200
    assert r.json()["status"] == "healthy"


def test_upload_parse_and_index():
    j = _upload_md()
    assert j["id"] > 0
    status = _poll_status(j["id"])
    assert status["parse_status"] == "completed"
    assert status["chunk_count"] >= 5  # children + parents + table_rows + table_summary


def test_search():
    r = client.post("/api/v1/search/search", json={"query": "退款时效是几天？"})
    assert r.status_code == 200
    j = r.json()
    assert j["intent"] in ("factual", "conceptual", "comparative", "procedural")
    assert "retrieval_rounds" in j
    assert "trace" in j


def test_answer():
    r = client.post("/api/v1/search/answer", json={"query": "退款时效是几天？"})
    assert r.status_code == 200
    assert "answer" in r.json()


def test_stats():
    r = client.get("/api/v1/system/stats")
    assert r.status_code == 200
    assert "total_documents" in r.json()


def test_metrics():
    r = client.get("/metrics")
    assert r.status_code == 200
    assert b"rag_query_total" in r.content


def test_answer_metrics_incremented():
    """回归：/answer 曾把指标埋点写在 return 之后，导致永远不可达。"""
    from prometheus_client import REGISTRY
    labels = {"endpoint": "answer"}
    before = REGISTRY.get_sample_value("rag_query_total", labels) or 0.0

    r = client.post("/api/v1/search/answer", json={"query": "answer 指标埋点回归测试"})
    assert r.status_code == 200

    after = REGISTRY.get_sample_value("rag_query_total", labels) or 0.0
    assert after == before + 1


def test_stats_reports_pending_index_tasks():
    r = client.get("/api/v1/system/stats")
    assert r.status_code == 200
    assert "pending_index_tasks" in r.json()


def test_document_list():
    r = client.get("/api/v1/documents/")
    assert r.status_code == 200
    assert "items" in r.json() and "total" in r.json()


def test_404_returns_json():
    r = client.get("/api/v1/documents/999999")
    assert r.status_code == 404
    assert "detail" in r.json()
