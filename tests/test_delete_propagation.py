"""删除/更新传播：Qdrant/ES 清理、补偿任务与 reconcile。"""
import uuid

import pytest
from fastapi.testclient import TestClient

from src.api.main import app
from src.core.database import compute_checksum, store
from src.retrieval import es_retriever, qdrant_retriever
from src.retrieval.retrievers import ESRetriever, QdrantRetriever

client = TestClient(app)


@pytest.fixture(autouse=True)
def clean_sync_tasks():
    conn = store._get_conn()
    conn.execute("DELETE FROM index_sync_tasks")
    conn.commit()
    conn.close()
    yield


def _upload_md(text: str) -> int:
    unique = uuid.uuid4().hex[:8]
    r = client.post(
        "/api/v1/documents/upload",
        files={"file": (f"prop_{unique}.md", text.encode("utf-8"), "text/markdown")},
    )
    assert r.status_code == 200
    return r.json()["id"]


async def _boom(doc_id: int):
    raise RuntimeError("service down")


async def _ok(doc_id: int):
    return None


# ------------------------------------------------------------------
# 检索器层：确认删除调用带上了 doc_id 过滤条件
# ------------------------------------------------------------------
def test_qdrant_delete_by_doc_filters_doc_id():
    pytest.importorskip("qdrant_client")

    class FakeClient:
        def __init__(self):
            self.calls = []

        def delete(self, **kwargs):
            self.calls.append(kwargs)

    retriever = QdrantRetriever()
    fake = FakeClient()
    retriever._client = fake

    retriever._delete_by_doc_sync(42)

    assert fake.calls
    selector = fake.calls[0]["points_selector"]
    cond = selector.filter.must[0]
    assert cond.key == "doc_id"
    assert cond.match.value == 42


def test_es_delete_by_doc_uses_term_query():
    class FakeClient:
        def __init__(self):
            self.calls = []

        def delete_by_query(self, **kwargs):
            self.calls.append(kwargs)

    retriever = ESRetriever()
    fake = FakeClient()
    retriever._client = fake

    retriever._delete_by_doc_sync(42)

    assert fake.calls
    assert fake.calls[0]["query"] == {"term": {"doc_id": 42}}
    assert fake.calls[0]["refresh"] is True


# ------------------------------------------------------------------
# API 层：删除 → 元数据/切片删除 + 失败补偿
# ------------------------------------------------------------------
def test_delete_creates_compensation_tasks(monkeypatch):
    monkeypatch.setattr(qdrant_retriever, "delete_by_doc", _boom)
    monkeypatch.setattr(es_retriever, "delete_by_doc", _boom)

    doc_id = _upload_md("# 删除测试\n\n这段内容用于删除传播测试。\n")
    assert store.get_chunks_by_doc(doc_id)

    r = client.delete(f"/api/v1/documents/{doc_id}")

    assert r.status_code == 200
    body = r.json()
    assert body["index_cleanup"] == "pending"
    assert set(body["pending_targets"]) == {"qdrant", "es"}
    assert store.get_document(doc_id) is None
    assert store.get_chunks_by_doc(doc_id) == []

    tasks = [t for t in store.list_pending_sync_tasks(1000) if t["doc_id"] == doc_id]
    assert {t["target"] for t in tasks} == {"qdrant", "es"}
    assert store.count_pending_sync_tasks() >= 2


def test_reconcile_clears_pending_tasks(monkeypatch):
    monkeypatch.setattr(qdrant_retriever, "delete_by_doc", _boom)
    monkeypatch.setattr(es_retriever, "delete_by_doc", _boom)

    doc_id = _upload_md("# 重试测试\n\n补偿任务重试内容。\n")
    client.delete(f"/api/v1/documents/{doc_id}")
    assert store.count_pending_sync_tasks() >= 2

    monkeypatch.setattr(qdrant_retriever, "delete_by_doc", _ok)
    monkeypatch.setattr(es_retriever, "delete_by_doc", _ok)

    r = client.post("/api/v1/system/reconcile")

    assert r.status_code == 200
    body = r.json()
    assert body["failed"] == 0
    assert body["done"] >= 2
    assert store.count_pending_sync_tasks() == 0


# ------------------------------------------------------------------
# 更新：清理旧索引 → 重建切片
# ------------------------------------------------------------------
def test_replace_document_rebuilds_chunks(monkeypatch):
    monkeypatch.setattr(qdrant_retriever, "delete_by_doc", _ok)
    monkeypatch.setattr(es_retriever, "delete_by_doc", _ok)

    doc_id = _upload_md("# 旧标题\n\n这是旧内容。\n")
    new_text = "# 新标题\n\n这是更新后的内容。\n"

    r = client.put(
        f"/api/v1/documents/{doc_id}",
        files={"file": ("new.md", new_text.encode("utf-8"), "text/markdown")},
        data={"version": "v2.0", "department": "技术部"},
    )

    assert r.status_code == 200
    doc = store.get_document(doc_id)
    assert doc["parse_status"] == "completed"
    assert doc["version"] == "v2.0"

    joined = "\n".join(c["content"] for c in store.get_chunks_by_doc(doc_id))
    assert "更新后的内容" in joined
    assert "旧内容" not in joined


def test_replace_aborts_when_cleanup_fails(monkeypatch):
    monkeypatch.setattr(qdrant_retriever, "delete_by_doc", _ok)
    monkeypatch.setattr(es_retriever, "delete_by_doc", _ok)

    doc_id = _upload_md("# 原始标题\n\n这是原始内容。\n")
    before = "\n".join(c["content"] for c in store.get_chunks_by_doc(doc_id))

    monkeypatch.setattr(qdrant_retriever, "delete_by_doc", _boom)
    monkeypatch.setattr(es_retriever, "delete_by_doc", _boom)

    new_text = "# 新标题\n\n这是新内容，不应写入。\n"
    r = client.put(
        f"/api/v1/documents/{doc_id}",
        files={"file": ("new.md", new_text.encode("utf-8"), "text/markdown")},
    )

    assert r.status_code == 503
    assert "reconcile" in r.json()["detail"]
    doc = store.get_document(doc_id)
    assert doc["checksum"] != compute_checksum(new_text.encode("utf-8"))
    after = "\n".join(c["content"] for c in store.get_chunks_by_doc(doc_id))
    assert "这是原始内容" in after
    assert "这是新内容" not in before and "这是新内容" not in after
