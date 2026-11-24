"""元数据存储：文档 CRUD、增量去重、反馈回流。"""
from src.core.database import compute_checksum


def test_insert_and_get_document(test_store):
    doc_id = test_store.insert_document(
        title="政策.md", file_type="md", version="v3.0",
        department="客服部", checksum="abc123",
    )
    doc = test_store.get_document(doc_id)
    assert doc["title"] == "政策.md"
    assert doc["version"] == "v3.0"
    assert doc["department"] == "客服部"
    assert doc["authority_score"] == 0.5


def test_find_by_checksum(test_store):
    test_store.insert_document(title="a.md", file_type="md", checksum="checksum-x")
    assert test_store.find_by_checksum("checksum-x") is not None
    assert test_store.find_by_checksum("not-exist") is None


def test_apply_feedback_updates_authority(test_store):
    doc_id = test_store.insert_document(title="b.md", file_type="md")
    test_store.apply_feedback(doc_id, "positive")
    assert test_store.get_document(doc_id)["authority_score"] > 0.5
    test_store.apply_feedback(doc_id, "negative")
    test_store.apply_feedback(doc_id, "negative")
    assert test_store.get_document(doc_id)["authority_score"] < 0.6


def test_apply_feedback_clamped(test_store):
    doc_id = test_store.insert_document(title="c.md", file_type="md")
    for _ in range(50):
        test_store.apply_feedback(doc_id, "positive")
    assert test_store.get_document(doc_id)["authority_score"] <= 1.0


def test_apply_feedback_uses_ema_and_logs(test_store):
    doc_id = test_store.insert_document(title="d.md", file_type="md")

    new_score = test_store.apply_feedback(doc_id, "positive", request_id="req_x")

    # α=0.9：0.5 + 0.1*(1.0-0.5) = 0.55
    assert abs(new_score - 0.55) < 1e-9
    log = test_store.feedback_history(doc_id)
    assert len(log) == 1
    assert log[0]["request_id"] == "req_x"
    assert abs(log[0]["old_score"] - 0.5) < 1e-9
    assert abs(log[0]["new_score"] - 0.55) < 1e-9


def test_get_documents_meta(test_store):
    id1 = test_store.insert_document(title="1.md", file_type="md", version="v1.0")
    id2 = test_store.insert_document(title="2.md", file_type="md", version="v2.0")
    meta = test_store.get_documents_meta([id1, id2])
    assert id1 in meta and id2 in meta
    assert meta[id1]["version"] == "v1.0"


def test_compute_checksum_deterministic():
    assert compute_checksum(b"hello") == compute_checksum(b"hello")
    assert compute_checksum(b"hello") != compute_checksum(b"world")
