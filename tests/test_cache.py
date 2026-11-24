"""Redis 多级缓存：权限隔离、版本失效、命名空间与答案缓存。"""
import asyncio
import uuid

from fastapi.testclient import TestClient

from src.api.main import app
from src.retrieval import retrieval_cache
from src.retrieval.cache import RetrievalCache

client = TestClient(app)


class FakeRedis:
    """最小 Redis 替身：get / setex / incr。"""

    def __init__(self):
        self.data: dict[str, str] = {}

    async def get(self, key):
        value = self.data.get(key)
        return value.encode() if isinstance(value, str) else value

    async def setex(self, key, ttl, value):
        self.data[key] = value

    async def incr(self, key):
        self.data[key] = str(int(self.data.get(key, 0)) + 1)
        return int(self.data[key])


def test_cache_isolated_by_permission_groups():
    cache = RetrievalCache(FakeRedis())

    asyncio.run(cache.set("退款周期", {"answer": "15天"}, ["hr"]))
    assert asyncio.run(cache.get("退款周期", ["hr"])) == {"answer": "15天"}
    # 不同权限组不能命中同一缓存（防止越权泄漏）
    assert asyncio.run(cache.get("退款周期", ["public"])) is None


def test_cache_invalidated_by_version_bump():
    cache = RetrievalCache(FakeRedis())

    asyncio.run(cache.set("旧内容", {"v": 1}, ["*"]))
    assert asyncio.run(cache.get("旧内容", ["*"])) == {"v": 1}

    asyncio.run(cache.bump_version())

    # 文档变更后旧 key 不再命中
    assert asyncio.run(cache.get("旧内容", ["*"])) is None


def test_cache_l2_normalizes_query():
    cache = RetrievalCache(FakeRedis())

    asyncio.run(cache.set("Refund   Policy", {"v": 1}, ["*"]))
    # L2 归一化（大小写/空白）后应命中
    assert asyncio.run(cache.get("refund policy", ["*"])) == {"v": 1}


def test_cache_namespaces_isolated():
    cache = RetrievalCache(FakeRedis())

    asyncio.run(cache.set("同一查询", {"x": 1}, ["*"], namespace="search"))
    assert asyncio.run(cache.get("同一查询", ["*"], namespace="search")) == {"x": 1}
    assert asyncio.run(cache.get("同一查询", ["*"], namespace="answer")) is None


def test_answer_endpoint_uses_cache(monkeypatch):
    fake = FakeRedis()
    monkeypatch.setattr(retrieval_cache, "_redis", fake)
    monkeypatch.setattr(retrieval_cache, "_disabled_until", 0.0)

    query = f"缓存命中测试 {uuid.uuid4().hex[:8]}"
    first = client.post("/api/v1/search/answer", json={"query": query})
    assert first.status_code == 200
    assert first.json()["cached"] is False

    second = client.post("/api/v1/search/answer", json={"query": query})
    assert second.status_code == 200
    assert second.json()["cached"] is True
    assert second.json()["answer"] == first.json()["answer"]
