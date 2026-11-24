# ============================================================
# 多级缓存：L1 精确匹配 / L2 归一化查询
# 命名空间：search / answer / rewrite，各自独立
# key 包含权限组与索引版本，避免跨权限泄漏与文档变更后的陈旧结果
# Redis 不可用时熔断降级（30s 内跳过缓存），不阻塞请求
# ============================================================
from __future__ import annotations

import hashlib
import json
import time

from src.core.config import get_settings


class RetrievalCache:
    """Redis 多级缓存，含权限隔离、版本失效与熔断降级。"""

    def __init__(self, redis_client=None):
        self._redis = redis_client
        self._disabled_until = 0.0  # 熔断时间戳（连接失败后 30s 内跳过缓存）
        self._local_version = 0

    async def _get_redis(self):
        if self._redis is None:
            import redis.asyncio as aioredis
            settings = get_settings()
            self._redis = aioredis.from_url(
                settings.redis_url,
                socket_connect_timeout=0.5,
                socket_timeout=0.5,
                retry_on_timeout=False,
            )
        return self._redis

    def _circuit_open(self) -> bool:
        return time.time() < self._disabled_until

    def _trip_circuit(self):
        self._disabled_until = time.time() + 30

    # ------------------------------------------------------------------
    # 版本：文档增删改后 bump，使所有旧缓存 key 失效
    # ------------------------------------------------------------------
    async def _index_version(self) -> str:
        try:
            r = await self._get_redis()
            value = await r.get("rag:index:version")
            if value is not None:
                return value.decode() if isinstance(value, bytes) else str(value)
        except Exception:
            self._trip_circuit()
        return str(self._local_version)

    async def bump_version(self):
        """文档新增/更新/删除后调用，使旧检索与答案缓存整体失效。"""
        self._local_version += 1
        try:
            r = await self._get_redis()
            await r.incr("rag:index:version")
        except Exception:
            self._trip_circuit()

    # ------------------------------------------------------------------
    # key：命名空间 + 索引版本 + 权限组 + 查询
    # ------------------------------------------------------------------
    @staticmethod
    def _groups_tag(filter_groups: list[str] | None) -> str:
        groups = sorted(filter_groups or ["*"])
        return hashlib.md5("|".join(groups).encode()).hexdigest()[:12]

    def _key(self, query: str, namespace: str, filter_groups: list[str] | None,
             version: str, level: str) -> str:
        if level == "l2":
            query = " ".join(query.lower().split())
        digest = hashlib.md5(query.encode()).hexdigest()
        return (f"rag:cache:{namespace}:{version}:"
                f"{self._groups_tag(filter_groups)}:{level}:{digest}")

    async def get(self, query: str, filter_groups: list[str] | None = None,
                  namespace: str = "search") -> dict | None:
        if self._circuit_open():
            return None
        try:
            r = await self._get_redis()
            version = await self._index_version()
            for level in ("l1", "l2"):
                data = await r.get(self._key(query, namespace, filter_groups, version, level))
                if data:
                    return json.loads(data)
        except Exception:
            self._trip_circuit()
        return None

    async def set(self, query: str, result: dict, filter_groups: list[str] | None = None,
                  ttl: int = 3600, namespace: str = "search"):
        if self._circuit_open():
            return
        try:
            r = await self._get_redis()
            version = await self._index_version()
            payload = json.dumps(result, ensure_ascii=False, default=str)
            await r.setex(self._key(query, namespace, filter_groups, version, "l1"), ttl, payload)
            await r.setex(self._key(query, namespace, filter_groups, version, "l2"), ttl * 6, payload)
        except Exception:
            self._trip_circuit()
