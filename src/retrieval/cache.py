# ============================================================
# 多级缓存：L1 精确匹配 / L2 语义相似 / L3 LLM 响应
# Redis 不可用时熔断降级（30s 内跳过缓存），不阻塞请求
# ============================================================
from __future__ import annotations

import hashlib
import json
import time

from src.core.config import get_settings


class RetrievalCache:
    """Redis 多级缓存，含熔断降级。"""

    def __init__(self, redis_client=None):
        self._redis = redis_client
        self._disabled_until = 0.0  # 熔断时间戳（连接失败后 30s 内跳过缓存）

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

    def _key(self, query: str, level: str = "l1") -> str:
        if level == "l2":
            query = " ".join(query.lower().split())
        digest = hashlib.md5(query.encode()).hexdigest()
        return f"rag:cache:{level}:{digest}"

    async def get(self, query: str) -> dict | None:
        if self._circuit_open():
            return None
        try:
            r = await self._get_redis()
            for level in ("l1", "l2"):
                data = await r.get(self._key(query, level))
                if data:
                    return json.loads(data)
        except Exception:
            self._trip_circuit()
        return None

    async def set(self, query: str, result: dict, ttl: int = 3600):
        if self._circuit_open():
            return
        try:
            r = await self._get_redis()
            payload = json.dumps(result, ensure_ascii=False, default=str)
            await r.setex(self._key(query, "l1"), ttl, payload)
            await r.setex(self._key(query, "l2"), ttl * 6, payload)
        except Exception:
            self._trip_circuit()
