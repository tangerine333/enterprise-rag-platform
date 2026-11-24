# ============================================================
# Prometheus 监控指标 — 在线层可观测性
# 点踩率 / 点赞率 / 各端点延迟 / 查询量
# ============================================================
from __future__ import annotations

from prometheus_client import Counter, Gauge, Histogram

QUERY_TOTAL = Counter(
    "rag_query_total",
    "问答请求总数",
    ["endpoint"],  # search | answer
)

QUERY_LATENCY = Histogram(
    "rag_query_latency_seconds",
    "问答请求耗时（秒）",
    ["endpoint"],
    buckets=[0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0],
)

FEEDBACK_TOTAL = Counter(
    "rag_feedback_total",
    "用户反馈总数",
    ["feedback"],  # positive | negative
)

DOCUMENT_TOTAL = Counter(
    "rag_document_total",
    "文档上传总数",
    ["status"],  # completed | failed
)

INDEX_SYNC_PENDING = Gauge(
    "rag_index_sync_pending",
    "待补偿的索引同步任务数（Qdrant/ES 删除失败后落库）",
)
