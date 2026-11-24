# ============================================================
# 多路召回：Qdrant（向量语义）+ Elasticsearch（BM25 关键词）
# 同步客户端 + CPU 密集的 embedding 全部丢 asyncio.to_thread，
# 保证 FastAPI 事件循环不被阻塞（真异步）。
# ============================================================
from __future__ import annotations

import asyncio
import uuid

from src.core.config import get_settings
from src.retrieval.models import SearchResult


class QdrantRetriever:
    """BGE-M3 向量 → Qdrant 语义检索。"""

    def __init__(self, embedding_model=None):
        self._model = embedding_model
        self.settings = get_settings()
        self._client = None

    @property
    def client(self):
        if self._client is None:
            from qdrant_client import QdrantClient
            self._client = QdrantClient(url=self.settings.qdrant_url)
        return self._client

    async def encode(self, texts: list[str]):
        """BGE-M3 编码（CPU 密集，丢线程池）。"""
        return await asyncio.to_thread(self._encode_sync, texts)

    def _encode_sync(self, texts: list[str]):
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
                self._model = SentenceTransformer(
                    self.settings.embedding_model,
                    device=self.settings.embedding_device,
                )
            except Exception:
                import numpy as np
                return np.zeros((len(texts), self.settings.qdrant_vector_size), dtype="float32")
        try:
            return self._model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
        except Exception:
            import numpy as np
            return np.zeros((len(texts), self.settings.qdrant_vector_size), dtype="float32")

    async def search(self, query_vector, top_k: int = 100,
                     filter_groups: list[str] | None = None) -> list[SearchResult]:
        """Qdrant 向量检索（同步 IO 丢线程池）。"""
        return await asyncio.to_thread(self._search_sync, query_vector, top_k, filter_groups)

    def _search_sync(self, query_vector, top_k: int, filter_groups) -> list[SearchResult]:
        try:
            from qdrant_client.models import FieldCondition, Filter, MatchAny

            kwargs = {}
            if filter_groups and "*" not in filter_groups:
                kwargs["query_filter"] = Filter(
                    must=[FieldCondition(key="permission_groups", match=MatchAny(any=filter_groups))]
                )

            results = self.client.search(
                collection_name=self.settings.qdrant_collection,
                query_vector=list(query_vector),
                limit=top_k,
                with_payload=True,
                **kwargs,
            )
        except Exception:
            return []  # Qdrant 不可用时降级

        return [
            SearchResult(
                chunk_id=int(r.payload.get("chunk_id", r.id)),
                doc_id=int(r.payload.get("doc_id", 0)),
                content=r.payload.get("content", ""),
                score=float(r.score),
                heading=r.payload.get("heading"),
                source="qdrant",
                chunk_type=r.payload.get("chunk_type", "child"),
                table_summary=r.payload.get("table_summary"),
            )
            for r in results
        ]

    async def index_chunks(self, chunks: list[dict], embeddings) -> list[str]:
        """批量写入 Qdrant（丢线程池）。"""
        return await asyncio.to_thread(self._index_sync, chunks, embeddings)

    def _index_sync(self, chunks: list[dict], embeddings) -> list[str]:
        try:
            from qdrant_client.models import Distance, PointStruct, VectorParams
        except Exception:
            return []  # qdrant-client 未安装

        try:
            self.client.create_collection(
                collection_name=self.settings.qdrant_collection,
                vectors_config=VectorParams(
                    size=self.settings.qdrant_vector_size,
                    distance=Distance.COSINE,
                ),
            )
        except Exception:
            pass  # collection 已存在

        point_ids = []
        points = []
        for i, (chunk, emb) in enumerate(zip(chunks, embeddings)):
            pid = str(uuid.uuid4())
            point_ids.append(pid)
            points.append(PointStruct(
                id=pid,
                vector=[float(x) for x in emb],
                payload={
                    "doc_id": chunk.get("doc_id", 0),
                    "chunk_id": chunk.get("chunk_id", i),
                    "content": chunk.get("content", ""),
                    "heading": chunk.get("heading", ""),
                    "chunk_type": chunk.get("chunk_type", "child"),
                    "table_summary": chunk.get("table_summary", ""),
                    "permission_groups": chunk.get("permission_groups", "*"),
                },
            ))

        try:
            self.client.upsert(collection_name=self.settings.qdrant_collection, points=points)
        except Exception:
            pass  # Qdrant 不可用时跳过
        return point_ids


class ESRetriever:
    """Elasticsearch BM25 关键词检索。"""

    def __init__(self):
        self.settings = get_settings()
        self._client = None

    @property
    def client(self):
        if self._client is None:
            from elasticsearch import Elasticsearch
            self._client = Elasticsearch(
                f"http://{self.settings.es_host}:{self.settings.es_port}"
            )
        return self._client

    async def search(self, query: str, top_k: int = 50,
                     filter_groups: list[str] | None = None,
                     structured: bool = False) -> list[SearchResult]:
        """ES BM25 全文检索（丢线程池）。structured=True 时标题/步骤加权。"""
        return await asyncio.to_thread(self._search_sync, query, top_k, filter_groups, structured)

    def _search_sync(self, query: str, top_k: int, filter_groups, structured: bool) -> list[SearchResult]:
        fields = ["content", "heading^3", "table_structured^2"]
        if structured:
            fields = ["heading^4", "content", "table_structured"]

        body = {
            "query": {
                "multi_match": {
                    "query": query,
                    "fields": fields,
                    "type": "best_fields",
                }
            },
            "size": top_k,
        }
        if filter_groups and "*" not in filter_groups:
            body["query"] = {
                "bool": {
                    "must": [body["query"]],
                    "filter": [{"terms": {"permission_groups": filter_groups}}],
                }
            }

        try:
            resp = self.client.search(index=self.settings.es_index, body=body)
        except Exception:
            return []  # ES 不可用时降级

        results = []
        for hit in resp["hits"]["hits"]:
            src = hit["_source"]
            results.append(SearchResult(
                chunk_id=int(hit["_id"]) if str(hit["_id"]).isdigit() else 0,
                doc_id=int(src.get("doc_id", 0)),
                content=src.get("content", ""),
                score=float(hit["_score"] or 0.0),
                heading=src.get("heading"),
                source="elasticsearch",
                chunk_type=src.get("chunk_type", "child"),
                table_summary=src.get("table_summary"),
            ))
        return results

    async def index_chunks(self, chunks: list[dict]):
        """批量写入 ES（丢线程池）。"""
        await asyncio.to_thread(self._index_sync, chunks)

    def _index_sync(self, chunks: list[dict]):
        try:
            from elasticsearch.helpers import bulk
        except Exception:
            return  # elasticsearch 未安装

        actions = []
        for i, chunk in enumerate(chunks):
            actions.append({
                "_index": self.settings.es_index,
                "_id": str(chunk.get("chunk_id", i)),
                "_source": {
                    "doc_id": chunk.get("doc_id", 0),
                    "content": chunk.get("content", ""),
                    "heading": chunk.get("heading", ""),
                    "chunk_type": chunk.get("chunk_type", "child"),
                    "table_structured": chunk.get("metadata_json", "{}"),
                    "table_summary": chunk.get("table_summary", ""),
                    "permission_groups": chunk.get("permission_groups", "*"),
                },
            })

        try:
            bulk(self.client, actions)
        except Exception:
            pass  # ES 不可用时跳过
