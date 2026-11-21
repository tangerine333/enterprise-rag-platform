# ============================================================
# LLM 驱动的 Agentic RAG 检索引擎
# ============================================================
# 核心思路：让 LLM 在检索阶段实时判断充分性并动态调整策略
# 不搞"自研"模型 — 全部用成熟组件
# ============================================================
from __future__ import annotations

import hashlib
import json
import re
import time
import uuid
from dataclasses import dataclass, field
from typing import Callable

from src.core.config import get_settings, load_yaml_config


# ==================================================================
# 数据类
# ==================================================================
@dataclass
class SearchResult:
    chunk_id: int
    doc_id: int
    content: str
    score: float
    heading: str | None = None
    source: str = ""  # "qdrant" | "elasticsearch"
    table_summary: str | None = None


@dataclass
class AgenticResult:
    chunks: list[SearchResult]
    confidence: float
    rounds_used: int
    intent: str
    rewritten_queries: list[str]
    conflicts: list[dict] = field(default_factory=list)
    warning: str | None = None


# ==================================================================
# Qdrant 向量检索
# ==================================================================
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
        """BGE-M3 编码。"""
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(
                self.settings.embedding_model,
                device=self.settings.embedding_device,
            )
        embeddings = self._model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
        return embeddings

    async def search(self, query_vector, top_k: int = 100) -> list[SearchResult]:
        """Qdrant 向量检索。"""
        try:
            results = self.client.search(
                collection_name=self.settings.qdrant_collection,
                query_vector=query_vector.tolist(),
                limit=top_k,
                with_payload=True,
            )
        except Exception:
            return []  # Qdrant 不可用时降级

        return [
            SearchResult(
                chunk_id=r.id,
                doc_id=r.payload.get("doc_id", 0),
                content=r.payload.get("content", ""),
                score=r.score,
                heading=r.payload.get("heading"),
                source="qdrant",
                table_summary=r.payload.get("table_summary"),
            )
            for r in results
        ]

    def index_chunks(self, chunks: list[dict], embeddings) -> list[str]:
        """批量写入 Qdrant。"""
        from qdrant_client.models import Distance, PointStruct, VectorParams

        # 确保 collection 存在
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
                vector=emb.tolist(),
                payload={
                    "doc_id": chunk.get("doc_id", 0),
                    "chunk_id": chunk.get("id", i),
                    "content": chunk.get("content", ""),
                    "heading": chunk.get("heading", ""),
                    "chunk_type": chunk.get("chunk_type", "child"),
                    "table_summary": chunk.get("table_summary", ""),
                },
            ))

        self.client.upsert(collection_name=self.settings.qdrant_collection, points=points)
        return point_ids


# ==================================================================
# Elasticsearch 关键词检索
# ==================================================================
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

    async def search(self, query: str, top_k: int = 50) -> list[SearchResult]:
        """ES BM25 全文检索。"""
        try:
            resp = self.client.search(
                index=self.settings.es_index,
                body={
                    "query": {
                        "multi_match": {
                            "query": query,
                            "fields": ["content", "heading^2", "table_structured"],
                            "type": "best_fields",
                        }
                    },
                    "size": top_k,
                },
            )
        except Exception:
            return []  # ES 不可用时降级

        results = []
        for hit in resp["hits"]["hits"]:
            src = hit["_source"]
            results.append(SearchResult(
                chunk_id=int(hit["_id"]) if hit["_id"].isdigit() else 0,
                doc_id=src.get("doc_id", 0),
                content=src.get("content", ""),
                score=hit["_score"] or 0.0,
                heading=src.get("heading"),
                source="elasticsearch",
                table_summary=src.get("table_summary"),
            ))
        return results

    def index_chunks(self, chunks: list[dict]):
        """批量写入 ES。"""
        from elasticsearch.helpers import bulk

        actions = []
        for i, chunk in enumerate(chunks):
            actions.append({
                "_index": self.settings.es_index,
                "_id": str(chunk.get("id", i)),
                "_source": {
                    "doc_id": chunk.get("doc_id", 0),
                    "content": chunk.get("content", ""),
                    "heading": chunk.get("heading", ""),
                    "chunk_type": chunk.get("chunk_type", "child"),
                    "table_structured": chunk.get("metadata_json", "{}"),
                },
            })

        try:
            bulk(self.client, actions)
        except Exception:
            pass  # ES 不可用时跳过


# ==================================================================
# BGE-Reranker 重排序
# ==================================================================
class Reranker:
    """BGE-Reranker-v2-m3 Cross-Encoder 精排。"""

    def __init__(self):
        self.settings = get_settings()
        self._model = None

    async def rerank(self, query: str, candidates: list[SearchResult],
                     top_k: int = 10) -> list[SearchResult]:
        if not candidates:
            return []

        # 有模型则用 Cross-Encoder 精排
        if self._model is not None:
            pairs = [[query, c.content[:512]] for c in candidates]
            scores = self._model.compute_score(pairs)
            for c, s in zip(candidates, scores):
                c.score = float(s)
        # 否则直接按原始分数排序
        candidates.sort(key=lambda x: x.score, reverse=True)

        # MMR 多样性去重
        selected = [candidates[0]]
        remaining = candidates[1:]
        while len(selected) < top_k and remaining:
            best, best_mmr = None, float("-inf")
            for c in remaining:
                relevance = c.score
                max_dup = max(self._content_overlap(c.content, s.content) for s in selected)
                mmr = 0.7 * relevance - 0.3 * max_dup
                if mmr > best_mmr:
                    best_mmr, best = mmr, c
            if best:
                selected.append(best)
                remaining.remove(best)
            else:
                break

        return selected[:top_k]

    @staticmethod
    def _content_overlap(a: str, b: str) -> float:
        a_set = set(a[:300])
        b_set = set(b[:300])
        if not a_set or not b_set:
            return 0.0
        return len(a_set & b_set) / len(a_set | b_set)

    async def _lazy_load(self):
        if self._model is None:
            try:
                from sentence_transformers import CrossEncoder
                self._model = CrossEncoder(
                    self.settings.reranker_model,
                    device=self.settings.reranker_device,
                )
            except Exception:
                pass


# ==================================================================
# LLM 驱动的 Agentic RAG (LangGraph 编排)
# ==================================================================
class AgenticRAGEngine:
    """Agentic RAG 核心引擎。

    LLM 在检索阶段做三件事：
    1. 意图分类 + 查询改写（检索前）
    2. 检索充分性判断（检索后）
    3. 补充检索策略决策（不充分时）
    最多3轮，每轮新增文档净增量<20%时提前终止。
    """

    def __init__(self, llm=None):
        self.settings = get_settings()
        self.retrieval_config = load_yaml_config("retrieval.yaml")
        self.agentic_config = self.retrieval_config.get("agentic", {})
        self.max_rounds = self.agentic_config.get("max_retrieval_rounds", 3)
        self._llm = llm

    async def _get_llm(self):
        if self._llm is None:
            from langchain_openai import ChatOpenAI
            self._llm = ChatOpenAI(
                model=self.settings.llm_model,
                api_key=self.settings.openai_api_key,
                base_url=self.settings.openai_base_url,
                temperature=0.1,
            )
        return self._llm

    # ------------------------------------------------------------------
    # Step 0: 意图分类 + 初始检索策略选择 (LLM-driven)
    # ------------------------------------------------------------------
    async def classify_and_rewrite(self, query: str) -> dict:
        """LLM 进行意图分类 + 查询改写 + 检索策略选择。"""
        llm = await self._get_llm()

        prompt = f"""分析以下用户查询，输出JSON格式：

{{
  "intent": "factual|conceptual|comparative|procedural",
  "confidence": 0.0-1.0,
  "rewritten_queries": ["改写后的查询变体1", "变体2"],
  "keywords": ["关键词1", "关键词2"],
  "retrieval_strategy": "keyword_first|semantic_first|multi_query|hybrid",
  "reasoning": "简短说明分类理由"
}}

查询: {query}

JSON:"""

        try:
            resp = await llm.ainvoke(prompt)
            text = resp.content if hasattr(resp, "content") else str(resp)
            # 提取 JSON
            json_match = re.search(r'\{.*\}', text, re.DOTALL)
            if json_match:
                return json.loads(json_match.group())
        except Exception:
            pass

        # LLM不可用时的降级策略：规则分类
        return self._rule_based_classify(query)

    @staticmethod
    def _rule_based_classify(query: str) -> dict:
        """规则兜底（LLM 不可用时）。"""
        if re.search(r'(多少|金额|编号|几[个份张]|什么时候|谁|哪个|列出)', query):
            intent = "factual"
            strategy = "keyword_first"
        elif re.search(r'(什么是|定义|解释|含义|为什么|原理)', query):
            intent = "conceptual"
            strategy = "semantic_first"
        elif re.search(r'(对比|比较|区别|vs|和.*哪个)', query):
            intent = "comparative"
            strategy = "multi_query"
        elif re.search(r'(如何|怎么|步骤|流程|部署|安装|配置)', query):
            intent = "procedural"
            strategy = "hybrid"
        else:
            intent = "conceptual"
            strategy = "hybrid"

        return {
            "intent": intent,
            "confidence": 0.6,
            "rewritten_queries": [query],
            "keywords": re.findall(r'[一-鿿\w]{2,}', query),
            "retrieval_strategy": strategy,
        }

    # ------------------------------------------------------------------
    # Step 1: 检索充分性判断 (LLM-driven)
    # ------------------------------------------------------------------
    async def judge_sufficiency(self, query: str, chunks: list[SearchResult]) -> float:
        """LLM 判断检索结果是否足以回答问题。"""
        if not chunks:
            return 0.0

        llm = await self._get_llm()

        # 取 Top-10 给 LLM 判断
        snippets = "\n---\n".join(
            f"[{i+1}] {c.content[:300]}" for i, c in enumerate(chunks[:10])
        )

        prompt = f"""判断以下检索结果是否足以回答用户问题。只输出 0.0 到 1.0 的分数。

- 1.0: 检索结果完全覆盖了回答问题所需的全部信息
- 0.8: 基本充分，有少量细节缺失但不影响回答
- 0.5: 部分覆盖，缺少关键信息
- 0.2: 几乎没有相关信息

用户问题: {query}

检索结果:
{snippets}

充分性分数 (0.0-1.0):"""

        try:
            resp = await llm.ainvoke(prompt)
            text = resp.content if hasattr(resp, "content") else str(resp)
            match = re.search(r'(\d+\.?\d*)', text)
            if match:
                return min(max(float(match.group(1)), 0.0), 1.0)
        except Exception:
            pass

        # LLM不可用时的降级策略：基于检索分数的启发式评分
        if chunks:
            avg_score = sum(c.score for c in chunks[:10]) / min(len(chunks), 10)
            return min(avg_score * 3, 1.0)
        return 0.0

    # ------------------------------------------------------------------
    # Step 2: 查询改写 (LLM-driven)
    # ------------------------------------------------------------------
    async def rewrite_for_retrieval(self, query: str, current_chunks: list[SearchResult],
                                     round_num: int) -> str:
        """不充分时，LLM 改写查询以获取更好的检索结果。"""
        llm = await self._get_llm()

        strategies = ["同义词替换、上下位词泛化/特化", "提取已检索到的实体扩充查询",
                      "从检索结果中发现的线索（如'参见文档B'）追加检索"]

        # 已检索到的内容摘要
        snippets = "\n".join(c.content[:150] for c in current_chunks[:5])

        prompt = f"""当前检索结果不足以回答用户问题，请改写查询以获取更全面的信息。

原始查询: {query}
当前已检索到的内容:
{snippets}

第{round_num}轮改写策略: {strategies[min(round_num-1, len(strategies)-1)]}

请输出改写后的查询（1-2句话，不要加任何前缀说明）:"""

        try:
            resp = await llm.ainvoke(prompt)
            return (resp.content if hasattr(resp, "content") else str(resp)).strip()
        except Exception:
            pass

        return query

    # ------------------------------------------------------------------
    # Step 3: 冲突检测 (LLM-driven)
    # ------------------------------------------------------------------
    async def detect_conflicts(self, chunks: list[SearchResult]) -> list[dict]:
        """检测多文档间的信息冲突。"""
        if len(chunks) < 2:
            return []

        llm = await self._get_llm()

        snippets = "\n---\n".join(
            f"[文档{chunks[i].doc_id}] (chunk_{chunks[i].chunk_id}): {chunks[i].content[:300]}"
            for i in range(min(len(chunks), 8))
        )

        prompt = f"""检查以下文档片段是否存在信息冲突（矛盾）。如果存在冲突，输出JSON数组；如果无冲突，输出空数组[]。

冲突示例：文档A说"退款时效7天"，文档B说"退款时效15天"。

文档片段:
{snippets}

JSON (冲突列表，无冲突则[]):"""

        try:
            resp = await llm.ainvoke(prompt)
            text = resp.content if hasattr(resp, "content") else str(resp)
            json_match = re.search(r'\[.*\]', text, re.DOTALL)
            if json_match:
                conflicts = json.loads(json_match.group())
                if isinstance(conflicts, list):
                    return conflicts
        except Exception:
            pass

        return []

    async def resolve_conflicts(self, conflicts: list[dict], chunks: list[SearchResult]) -> str:
        """冲突消解：评估信源权威优先级，给出推荐结论。"""
        if not conflicts:
            return ""

        llm = await self._get_llm()

        prompt = f"""以下是从多个文档中检测到的信息冲突，请进行消解：

冲突列表: {json.dumps(conflicts, ensure_ascii=False)}

评估标准：
1. 文档版本新旧：新版 > 旧版
2. 发布部门层级：上级部门 > 下级部门
3. 最后更新时间：最近更新 > 较早更新
4. 如果无法判断权威性，同时呈现多方说法

请输出冲突消解结果（简洁，2-3句话）:"""

        try:
            resp = await llm.ainvoke(prompt)
            return (resp.content if hasattr(resp, "content") else str(resp)).strip()
        except Exception:
            return ""

    # ------------------------------------------------------------------
    # 主流程: Agentic RAG
    # ------------------------------------------------------------------
    async def retrieve(
        self,
        query: str,
        qdrant_retriever: QdrantRetriever,
        es_retriever: ESRetriever,
        reranker: Reranker,
        top_k: int = 10,
    ) -> AgenticResult:
        """Agentic RAG 主循环。

        流程: 意图分类 → 初始检索 → 充分性判断 → (不充分→改写→补充检索)×3 → 冲突检测 → 重排序
        """
        t0 = time.perf_counter()
        rewritten_history = [query]
        all_chunks: dict[int, SearchResult] = {}
        best_score = 0.0
        best_chunks = []

        # --- Step 0: 意图分类 + 查询改写 ---
        analysis = await self.classify_and_rewrite(query)
        intent = analysis.get("intent", "conceptual")
        search_queries = analysis.get("rewritten_queries", [query])

        # --- 多轮检索 ---
        for round_num in range(1, self.max_rounds + 1):
            current_query = search_queries[0] if search_queries else query

            # 根据策略选择检索方式
            strategy = analysis.get("retrieval_strategy", "hybrid")

            round_chunks = []
            if strategy in ("keyword_first", "hybrid"):
                es_results = await es_retriever.search(current_query, top_k=50)
                round_chunks.extend(es_results)
            if strategy in ("semantic_first", "multi_query", "hybrid"):
                embeddings = await qdrant_retriever.encode([current_query])
                qdrant_results = await qdrant_retriever.search(embeddings[0], top_k=100)
                round_chunks.extend(qdrant_results)

            # 多查询并行（对比型）
            if strategy == "multi_query" and len(search_queries) > 1:
                for sq in search_queries[1:3]:
                    emb = await qdrant_retriever.encode([sq])
                    qr = await qdrant_retriever.search(emb[0], top_k=30)
                    round_chunks.extend(qr)zz

            # RRF 融合 + 去重
            seen = set()
            fused = []
            for c in sorted(round_chunks, key=lambda x: x.score, reverse=True):
                key = (c.doc_id, c.chunk_id)
                if key not in seen:
                    seen.add(key)
                    fused.append(c)

            # 合并到全局集合
            before_count = len(all_chunks)
            for c in fused:
                key = (c.doc_id, c.chunk_id)
                if key not in all_chunks:
                    all_chunks[key] = c

            new_count = len(all_chunks) - before_count
            total_count = len(all_chunks)

            # --- 充分性判断 ---
            current_list = sorted(all_chunks.values(), key=lambda x: x.score, reverse=True)
            sufficiency = await self.judge_sufficiency(query, current_list[:10])

            if sufficiency > best_score:
                best_score = sufficiency
                best_chunks = current_list[:top_k]

            # 充分 → 直接进入生成
            threshold = self.agentic_config.get("early_stop_threshold", 0.8)
            if sufficiency >= threshold:
                break

            # 新增文档净增量 < 20% → 提前终止
            if round_num > 1 and total_count > 0:
                gain_rate = new_count / total_count if total_count > 0 else 0
                if gain_rate < self.agentic_config.get("min_incremental_gain", 0.2):
                    break

            # 不充分 → 改写查询，下一轮
            if round_num < self.max_rounds:
                new_query = await self.rewrite_for_retrieval(query, current_list[:5], round_num)
                if new_query and new_query != query:
                    rewritten_history.append(new_query)
                    search_queries = [new_query]

        # --- 冲突检测 ---
        all_list = sorted(all_chunks.values(), key=lambda x: x.score, reverse=True)
        conflicts = await self.detect_conflicts(all_list[:10])

        # --- 重排序 ---
        if best_chunks:
            reranked = await reranker.rerank(query, best_chunks, top_k=top_k)
        else:
            reranked = await reranker.rerank(query, all_list[:top_k], top_k=top_k)

        warning = None
        if best_score < 0.5:
            warning = "检索结果置信度较低，以下为当前最佳匹配"

        return AgenticResult(
            chunks=reranked,
            confidence=best_score,
            rounds_used=len(rewritten_history),
            intent=intent,
            rewritten_queries=rewritten_history,
            conflicts=conflicts,
            warning=warning,
        )


# ==================================================================
# 多级缓存
# ==================================================================
class RetrievalCache:
    """L1 精确匹配 / L2 语义相似 / 按需扩展 L3 LLM响应缓存。"""

    def __init__(self, redis_client=None):
        self._redis = redis_client

    async def _get_redis(self):
        if self._redis is None:
            import redis.asyncio as aioredis
            settings = get_settings()
            self._redis = aioredis.from_url(settings.redis_url)
        return self._redis

    def _cache_key(self, query: str) -> str:
        return f"rag:cache:{hashlib.md5(query.encode()).hexdigest()}"

    async def get(self, query: str) -> dict | None:
        try:
            r = await self._get_redis()
            data = await r.get(self._cache_key(query))
            if data:
                return json.loads(data)
        except Exception:
            pass
        return None

    async def set(self, query: str, result: dict, ttl: int = 3600):
        try:
            r = await self._get_redis()
            await r.setex(self._cache_key(query), ttl, json.dumps(result, ensure_ascii=False, default=str))
        except Exception:
            pass


# ==================================================================
# 单例
# ==================================================================
qdrant_retriever = QdrantRetriever()
es_retriever = ESRetriever()
reranker = Reranker()
agentic_engine = AgenticRAGEngine()
retrieval_cache = RetrievalCache()
