# ============================================================
# Agentic RAG 检索引擎包
# ============================================================
from src.retrieval.models import AgenticResult, SearchResult
from src.retrieval.retrievers import ESRetriever, QdrantRetriever
from src.retrieval.reranker import Reranker
from src.retrieval.fusion import dedupe, reciprocal_rank_fusion
from src.retrieval.cache import RetrievalCache
from src.retrieval.tools import RetrievalToolRegistry, registry as tool_registry
from src.retrieval.agentic import AgenticRAGEngine

__all__ = [
    "AgenticResult",
    "SearchResult",
    "QdrantRetriever",
    "ESRetriever",
    "Reranker",
    "AgenticRAGEngine",
    "RetrievalCache",
    "RetrievalToolRegistry",
    "tool_registry",
    "reciprocal_rank_fusion",
    "dedupe",
]

# 单例
qdrant_retriever = QdrantRetriever()
es_retriever = ESRetriever()
reranker = Reranker()
agentic_engine = AgenticRAGEngine()
retrieval_cache = RetrievalCache()
