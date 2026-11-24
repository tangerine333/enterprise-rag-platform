# ============================================================
# 应用配置 — 实用主义：只用成熟的组件
# ============================================================
from __future__ import annotations

from pathlib import Path

import yaml
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    app_name: str = "agentic-rag-platform"
    app_version: str = "2.0.0"
    debug: bool = False
    api_prefix: str = "/api/v1"

    # --- Qdrant ---
    qdrant_url: str = "http://localhost:6333"
    qdrant_collection: str = "enterprise_docs"
    qdrant_vector_size: int = 1024  # BGE-M3 dimension

    # --- Elasticsearch ---
    es_host: str = "localhost"
    es_port: int = 9200
    es_index: str = "rag_documents"

    # --- Redis ---
    redis_url: str = "redis://localhost:6379/0"

    # --- LLM (OpenAI API 兼容) ---
    openai_api_key: str = ""
    openai_base_url: str = "https://api.openai.com/v1"
    # 生成大模型（回答质量优先）
    llm_model: str = "gpt-4o"
    # 意图分类用轻量/便宜模型（降本：分类任务不需要大模型）
    intent_model: str = "gpt-4o-mini"
    llm_max_tokens: int = 2048
    llm_temperature: float = 0.1

    # --- Embedding (BGE-M3, 本地部署) ---
    embedding_model: str = "BAAI/bge-m3"
    embedding_device: str = "cpu"

    # --- Reranker ---
    reranker_model: str = "BAAI/bge-reranker-v2-m3"
    reranker_device: str = "cpu"

    # --- 认证（服务端权限注入，不信任客户端） ---
    auth_enabled: bool = False                 # 生产置 true
    api_keys: dict[str, str] = {}              # {"key": "权限组，逗号分隔"}；"*"=全员

    # --- 路径 ---
    project_root: Path = Path(__file__).parent.parent.parent
    configs_dir: Path = Path(__file__).parent.parent.parent / "configs"


def load_yaml_config(filename: str) -> dict:
    path = Path(__file__).parent.parent.parent / "configs" / filename
    if path.exists():
        with open(path, encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    return {}


_settings: Settings | None = None


def get_settings() -> Settings:
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings
