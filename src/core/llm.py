# ============================================================
# LLM 工厂 — 统一创建 ChatOpenAI，处理无 key 降级 + 短超时
# langchain-openai 1.x 字段名：model_name / openai_api_key /
# openai_api_base / request_timeout / max_retries
# ============================================================
from __future__ import annotations

from src.core.config import get_settings


def create_llm(
    model: str | None = None,
    temperature: float = 0.1,
    max_tokens: int = 2048,
    timeout: float = 30.0,
    max_retries: int = 1,
):
    """创建 ChatOpenAI；无有效 API key 时返回 None（调用方走规则/降级）。"""
    settings = get_settings()
    key = settings.openai_api_key or ""
    if not key or key == "your-api-key":
        return None

    from langchain_openai import ChatOpenAI
    return ChatOpenAI(
        model_name=model or settings.llm_model,
        openai_api_key=key,
        openai_api_base=settings.openai_base_url,
        temperature=temperature,
        max_tokens=max_tokens,
        request_timeout=timeout,
        max_retries=max_retries,
    )
