# ============================================================
# 认证 + 服务端权限注入
# user_groups 由服务端从 API Key 解析，绝不信任客户端请求体。
# ============================================================
from __future__ import annotations

from fastapi import Header, HTTPException

from src.core.config import get_settings


def resolve_user_groups(x_api_key: str | None) -> list[str]:
    """从 API Key 解析用户权限组。返回 ["*"] 表示全员可见。"""
    settings = get_settings()

    # 未启用认证 → 开放模式（本地开发）
    if not settings.auth_enabled:
        return ["*"]

    # 启用认证但未配置 key → 开放模式（避免误锁死，日志里提示）
    if not settings.api_keys:
        return ["*"]

    if not x_api_key or x_api_key not in settings.api_keys:
        raise HTTPException(status_code=401, detail="无效或缺失的 API Key")

    groups = settings.api_keys[x_api_key]
    if not groups or groups.strip() in ("*", ""):
        return ["*"]
    return [g.strip() for g in groups.split(",") if g.strip()]


async def get_user_groups(x_api_key: str | None = Header(default=None)) -> list[str]:
    """FastAPI 依赖：解析当前请求的权限组。"""
    return resolve_user_groups(x_api_key)
