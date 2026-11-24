# ============================================================
# 测试全局配置 — 强制无 LLM / 无外部服务，保证测试可离线跑
# ============================================================
import os

# 必须在导入 src 之前设置，避免真实 API 调用
os.environ["OPENAI_API_KEY"] = ""
os.environ["OPENAI_BASE_URL"] = "http://127.0.0.1:9/v1"
os.environ["AUTH_ENABLED"] = "false"

import pytest


@pytest.fixture
def test_store(tmp_path):
    """隔离的元数据存储（临时 SQLite）。"""
    from src.core.database import MetadataStore
    return MetadataStore(db_path=str(tmp_path / "test.db"))
