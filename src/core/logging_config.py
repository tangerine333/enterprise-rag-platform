# ============================================================
# 结构化日志 — request_id 通过 contextvars 贯穿所有日志
# ============================================================
from __future__ import annotations

import contextvars
import logging

# 当前请求的 request_id（中间件设置）
request_id_var: contextvars.ContextVar[str] = contextvars.ContextVar("request_id", default="-")


class RequestIdFilter(logging.Filter):
    def filter(self, record):
        record.request_id = request_id_var.get()
        return True


def setup_logging() -> logging.Logger:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter(
        "%(asctime)s %(levelname)s [%(request_id)s] %(name)s: %(message)s"
    ))
    handler.addFilter(RequestIdFilter())

    root = logging.getLogger()
    root.handlers = [handler]
    root.setLevel(logging.INFO)

    return logging.getLogger("rag")


logger = setup_logging()
