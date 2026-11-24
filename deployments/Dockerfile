# ============================================================
# 一文通 · 多文档智能问答系统 — Dockerfile
# ============================================================
FROM python:3.11-slim

WORKDIR /app

# 系统依赖
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# 先装 CPU 版 torch（避免默认 CUDA 版导致镜像膨胀到 5GB+）
RUN pip install --no-cache-dir torch==2.5.1 --index-url https://download.pytorch.org/whl/cpu

# 其余依赖
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 源码
COPY src/ ./src/
COPY configs/ ./configs/
COPY frontend/ ./frontend/

EXPOSE 8000 8501

CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
