# ============================================================
# 运维脚本 — 压测工具
# ============================================================
"""检索延迟压测。

用法:
    python scripts/benchmark_latency.py --api-base http://localhost:8000 --concurrency 10 --count 100
"""
from __future__ import annotations

import argparse
import asyncio
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import httpx

SAMPLE_QUERIES = [
    "公司的数据安全策略是什么？",
    "合同审批流程有几个步骤？",
    "如何处理客户投诉？",
    "新员工入职需要哪些材料？",
    "项目延期申请的审批权限是谁？",
]


async def benchmark(api_base: str, concurrency: int, count: int):
    """并发压测。"""
    latencies = []
    errors = 0

    async def make_request(client: httpx.AsyncClient, query: str):
        nonlocal errors
        t0 = time.perf_counter()
        try:
            resp = await client.post(
                f"{api_base}/api/v1/search/search",
                json={
                    "query": query,
                    "kb_ids": [1],
                    "top_k": 10,
                    "enable_agentic": False,
                },
                timeout=30,
            )
            if resp.status_code == 200:
                latencies.append((time.perf_counter() - t0) * 1000)
            else:
                errors += 1
        except Exception:
            errors += 1

    async with httpx.AsyncClient() as client:
        tasks = []
        for i in range(count):
            query = SAMPLE_QUERIES[i % len(SAMPLE_QUERIES)]
            tasks.append(make_request(client, query))
            if len(tasks) >= concurrency:
                await asyncio.gather(*tasks)
                tasks = []

        if tasks:
            await asyncio.gather(*tasks)

    if latencies:
        latencies.sort()
        print(f"\n===== 压测结果 =====")
        print(f"总请求数: {count}")
        print(f"成功数: {len(latencies)}")
        print(f"失败数: {errors}")
        print(f"P50: {latencies[len(latencies) // 2]:.1f}ms")
        print(f"P99: {latencies[int(len(latencies) * 0.99)]:.1f}ms")
        print(f"平均: {statistics.mean(latencies):.1f}ms")
        print(f"QPS (估算): {len(latencies) / (sum(latencies) / 1000 / concurrency):.1f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--api-base", default="http://localhost:8000")
    parser.add_argument("--concurrency", type=int, default=10)
    parser.add_argument("--count", type=int, default=100)
    args = parser.parse_args()

    asyncio.run(benchmark(args.api_base, args.concurrency, args.count))
