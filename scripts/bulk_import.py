# ============================================================
# 运维脚本 — 批量文档导入
# ============================================================
"""批量导入文档到知识库。

用法:
    python scripts/bulk_import.py --kb-id 1 --dir ./docs/ --pattern "*.pdf"
"""
from __future__ import annotations

import argparse
import asyncio
import os
import sys
from pathlib import Path

# 添加项目根目录到 sys.path
sys.path.insert(0, str(Path(__file__).parent.parent))

import httpx


async def bulk_import(api_base: str, kb_id: int, directory: str, pattern: str = "*"):
    """批量导入目录下的所有文件。"""
    dir_path = Path(directory)
    if not dir_path.exists():
        print(f"目录不存在: {directory}")
        return

    files = list(dir_path.glob(pattern))
    print(f"找到 {len(files)} 个文件")

    async with httpx.AsyncClient(timeout=60) as client:
        for i, file_path in enumerate(files, 1):
            if file_path.is_file():
                print(f"[{i}/{len(files)}] 上传: {file_path.name}")
                try:
                    with open(file_path, "rb") as f:
                        resp = await client.post(
                            f"{api_base}/api/v1/documents/upload",
                            data={"kb_id": str(kb_id)},
                            files={"file": (file_path.name, f, "application/octet-stream")},
                        )
                    if resp.status_code == 200:
                        result = resp.json()
                        print(f"  → 成功, doc_id={result['id']}")
                    else:
                        print(f"  → 失败: {resp.status_code} {resp.text}")
                except Exception as e:
                    print(f"  → 异常: {e}")

    print(f"\n完成! 共处理 {len(files)} 个文件")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="批量导入文档")
    parser.add_argument("--api-base", default="http://localhost:8000")
    parser.add_argument("--kb-id", type=int, required=True)
    parser.add_argument("--dir", required=True)
    parser.add_argument("--pattern", default="*")
    args = parser.parse_args()

    asyncio.run(bulk_import(args.api_base, args.kb_id, args.dir, args.pattern))
