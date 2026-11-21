# ============================================================
# 黄金测试集生成器 — LLM 粗生成 + 人工精标注
# ============================================================
"""
从真实文档批量生成 Q-A-E 候选，输出 JSONL 供人工核对。

工作流：
  1. 读取文档目录（md/txt/pdf 均可，先转文本）
  2. 对每篇文档，用 LLM 生成候选 QA（含 ground_truth + evidence_doc_id）
  3. 输出到 eval/golden_set.candidates.jsonl
  4. 人工逐条核对 ground_truth 和 evidence，确认后 rename 为 golden_set.jsonl

用法：
  python eval/build_golden_set.py --docs ./docs/ --output eval/golden_set.candidates.jsonl
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from langchain_openai import ChatOpenAI

from src.core.config import get_settings

# 每篇文档生成的问题数量（可按难度分配）
QUESTIONS_PER_DOC = 3

# 出题 Prompt：要求 LLM 输出 Q-A-E，且明确证据位置
GEN_PROMPT = """你是 RAG 测试集标注专家。基于下面这篇企业文档，生成 {n} 个可用于评测的问答对。

要求：
1. 问题覆盖不同类型：事实查询、概念解释、对比、操作步骤
2. 难度有区分：有的是"文档直接写明"，有的是"需要综合文档内多处信息"
3. ground_truth 要准确、可核对，不能编造文档里没有的信息
4. 只输出 JSON 数组，每个元素格式：
   {{"question": "...", "ground_truth": "...", "intent": "factual|conceptual|comparative|procedural", "difficulty": "easy|medium|hard"}}

文档内容：
{doc_content}

JSON 数组："""


def build_candidates(docs_dir: str, output: str, n_per_doc: int):
    settings = get_settings()
    if not settings.openai_api_key or settings.openai_api_key == "your-api-key":
        print("⚠️  未设置 OPENAI_API_KEY，无法用 LLM 生成。")
        print("   请先配置 .env，或手动编写 golden_set.jsonl（见种子示例）。")
        return

    llm = ChatOpenAI(
        model=settings.llm_model,
        api_key=settings.openai_api_key,
        base_url=settings.openai_base_url,
        temperature=0.2,
    )

    docs_dir = Path(docs_dir)
    text_files = list(docs_dir.rglob("*.md")) + list(docs_dir.rglob("*.txt"))
    print(f"找到 {len(text_files)} 个文本文件")

    candidates = []
    for i, f in enumerate(text_files, 1):
        try:
            content = f.read_text(encoding="utf-8")[:8000]  # 截断，控制 token
        except Exception:
            continue
        if len(content.strip()) < 50:
            continue

        print(f"[{i}/{len(text_files)}] 处理: {f.name}")
        prompt = GEN_PROMPT.format(n=n_per_doc, doc_content=content)
        try:
            resp = llm.invoke(prompt)
            text = resp.content if hasattr(resp, "content") else str(resp)
            # 提取 JSON 数组
            start, end = text.find("["), text.rfind("]")
            if start == -1 or end == -1:
                print(f"  → LLM 未返回 JSON 数组，跳过")
                continue
            arr = json.loads(text[start : end + 1])
            for item in arr:
                item["evidence_doc_id"] = i  # 证据文档暂标为当前文档，人工核对
                item["id"] = f"cand-{len(candidates):04d}"
                candidates.append(item)
            print(f"  → 生成 {len(arr)} 条候选")
        except Exception as e:
            print(f"  → 生成失败: {e}")

    out_path = Path(output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for c in candidates:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")

    print(f"\n共生成 {len(candidates)} 条候选 → {out_path}")
    print("下一步（人工，黄金集的核心）：")
    print("  1. 逐条核对 ground_truth 是否准确")
    print("  2. 核对/补全 evidence_doc_ids（跨文档的题要列多个）")
    print("  3. 核对 difficulty 和 intent 是否标注正确")
    print("  4. 确认后改名 golden_set.jsonl 进入评估")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="生成黄金测试集候选")
    parser.add_argument("--docs", required=True, help="文档目录")
    parser.add_argument("--output", default="eval/golden_set.candidates.jsonl")
    parser.add_argument("--n-per-doc", type=int, default=QUESTIONS_PER_DOC)
    args = parser.parse_args()

    build_candidates(args.docs, args.output, args.n_per_doc)
