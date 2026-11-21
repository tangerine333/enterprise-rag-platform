# ============================================================
# 黄金测试集评估 Harness — 检索指标（无需 LLM）+ RAGAS（需 LLM）
# ============================================================
"""
对 golden_set.jsonl 里的每条 Q-A-E，跑检索 + 生成，产出可对比指标。

两层评估（对应简历里两个数字）：
  检索指标（纯规则，可进 CI，不需要 LLM）：
    Recall@K        — Top-K 召回里是否命中 evidence_doc_ids
    MRR             — 第一个证据文档的排名倒数
    Redundancy      — 召回块里"无关文档"的比例（对应"检索冗余-35%"）
  生成指标（RAGAS，需 OPENAI_API_KEY）：
    Faithfulness / Answer Relevance / Context Precision / Context Recall

用法：
  # 只跑检索指标（不需要 LLM，最快，CI 用）
  python eval/run_eval.py --mode retrieval

  # 跑完整评估（检索 + RAGAS 生成指标，需要 LLM）
  python eval/run_eval.py --mode full
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.core.database import store


# ============================================================
# 黄金测试集加载
# ============================================================
def load_golden_set(path: str) -> list[dict]:
    items = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    return items


# ============================================================
# 检索评估（纯规则，不依赖 LLM）
# ============================================================
def evaluate_retrieval(items: list[dict], retrieve_fn) -> dict:
    """对每条黄金样本跑检索，统计 Recall@K / MRR / Redundancy。

    retrieve_fn(query) -> list[SearchResult]（含 doc_id 字段）
    """
    recall_hits = {5: 0, 10: 0}
    reciprocal_ranks = []
    redundancies = []
    per_item = []

    for item in items:
        evidence = set(item["evidence_doc_ids"])
        results = retrieve_fn(item["question"])

        # 取 doc_id 序列（去重，保序）
        seen, doc_ids = set(), []
        for r in results:
            if r.doc_id not in seen:
                seen.add(r.doc_id)
                doc_ids.append(r.doc_id)

        # Recall@K：Top-K 是否覆盖所有证据文档
        for k in recall_hits:
            if evidence <= set(doc_ids[:k]):
                recall_hits[k] += 1

        # MRR：第一个证据文档出现的位置
        for rank, did in enumerate(doc_ids, start=1):
            if did in evidence:
                reciprocal_ranks.append(1.0 / rank)
                break
        else:
            reciprocal_ranks.append(0.0)

        # Redundancy：召回结果中，doc_id 不在证据集里的占比
        total = len(doc_ids)
        irrelevant = sum(1 for d in doc_ids if d not in evidence)
        redundancies.append(irrelevant / total if total else 0.0)

        per_item.append({
            "id": item["id"],
            "question": item["question"],
            "evidence": sorted(evidence),
            "retrieved_docs": doc_ids[:10],
            "hit": bool(evidence <= set(doc_ids[:10])),
        })

    n = len(items)
    return {
        "count": n,
        "recall_at_5": recall_hits[5] / n if n else 0.0,
        "recall_at_10": recall_hits[10] / n if n else 0.0,
        "mrr": sum(reciprocal_ranks) / n if n else 0.0,
        "redundancy": sum(redundancies) / n if n else 0.0,
        "per_item": per_item,
    }


# ============================================================
# 报告输出
# ============================================================
def print_retrieval_report(metrics: dict):
    print()
    print("=" * 62)
    print("📊 检索层评估结果（黄金测试集）")
    print("=" * 62)
    print(f"  样本数:        {metrics['count']}")
    print(f"  Recall@5:      {metrics['recall_at_5']:.1%}  (证据文档进入 Top-5 的比例)")
    print(f"  Recall@10:     {metrics['recall_at_10']:.1%}  (证据文档进入 Top-10 的比例)")
    print(f"  MRR:           {metrics['mrr']:.3f}  (第一个证据文档排名的倒数平均)")
    print(f"  Redundancy:    {metrics['redundancy']:.1%}  (召回里无关文档占比，越低越好)")
    print("-" * 62)
    print("  未命中的样本（需重点排查）：")
    for p in metrics["per_item"]:
        if not p["hit"]:
            print(f"    ✗ {p['id']}: {p['question']}")
            print(f"      证据文档 {p['evidence']}，实际召回 {p['retrieved_docs']}")
    if all(p["hit"] for p in metrics["per_item"]):
        print("    （全部命中）")
    print()
    print("💡 关键解读：")
    print("  1. 关注相对变化而非绝对值 — 优化前后对比才有意义")
    print("  2. Recall@10 < 0.9 → 检索没召回足够信息，调大 top_k 或改进 embedding")
    print("  3. Redundancy > 0.5 → 召回太宽，加 Reranker 或收紧意图路由")
    print("  4. 这是你简历'检索冗余-35%'的来源 — 基线 vs 优化后的 Redundancy 差值")


# ============================================================
# 主入口
# ============================================================
async def main():
    parser = argparse.ArgumentParser(description="黄金测试集评估")
    parser.add_argument("--golden-set", default="eval/golden_set.jsonl")
    parser.add_argument("--mode", choices=["retrieval", "full"], default="retrieval",
                        help="retrieval=只跑检索指标(无需LLM)；full=检索+RAGAS(需LLM)")
    args = parser.parse_args()

    items = load_golden_set(args.golden_set)
    if not items:
        print(f"⚠️  黄金测试集为空或不存在: {args.golden_set}")
        return
    print(f"加载 {len(items)} 条黄金样本")

    # 按难度统计
    dist = {}
    for it in items:
        dist[it.get("difficulty", "?")] = dist.get(it.get("difficulty", "?"), 0) + 1
    print(f"难度分布: {dist}")

    # --- 检索评估：直接复用项目里的检索引擎 ---
    # 注意：需要先启动 Qdrant + ES 并导入文档，否则这里会返回空
    from src.retrieval import agentic_engine, qdrant_retriever, es_retriever, reranker

    def retrieve_fn(query: str):
        """同步包装：跑一次 Agentic 检索，返回 SearchResult 列表。"""
        result = asyncio.run(
            agentic_engine.retrieve(query, qdrant_retriever, es_retriever, reranker, top_k=10)
        )
        return result.chunks

    metrics = evaluate_retrieval(items, retrieve_fn)
    print_retrieval_report(metrics)

    # --- 完整模式：追加 RAGAS 生成指标 ---
    if args.mode == "full":
        if not os.getenv("OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY") == "your-api-key":
            print("\n⚠️  未设置 OPENAI_API_KEY，跳过 RAGAS 生成指标（检索指标已跑完）。")
            return
        print("\n🚀 追加 RAGAS 生成指标评估...")
        await run_ragas(items, retrieve_fn)


async def run_ragas(items: list[dict], retrieve_fn):
    """对每条样本：检索 → 生成答案 → RAGAS 评分。"""
    from ragas import SingleTurnSample
    from ragas.metrics import (
        Faithfulness,
        AnswerRelevancy,
        ContextPrecision,
        ContextRecall,
    )

    from src.retrieval import agentic_engine

    metrics = {
        "Faithfulness": Faithfulness(),
        "Answer Relevance": AnswerRelevancy(),
        "Context Precision": ContextPrecision(),
        "Context Recall": ContextRecall(),
    }
    results = {name: [] for name in metrics}

    for i, item in enumerate(items, 1):
        chunks = retrieve_fn(item["question"])
        contexts = [c.content for c in chunks[:10]]

        # 生成答案（复用项目里的 _generate_answer 逻辑）
        from src.api.search import _generate_answer
        answer_text, _, _ = await _generate_answer(
            item["question"], chunks, [], ""
        )

        sample = SingleTurnSample(
            user_input=item["question"],
            response=answer_text,
            retrieved_contexts=contexts,
            reference=item["ground_truth"],
        )
        for name, metric in metrics.items():
            try:
                results[name].append(metric.single_turn_score(sample))
            except Exception as e:
                print(f"  {name}: ❌ {e}")
        print(f"[{i}/{len(items)}] {item['question'][:40]}...")

    print()
    print("=" * 62)
    print("📊 RAGAS 生成指标（平均分）")
    print("=" * 62)
    for name, scores in results.items():
        if scores:
            avg = sum(scores) / len(scores)
            print(f"  {name:<22} {avg:.3f}")
    print()
    print("💡 关注：Faithfulness<0.7=模型在编造；Context Recall<0.5=检索没召够")


if __name__ == "__main__":
    asyncio.run(main())
