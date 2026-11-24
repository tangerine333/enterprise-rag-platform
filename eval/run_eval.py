# ============================================================
# 黄金测试集评估 Harness
#   检索指标（纯规则，可进 CI，无需 LLM）：Recall@K / MRR / Redundancy
#   RAGAS 6 维度（需 LLM）：忠实度/相关性/上下文精度/召回/噪音敏感/正确性
#   基线(固定管道) vs Agentic(自适应) 对比 —— 验证自适应检索的收益
#
# 用法：
#   python eval/run_eval.py --mode retrieval   # 检索指标 + 基线对比（无需 LLM）
#   python eval/run_eval.py --mode full        # 追加 RAGAS 6 维度（需 OPENAI_API_KEY）
# ============================================================
from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


def load_golden_set(path: str) -> list[dict]:
    items = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#"):
                items.append(json.loads(line))
    return items


# ============================================================
# 检索评估（纯规则，不依赖 LLM）
# ============================================================
async def evaluate_retrieval(items: list[dict], retrieve_fn) -> dict:
    recall_hits = {5: 0, 10: 0}
    reciprocal_ranks = []
    redundancies = []
    per_item = []

    for item in items:
        evidence = set(item["evidence_doc_ids"])
        results = await retrieve_fn(item["question"])

        seen, doc_ids = set(), []
        for r in results:
            if r.doc_id not in seen:
                seen.add(r.doc_id)
                doc_ids.append(r.doc_id)

        for k in recall_hits:
            if evidence <= set(doc_ids[:k]):
                recall_hits[k] += 1

        for rank, did in enumerate(doc_ids, start=1):
            if did in evidence:
                reciprocal_ranks.append(1.0 / rank)
                break
        else:
            reciprocal_ranks.append(0.0)

        total = len(doc_ids)
        irrelevant = sum(1 for d in doc_ids if d not in evidence)
        redundancies.append(irrelevant / total if total else 0.0)

        per_item.append({"id": item["id"], "question": item["question"],
                         "evidence": sorted(evidence), "retrieved_docs": doc_ids[:10],
                         "hit": bool(evidence <= set(doc_ids[:10]))})

    n = len(items)
    return {
        "count": n,
        "recall_at_5": recall_hits[5] / n if n else 0.0,
        "recall_at_10": recall_hits[10] / n if n else 0.0,
        "mrr": sum(reciprocal_ranks) / n if n else 0.0,
        "redundancy": sum(redundancies) / n if n else 0.0,
        "per_item": per_item,
    }


def print_compare(base: dict, agentic: dict):
    print()
    print("=" * 66)
    print("📊 检索层评估：基线(固定管道) vs Agentic(自适应)")
    print("=" * 66)
    print(f"{'指标':<18}{'基线':>14}{'Agentic':>14}{'变化':>16}")
    print("-" * 66)
    for key, label in [("recall_at_5", "Recall@5"), ("recall_at_10", "Recall@10"),
                       ("mrr", "MRR"), ("redundancy", "Redundancy(冗余)")]:
        b, a = base[key], agentic[key]
        delta = a - b
        sign = "+" if delta >= 0 else ""
        print(f"{label:<18}{b:>13.1%}{a:>13.1%}{sign}{delta:>15.1%}")
    print("-" * 66)

    # 冗余相对降幅
    if base["redundancy"] > 0:
        reduction = (base["redundancy"] - agentic["redundancy"]) / base["redundancy"]
        print(f"  检索冗余相对降幅: {reduction:.1%}")
    print()

    missed = [p for p in agentic["per_item"] if not p["hit"]]
    if missed:
        print(f"  Agentic 未命中 {len(missed)} 条（需排查）：")
        for p in missed[:10]:
            print(f"    ✗ {p['id']}: {p['question']}  证据{p['evidence']} 实际{p['retrieved_docs']}")
    else:
        print("  （Agentic 全部命中）")
    print()


# ============================================================
# RAGAS 6 维度评估
# ============================================================
async def run_ragas(items: list[dict], retrieve_fn, generate_fn):
    from ragas import SingleTurnSample
    from ragas.metrics import (
        AnswerCorrectness, AnswerRelevancy, ContextPrecision,
        ContextRecall, Faithfulness, NoiseSensitivity,
    )

    metrics = {
        "Faithfulness(忠实度)": Faithfulness(),
        "Answer Relevance(相关性)": AnswerRelevancy(),
        "Context Precision(上下文精度)": ContextPrecision(),
        "Context Recall(上下文召回)": ContextRecall(),
        "Noise Sensitivity(噪音敏感)": NoiseSensitivity(),
        "Answer Correctness(正确性)": AnswerCorrectness(),
    }
    results = {name: [] for name in metrics}

    for i, item in enumerate(items, 1):
        chunks = await retrieve_fn(item["question"])
        contexts = [c.content for c in chunks[:10]]
        answer_text = await generate_fn(item["question"], chunks)

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
                pass
        if i % 50 == 0:
            print(f"  [{i}/{len(items)}]")

    print()
    print("=" * 66)
    print("📊 RAGAS 6 维度生成指标（Agentic 平均分）")
    print("=" * 66)
    for name, scores in results.items():
        if scores:
            avg = sum(scores) / len(scores)
            print(f"  {name:<34} {avg:.3f}")
    print()
    print("💡 Faithfulness<0.7=模型在编造；Context Recall<0.5=检索没召够")


# ============================================================
# 主入口
# ============================================================
async def main():
    parser = argparse.ArgumentParser(description="黄金测试集评估")
    parser.add_argument("--golden-set", default="eval/golden_set.jsonl")
    parser.add_argument("--mode", choices=["retrieval", "full"], default="retrieval")
    args = parser.parse_args()

    items = load_golden_set(args.golden_set)
    if not items:
        print(f"⚠️  黄金测试集为空: {args.golden_set}")
        return
    from collections import Counter
    dist = Counter(it["difficulty"] for it in items)
    print(f"加载 {len(items)} 条黄金样本，难度分布: {dict(dist)}")

    from src.retrieval import (
        agentic_engine, es_retriever, qdrant_retriever, reranker,
        dedupe, reciprocal_rank_fusion,
    )

    async def baseline_fn(query: str):
        """基线：固定单轮混合检索（无意图路由/无迭代改写）。"""
        emb = await qdrant_retriever.encode([query])
        dense = await qdrant_retriever.search(emb[0], top_k=100)
        sparse = await es_retriever.search(query, top_k=50)
        fused = dedupe(reciprocal_rank_fusion([dense, sparse]))
        return await reranker.rerank(query, fused, top_k=10)

    async def agentic_fn(query: str):
        """Agentic：意图路由 + 充分性判断 + 迭代改写。"""
        r = await agentic_engine.retrieve(
            query, qdrant_retriever, es_retriever, reranker, top_k=10
        )
        return r.chunks

    base = await evaluate_retrieval(items, baseline_fn)
    agentic = await evaluate_retrieval(items, agentic_fn)
    print_compare(base, agentic)

    if args.mode == "full":
        api_key = os.getenv("OPENAI_API_KEY", "")
        if not api_key or api_key == "your-api-key":
            print("⚠️  未设置 OPENAI_API_KEY，跳过 RAGAS 生成指标。")
            return
        from src.api.search import _generate_answer

        async def gen_fn(query, chunks):
            text, _, _ = await _generate_answer(query, chunks, [], "")
            return text

        await run_ragas(items, agentic_fn, gen_fn)


if __name__ == "__main__":
    asyncio.run(main())
