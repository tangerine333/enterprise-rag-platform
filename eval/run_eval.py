# ============================================================
# 黄金测试集评估 Harness
#   检索指标（纯规则，可进 CI，无需 LLM）：Recall@K / MRR / Redundancy
#   答案完整度（要点覆盖率，规则口径，需生成）：answer_completeness
#   RAGAS 6 维度（需 LLM）：忠实度/相关性/上下文精度/召回/噪音敏感/正确性
#   基线(固定管道) vs Agentic(自适应) 对比 + 分类子集 + badcase 归因
#
# 用法：
#   python eval/run_eval.py --mode retrieval   # 检索指标 + 基线对比（无需 LLM）
#   python eval/run_eval.py --mode full        # 追加 RAGAS 与完整度（需 OPENAI_API_KEY）
#   python eval/run_eval.py --sample 50        # 小样本快速验证
# ============================================================
from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from collections import Counter
from datetime import datetime
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
        redundancy = irrelevant / total if total else 0.0
        redundancies.append(redundancy)

        per_item.append({
            "id": item["id"],
            "category": item.get("category", "general"),
            "question": item["question"],
            "evidence": sorted(evidence),
            "retrieved_docs": doc_ids[:10],
            "hit": bool(evidence <= set(doc_ids[:10])),
            "redundancy": redundancy,
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
# 答案完整度、分类子集与 badcase 归因
# ============================================================
def _bigrams(text: str) -> set[str]:
    text = "".join(text.split())
    if len(text) <= 1:
        return {text} if text else set()
    return {text[i:i + 2] for i in range(len(text) - 1)}


def point_covered(point: str, answer: str, threshold: float = 0.6) -> bool:
    """要点是否被答案覆盖：完整命中或字符二元组覆盖率 ≥ 阈值。"""
    if not point:
        return True
    if point in answer:
        return True
    pb, ab = _bigrams(point), _bigrams(answer)
    if not pb:
        return False
    return len(pb & ab) / len(pb) >= threshold


def answer_completeness(answer: str, key_points: list[str]) -> float:
    if not key_points:
        return 0.0
    return sum(1 for p in key_points if point_covered(p, answer)) / len(key_points)


def category_breakdown(per_item: list[dict]) -> dict:
    out = {}
    for cat in sorted({p["category"] for p in per_item}):
        rows = [p for p in per_item if p["category"] == cat]
        n = len(rows)
        out[cat] = {
            "count": n,
            "recall_at_10": sum(1 for p in rows if p["hit"]) / n if n else 0.0,
            "avg_redundancy": sum(p["redundancy"] for p in rows) / n if n else 0.0,
        }
    return out


def attribute_failures(per_item: list[dict]) -> dict:
    """把未命中样本归因到解析/召回/重排/生成四个环节。"""
    from src.core.database import store

    conn = store._get_conn()
    total_chunks = conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
    conn.close()
    if total_chunks == 0:
        return {
            "available": False,
            "reason": "知识库为空，无法区分解析缺失与召回未命中",
            "distribution": {},
        }

    def has_chunks(doc_id) -> bool:
        try:
            return bool(store.get_chunks_by_doc(int(doc_id)))
        except Exception:
            return False

    dist = Counter()
    examples = []
    for p in per_item:
        if p["hit"]:
            continue
        evidence, retrieved = p["evidence"], p["retrieved_docs"]
        if any(not has_chunks(d) for d in evidence):
            label = "parse_missing"
        elif not set(evidence) & set(retrieved):
            label = "recall_miss"
        elif not set(evidence) <= set(retrieved[:5]):
            label = "rerank_drop"
        else:
            label = "generation_omission"
        dist[label] += 1
        if len(examples) < 5:
            examples.append({
                "id": p.get("id", "?"), "label": label,
                "evidence": evidence, "retrieved_top5": retrieved[:5],
            })
    return {"available": True, "distribution": dict(dist), "examples": examples}


async def evaluate_completeness(items: list[dict], retrieve_fn, generate_fn) -> dict:
    """要点覆盖率口径的答案完整度（规则匹配，不依赖 LLM judge）。"""
    per_item, scores = [], []
    for item in items:
        chunks = await retrieve_fn(item["question"])
        answer = await generate_fn(item["question"], chunks)
        points = item.get("key_points") or [item.get("ground_truth", "")]
        score = answer_completeness(answer, points)
        scores.append(score)
        per_item.append({
            "id": item["id"],
            "category": item.get("category", "general"),
            "score": round(score, 4),
            "answer": answer[:300],
        })

    by_cat = {}
    for cat in sorted({p["category"] for p in per_item}):
        vals = [p["score"] for p in per_item if p["category"] == cat]
        by_cat[cat] = sum(vals) / len(vals) if vals else 0.0
    return {
        "avg": sum(scores) / len(scores) if scores else 0.0,
        "by_category": by_cat,
        "per_item": per_item,
    }


def save_results(payload: dict, mode: str) -> Path:
    out_dir = Path(__file__).parent / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{mode}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    return path


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
    errors: dict[str, int] = {}

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
                # ragas 0.2.x 的 metric 是异步接口
                if hasattr(metric, "single_turn_ascore"):
                    score = await metric.single_turn_ascore(sample)
                else:
                    score = metric.single_turn_score(sample)
                results[name].append(float(score))
            except Exception:
                errors[name] = errors.get(name, 0) + 1
        if i % 50 == 0:
            print(f"  [{i}/{len(items)}]")

    print()
    print("=" * 66)
    print("📊 RAGAS 6 维度生成指标（Agentic 平均分）")
    print("=" * 66)
    averages = {}
    for name, scores in results.items():
        if scores:
            averages[name] = sum(scores) / len(scores)
            print(f"  {name:<34} {averages[name]:.3f}")
    if errors:
        print(f"  ⚠️ 指标调用失败次数: {errors}")
    print()
    print("💡 Faithfulness<0.7=模型在编造；Context Recall<0.5=检索没召够")
    return averages


# ============================================================
# 主入口
# ============================================================
async def main():
    parser = argparse.ArgumentParser(description="黄金测试集评估")
    parser.add_argument("--golden-set", default="eval/golden_set.jsonl")
    parser.add_argument("--mode", choices=["retrieval", "full"], default="retrieval")
    parser.add_argument("--sample", type=int, default=0, help="只跑前 N 条（0=全量）")
    args = parser.parse_args()

    items = load_golden_set(args.golden_set)
    if not items:
        print(f"⚠️  黄金测试集为空: {args.golden_set}")
        return
    if args.sample and args.sample > 0:
        items = items[: args.sample]

    dist = Counter(it["difficulty"] for it in items)
    cats = Counter(it.get("category", "general") for it in items)
    print(f"加载 {len(items)} 条黄金样本，难度分布: {dict(dist)}，类别分布: {dict(cats)}")

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

    breakdown = category_breakdown(agentic["per_item"])
    attribution = attribute_failures(agentic["per_item"])

    print("📂 分类子集（Agentic）")
    for name, m in breakdown.items():
        print(f"  {name:<10} n={m['count']:>3}  Recall@10={m['recall_at_10']:.1%}  冗余={m['avg_redundancy']:.1%}")
    print("🧭 badcase 归因")
    if attribution["available"]:
        print(f"  {attribution['distribution']}")
        for ex in attribution["examples"]:
            print(f"    - {ex['id']} [{ex['label']}] evidence={ex['evidence']} top5={ex['retrieved_top5']}")
    else:
        print(f"  {attribution['reason']}")

    summary_keys = ("count", "recall_at_5", "recall_at_10", "mrr", "redundancy")
    payload = {
        "mode": "retrieval",
        "timestamp": datetime.now().isoformat(),
        "count": len(items),
        "difficulty_distribution": dict(dist),
        "category_distribution": dict(cats),
        "baseline": {k: base[k] for k in summary_keys},
        "agentic": {k: agentic[k] for k in summary_keys},
        "category_breakdown": breakdown,
        "attribution": attribution,
        "per_item": agentic["per_item"],
    }

    if args.mode == "full":
        api_key = os.getenv("OPENAI_API_KEY", "")
        if not api_key or api_key == "your-api-key":
            print("⚠️  未设置 OPENAI_API_KEY，跳过 RAGAS 与完整度。")
        else:
            from src.api.search import _generate_answer

            async def gen_fn(query, chunks):
                text, _, _ = await _generate_answer(query, chunks, [], "")
                return text

            try:
                ragas_scores = await run_ragas(items, agentic_fn, gen_fn)
            except Exception as exc:
                ragas_scores = {}
                print(f"⚠️  RAGAS 运行失败: {exc}")

            base_comp = await evaluate_completeness(items, baseline_fn, gen_fn)
            agentic_comp = await evaluate_completeness(items, agentic_fn, gen_fn)
            print("📝 答案完整度（要点覆盖率）")
            print(f"  基线={base_comp['avg']:.1%}  Agentic={agentic_comp['avg']:.1%}")
            for name, value in agentic_comp["by_category"].items():
                print(f"  {name:<10} {value:.1%}")

            payload.update({
                "mode": "full",
                "ragas": ragas_scores,
                "completeness": {
                    "baseline": base_comp["avg"],
                    "agentic": agentic_comp["avg"],
                    "by_category_agentic": agentic_comp["by_category"],
                    "per_item_agentic": agentic_comp["per_item"],
                },
            })

    path = save_results(payload, payload["mode"])
    print(f"📄 结果已写入 {path}")


if __name__ == "__main__":
    asyncio.run(main())
