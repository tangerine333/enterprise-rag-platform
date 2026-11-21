# ============================================================
# 最简单的 RAGAS 评估 Demo
# 6 个核心指标，一条命令跑通
# ============================================================
"""
RAGAS 评估最小实操 — 对应学习清单 P1-8

6 个核心指标：
  Faithfulness          — 回答中每句话都能在检索文档中找到依据吗？
  Answer Relevance      — 回答是否紧扣问题？
  Context Precision     — 相关文档排在不相关文档前面了吗？
  Context Recall        — 检索是否覆盖了回答所需的全部信息？
  Noise Sensitivity     — 混入不相关文档时回答会被干扰吗？
  Answer Correctness    — 回答与标准答案的语义一致性和事实准确性

运行方式：
  cd enterprise-rag-platform
  python scripts/ragas_demo.py
"""

from __future__ import annotations

import os
import sys

# 把项目根目录加入 path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ragas import SingleTurnSample
from ragas.metrics import (
    Faithfulness,
    AnswerRelevancy,
    ContextPrecision,
    ContextRecall,
    NoiseSensitivity,
    AnswerCorrectness,
)

# ── 准备测试数据 ──────────────────────────────────────────────
# 每个样本包含: question, answer, contexts（检索到的文档）, ground_truth（标准答案）

SAMPLES = [
    {
        "question": "产品X的最大QPS是多少？",
        "answer": "产品X的最大QPS为10000，在峰值情况下可以达到12000。",
        "contexts": [
            "产品X的QPS上限为10000，设计架构支持水平扩展。",
            "在压力测试中，产品X的峰值QPS达到了12000，但官方建议不超过10000。",
            "产品Y的QPS上限为5000。",  # 噪音文档
        ],
        "ground_truth": "产品X的最大QPS为10000，峰值可达12000。",
    },
    {
        "question": "Redis支持哪些数据类型？",
        "answer": "Redis支持字符串、列表、集合、有序集合和哈希表五种基本数据类型。",
        "contexts": [
            "Redis支持多种数据结构，包括String、List、Set、Sorted Set、Hash。",
            "Redis还支持Bitmap、HyperLogLog、Geo、Stream等高级数据结构。",
            "MongoDB是一个文档数据库，支持BSON格式。",  # 噪音
        ],
        "ground_truth": "Redis支持String、List、Set、Sorted Set、Hash五种基本类型，以及Bitmap、HyperLogLog等高级类型。",
    },
    {
        "question": "什么是HNSW索引？",
        "answer": "HNSW（Hierarchical Navigable Small World）是一种近似最近邻搜索算法，通过多层图结构实现快速向量检索，时间复杂度为O(log N)。",
        "contexts": [
            "HNSW是一种基于图的近似最近邻搜索算法，通过分层结构加速检索。",
            "HNSW的时间复杂度为O(log N)，空间复杂度为O(N)，适用于百万级向量检索。",
            "HNSW由Yu.A. Malkov和D.A. Yashunin在2016年提出。",
        ],
        "ground_truth": "HNSW是一种分层可导航小世界图算法，用于近似最近邻搜索，时间复杂度O(log N)。",
    },
    {
        "question": "Docker和虚拟机的主要区别是什么？",
        "answer": "Docker容器共享宿主机操作系统内核，启动速度快（秒级），占用资源少。虚拟机需要完整的Guest OS，启动慢（分钟级），资源占用大。",
        "contexts": [
            "Docker容器与宿主机共享内核，无需模拟完整的操作系统，因此启动速度极快（通常在秒级）。",
            "虚拟机通过Hypervisor虚拟化硬件，每个VM需要运行完整的Guest OS，启动需数分钟。",
            "容器镜像通常为MB级别，而虚拟机镜像为GB级别。",
            "Kubernetes是一个容器编排平台。",  # 不太相关的文档
        ],
        "ground_truth": "Docker共享宿主机内核、启动快、资源占用少；虚拟机需要完整OS、启动慢、资源占用大。",
    },
    {
        "question": "LangGraph中add_conditional_edges的作用是什么？",
        "answer": "add_conditional_edges根据当前状态动态决定下一个节点，实现条件分支路由，是构建Agent循环的关键机制。",
        "contexts": [
            "add_conditional_edges方法接收一个条件函数，根据State返回下一个节点名称。",
            "当条件函数返回END时，工作流终止；返回节点名时，流程跳转到对应节点。",
            "LangGraph的StateGraph支持条件边、普通边和并行节点。",
        ],
        "ground_truth": "add_conditional_edges根据State动态路由到不同节点，实现条件分支和循环。",
    },
]

# AnswerCorrectness 额外样本（问题更难，答案需要推理）
CORRECTNESS_SAMPLES = [
    {
        "question": "如果产品X需要支持20000 QPS，应该如何扩展？",
        "answer": "需要进行水平扩展，部署多个实例并通过负载均衡分发请求。因为单实例最大QPS为10000，需要至少2个实例才能支持20000 QPS。",
        "contexts": [
            "产品X的单实例QPS上限为10000，设计架构支持水平扩展。",
            "水平扩展通过增加实例数量来提升总吞吐量，配合负载均衡器使用。",
            "产品X支持Kubernetes部署，可以配置HPA自动扩缩容。",
        ],
        "ground_truth": "通过水平扩展部署至少2个实例，配合负载均衡，可支持20000 QPS。",
    },
]


# ── 运行评估 ────────────────────────────────────────────────────
def run():
    # 确保有 OpenAI API Key（ragas 内部用 LLM 做评估）
    api_key = os.getenv("OPENAI_API_KEY", "")
    base_url = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")

    if not api_key or api_key == "your-api-key":
        print("=" * 60)
        print("⚠️  未设置有效的 OPENAI_API_KEY")
        print("=" * 60)
        print()
        print("RAGAS 需要调用 LLM 来做评估（例如判断 Faithfulness）。")
        print("请先设置环境变量：")
        print()
        print("  # Windows PowerShell:")
        print('  $env:OPENAI_API_KEY="sk-xxx"')
        print('  $env:OPENAI_BASE_URL="https://api.openai.com/v1"')
        print()
        print("  # 或者在 .env 文件中配置")
        print()
        print("然后重新运行: python scripts/ragas_demo.py")
        print()
        print("=" * 60)
        print()
        print("📋 如果你暂时没有 API Key，可以先看下面的评估数据示例：")
        print()
        print("  每个指标的含义和计算方式：")
        print("  ┌─────────────────────┬──────────────────────────────────────┐")
        print("  │ Faithfulness        │ 拆原子断言 → 逐一检查是否有文档支撑 │")
        print("  │ Answer Relevance    │ 基于回答反向生成问题 → 计算相似度   │")
        print("  │ Context Precision   │ 相关文档是否排在前面                 │")
        print("  │ Context Recall      │ 答案所需信息是否都在检索结果中       │")
        print("  │ Noise Sensitivity   │ 混入噪音文档后答案是否被干扰         │")
        print("  │ Answer Correctness  │ 答案与标准答案的语义和事实一致性     │")
        print("  └─────────────────────┴──────────────────────────────────────┘")
        return

    print("🚀 开始 RAGAS 6 维度评估...")
    print(f"   LLM: {os.getenv('LLM_MODEL', 'gpt-4o')}")
    print(f"   样本数: {len(SAMPLES)}")
    print()

    # ── 指标 1-5：Faithfulness / Answer Relevance / Context Precision / Context Recall / Noise Sensitivity ──
    metrics = {
        "Faithfulness": Faithfulness(),
        "Answer Relevance": AnswerRelevancy(),
        "Context Precision": ContextPrecision(),
        "Context Recall": ContextRecall(),
        "Noise Sensitivity": NoiseSensitivity(),
    }

    results: dict[str, list[float]] = {name: [] for name in metrics}

    for i, sample in enumerate(SAMPLES):
        print(f"  评估样本 {i + 1}/{len(SAMPLES)}: {sample['question'][:40]}...")
        rag_sample = SingleTurnSample(
            user_input=sample["question"],
            response=sample["answer"],
            retrieved_contexts=sample["contexts"],
            reference=sample["ground_truth"],
        )

        for name, metric in metrics.items():
            try:
                score = metric.single_turn_score(rag_sample)
                results[name].append(score)
                print(f"    {name}: {score:.4f}")
            except Exception as e:
                print(f"    {name}: ❌ {e}")

    # ── 指标 6：Answer Correctness ──
    print(f"  评估答案正确性 ({len(CORRECTNESS_SAMPLES)} 样本)...")
    ac_scores = []
    for sample in CORRECTNESS_SAMPLES:
        rag_sample = SingleTurnSample(
            user_input=sample["question"],
            response=sample["answer"],
            retrieved_contexts=sample["contexts"],
            reference=sample["ground_truth"],
        )
        try:
            score = AnswerCorrectness().single_turn_score(rag_sample)
            ac_scores.append(score)
            print(f"    Answer Correctness: {score:.4f}")
        except Exception as e:
            print(f"    Answer Correctness: ❌ {e}")

    results["Answer Correctness"] = ac_scores

    # ── 汇总报告 ────────────────────────────────────────────────────
    print()
    print("=" * 60)
    print("📊 RAGAS 评估结果汇总")
    print("=" * 60)
    print(f"{'指标':<30} {'平均分':>8}  {'说明'}")
    print("-" * 60)

    desc = {
        "Faithfulness": "回答是否忠于检索文档（>0.8 合格）",
        "Answer Relevance": "回答是否扣题（>0.8 合格）",
        "Context Precision": "相关文档排序质量（>0.7 合格）",
        "Context Recall": "检索覆盖度（>0.7 合格）",
        "Noise Sensitivity": "噪音抗干扰（越低越好，<0.3 合格）",
        "Answer Correctness": "答案与标准答案的一致性（>0.7 合格）",
    }

    for name, scores in results.items():
        if scores:
            avg = sum(scores) / len(scores)
            bar = "█" * int(avg * 20) + "░" * (20 - int(avg * 20))
            print(f"{name:<30} {avg:>7.3f}  {bar}  {desc.get(name, '')}")

    print("-" * 60)
    all_scores = [s for scores in results.values() for s in scores]
    if all_scores:
        overall = sum(all_scores) / len(all_scores)
        print(f"{'整体平均':<30} {overall:>7.3f}")


    print()
    print("💡 关键解读：")
    print("  1. 关注相对变化而非绝对值 — 优化前后对比才有意义")
    print("  2. Faithfulness < 0.7 → 模型在编造，检查 Prompt 和检索质量")
    print("  3. Context Recall < 0.5 → 检索没召回足够信息，调大 top_k 或改进 embedding")
    print("  4. Noise Sensitivity > 0.3 → 模型容易被无关文档干扰，加 Reranker")
    print("  5. Answer Correctness < 0.6 → 模型回答与标准答案偏差大，检查 Prompt")


if __name__ == "__main__":
    run()
