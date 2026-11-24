"""意图分类：规则兜底（无 LLM 场景）。"""
from src.retrieval.intent import IntentClassifier, INTENT_STRATEGY_MAP


def test_rule_based_factual():
    r = IntentClassifier._rule_based("合同金额是多少？")
    assert r["intent"] == "factual"
    assert r["retrieval_strategy"] == "keyword_first"


def test_rule_based_conceptual():
    r = IntentClassifier._rule_based("什么是HNSW索引？")
    assert r["intent"] == "conceptual"
    assert r["retrieval_strategy"] == "semantic_first"


def test_rule_based_comparative():
    r = IntentClassifier._rule_based("产品X和产品Y的QPS有什么区别？")
    assert r["intent"] == "comparative"
    assert r["retrieval_strategy"] == "multi_query"


def test_rule_based_procedural():
    r = IntentClassifier._rule_based("合同审批流程有哪些步骤？")
    assert r["intent"] == "procedural"
    assert r["retrieval_strategy"] == "structured_filter"


def test_strategy_map_complete():
    for intent in ("factual", "conceptual", "comparative", "procedural"):
        assert intent in INTENT_STRATEGY_MAP


def test_classify_falls_back_to_rule_without_llm():
    # 无 LLM（api key 为空）时 classify 应走规则兜底，不抛异常
    import asyncio
    c = IntentClassifier()
    r = asyncio.run(c.classify("退款时效是几天？"))
    assert r["intent"] in INTENT_STRATEGY_MAP
    assert r["retrieval_strategy"] in INTENT_STRATEGY_MAP.values()
