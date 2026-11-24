# ============================================================
# 轻量意图分类器 — "轻量模型降本"
# 意图分类用便宜/快速的小模型（默认 gpt-4o-mini），大模型只留给生成。
# LLM 输出走 with_structured_output（Pydantic 结构化），失败回退正则/规则。
# ============================================================
from __future__ import annotations

import re
from typing import Literal

from pydantic import BaseModel, Field

from src.core.config import get_settings


class IntentResult(BaseModel):
    """意图分类的结构化输出 Schema。"""
    intent: Literal["factual", "conceptual", "comparative", "procedural"] = Field(
        description="用户查询意图：事实型/概念型/对比型/操作型"
    )
    confidence: float = Field(ge=0.0, le=1.0, description="分类置信度")
    rewritten_queries: list[str] = Field(default_factory=list, description="改写/扩展后的查询变体")
    keywords: list[str] = Field(default_factory=list, description="抽取的关键词")
    reasoning: str = Field(default="", description="分类理由")


INTENT_STRATEGY_MAP = {
    "factual": "keyword_first",
    "conceptual": "semantic_first",
    "comparative": "multi_query",
    "procedural": "structured_filter",
}


class IntentClassifier:
    """轻量意图分类 + 查询改写 + 检索策略路由。"""

    def __init__(self, llm=None):
        self._llm = llm
        self.settings = get_settings()

    async def _get_llm(self):
        if self._llm is None:
            from src.core.llm import create_llm
            self._llm = create_llm(model=self.settings.intent_model, temperature=0.0, max_tokens=512)
        return self._llm

    async def classify(self, query: str) -> dict:
        """意图分类。结构化输出优先，正则/规则兜底。"""
        llm = await self._get_llm()
        if llm is None:
            return self._rule_based(query)

        prompt = f"""分析用户查询的意图类型。事实型=查具体数值/编号/事实；概念型=解释定义/原理；
对比型=A与B比较/区别；操作型=如何做/流程步骤。

查询: {query}"""

        # 结构化输出（失败回退）
        try:
            structured = llm.with_structured_output(IntentResult)
            result: IntentResult = await structured.ainvoke(prompt)
            data = result.model_dump()
            if data.get("intent") in INTENT_STRATEGY_MAP:
                data["retrieval_strategy"] = INTENT_STRATEGY_MAP[data["intent"]]
                data["rewritten_queries"] = data.get("rewritten_queries") or [query]
                data["keywords"] = data.get("keywords") or re.findall(r'[一-鿿\w]{2,}', query)
                return data
        except Exception:
            pass

        # 正则兜底（老式 JSON 解析）
        try:
            resp = await llm.ainvoke(prompt + "\n输出JSON: {\"intent\":..., \"confidence\":..., \"rewritten_queries\":[...], \"keywords\":[...]}")
            text = resp.content if hasattr(resp, "content") else str(resp)
            m = re.search(r"\{.*\}", text, re.DOTALL)
            if m:
                import json
                data = json.loads(m.group())
                if data.get("intent") in INTENT_STRATEGY_MAP:
                    data["retrieval_strategy"] = INTENT_STRATEGY_MAP[data["intent"]]
                    data["rewritten_queries"] = data.get("rewritten_queries") or [query]
                    return data
        except Exception:
            pass

        return self._rule_based(query)

    @staticmethod
    def _rule_based(query: str) -> dict:
        """规则兜底（LLM 不可用时，零成本）。"""
        if re.search(r'(多少|金额|编号|几[个份张]|什么时候|谁|哪个|列出|退款|有效期)', query):
            intent = "factual"
        elif re.search(r'(什么是|定义|解释|含义|为什么|原理|概念)', query):
            intent = "conceptual"
        elif re.search(r'(对比|比较|区别|vs|和.*哪个|异同)', query):
            intent = "comparative"
        elif re.search(r'(如何|怎么|步骤|流程|部署|安装|配置|操作)', query):
            intent = "procedural"
        else:
            intent = "conceptual"

        return {
            "intent": intent,
            "confidence": 0.6,
            "rewritten_queries": [query],
            "keywords": re.findall(r'[一-鿿\w]{2,}', query),
            "retrieval_strategy": INTENT_STRATEGY_MAP[intent],
            "reasoning": "规则兜底分类",
        }
