# ============================================================
# 多文档冲突检测与消解 — 信源权威性「真实」加权
# 三阶段：LLM 冲突检测 → 权威性自动加权（版本/部门层级/更新时间/
# 反馈可信度）→ 多方呈现 + 推荐结论
# ============================================================
from __future__ import annotations

import json
import re
from datetime import datetime

from pydantic import BaseModel, Field

from src.retrieval.models import SearchResult


class ConflictClaim(BaseModel):
    claim_a: str = Field(description="文档A的说法")
    claim_b: str = Field(description="文档B的说法")
    topic: str = Field(description="冲突主题")


class ConflictList(BaseModel):
    conflicts: list[ConflictClaim] = Field(default_factory=list, description="冲突列表，无冲突则为空")


# 部门层级表（值越大权威越高）。未匹配到的部门取默认 1。
DEPARTMENT_HIERARCHY = {
    "总部": 3, "集团": 3, "总公司": 3, "管理层": 3, "总裁办": 3,
    "合规": 3, "法务": 3, "信息安全": 3,
    "部门": 2, "分公司": 2, "事业部": 2, "技术部": 2, "财务部": 2, "人力部": 2,
    "小组": 1, "团队": 1, "项目组": 1,
}
DEFAULT_DEPARTMENT_LEVEL = 1


def parse_version(version: str | None) -> float:
    """把 "v2.0" / "2.0" / "2024.1" 解析为可比较的数值。"""
    if not version:
        return 0.0
    m = re.search(r"(\d+(?:\.\d+)*)", str(version))
    if not m:
        return 0.0
    try:
        return float(m.group(1))
    except ValueError:
        return 0.0


def department_level(department: str | None) -> int:
    if not department:
        return DEFAULT_DEPARTMENT_LEVEL
    for name, level in DEPARTMENT_HIERARCHY.items():
        if name in department:
            return level
    return DEFAULT_DEPARTMENT_LEVEL


def recency_score(updated_at: str | None) -> float:
    """更新时间越近，权威性越高（0-1）。"""
    if not updated_at:
        return 0.5
    try:
        dt = datetime.fromisoformat(updated_at)
    except (ValueError, TypeError):
        return 0.5
    # 以 2020-01-01 为基准，映射到 0-1（最近5年线性）
    base = datetime(2020, 1, 1)
    delta = (dt - base).total_seconds()
    return max(0.0, min(1.0, delta / (5 * 365 * 24 * 3600)))


def compute_authority(doc: SearchResult) -> dict:
    """综合计算单篇文档的权威性，返回分数 + 各维度分解（可解释）。"""
    version_score = min(parse_version(doc.version) / 10.0, 1.0)  # v10 视为满分
    dept_score = department_level(doc.department) / 3.0
    recency = recency_score(doc.updated_at)
    feedback = float(doc.authority_score or 0.5)

    # 加权综合：反馈可信度 > 版本新旧 > 部门层级 > 更新时间
    total = 0.35 * feedback + 0.30 * version_score + 0.20 * dept_score + 0.15 * recency
    return {
        "doc_id": doc.doc_id,
        "version": doc.version,
        "department": doc.department,
        "updated_at": doc.updated_at,
        "authority_score": round(total, 4),
        "breakdown": {
            "feedback_weight": round(feedback, 3),
            "version": round(version_score, 3),
            "department_level": round(dept_score, 3),
            "recency": round(recency, 3),
        },
    }


class ConflictResolver:
    """冲突检测 + 权威性加权消解。"""

    def __init__(self, llm=None):
        self._llm = llm

    async def _get_llm(self):
        if self._llm is None:
            from src.core.llm import create_llm
            from src.core.config import get_settings
            self._llm = create_llm(model=get_settings().llm_model, temperature=0.0)
        return self._llm

    async def detect(self, chunks: list[SearchResult]) -> list[dict]:
        """LLM 检测多文档间信息冲突，标注矛盾点。"""
        if len(chunks) < 2:
            return []

        llm = await self._get_llm()
        if llm is None:
            return []
        snippets = "\n---\n".join(
            f"[文档{chunks[i].doc_id}] {chunks[i].content[:300]}"
            for i in range(min(len(chunks), 8))
        )
        prompt = f"""检查以下文档片段是否存在信息冲突（矛盾）。存在则列出每处冲突的双方说法与主题；无冲突则返回空列表。

文档片段:
{snippets}"""

        # 结构化输出（失败回退正则）
        try:
            structured = llm.with_structured_output(ConflictList)
            result: ConflictList = await structured.ainvoke(prompt)
            conflicts = [c.model_dump() for c in result.conflicts]
            if isinstance(conflicts, list):
                return conflicts
        except Exception:
            pass

        # 正则兜底
        try:
            resp = await llm.ainvoke(prompt + "\n输出JSON数组，无冲突输出[]：")
            text = resp.content if hasattr(resp, "content") else str(resp)
            m = re.search(r"\[.*\]", text, re.DOTALL)
            if m:
                data = json.loads(m.group())
                if isinstance(data, list):
                    return data
        except Exception:
            pass
        return []

    def rank_by_authority(self, chunks: list[SearchResult]) -> list[dict]:
        """按信源权威性对文档排序（真实计算，非 Prompt 空话）。"""
        metas = [compute_authority(c) for c in chunks]
        metas.sort(key=lambda x: x["authority_score"], reverse=True)
        return metas

    async def resolve(self, conflicts: list[dict], chunks: list[SearchResult]) -> str:
        """冲突消解：权威性加权排序 → 多方呈现 + 推荐结论。"""
        if not conflicts:
            return ""

        ranked = self.rank_by_authority(chunks)
        top = ranked[0] if ranked else None

        # 真实加权后的排序信息，喂给 LLM 做最终措辞
        authority_summary = "\n".join(
            f"- 文档{r['doc_id']} 权威分 {r['authority_score']}（版本:{r['version'] or '未知'}, "
            f"部门:{r['department'] or '未知'}, 更新时间:{r['updated_at'] or '未知'}）"
            for r in ranked
        )

        llm = await self._get_llm()
        if llm is not None:
            prompt = f"""以下是从多个文档检测到的信息冲突，已按信源权威性加权排序（分数越高越可信）：

{authority_summary}

冲突列表: {json.dumps(conflicts, ensure_ascii=False)}

请输出消解结论（2-3句话）：同时呈现多方说法，并给出推荐结论及依据（优先采用权威分最高的信源）。"""

            try:
                resp = await llm.ainvoke(prompt)
                return (resp.content if hasattr(resp, "content") else str(resp)).strip()
            except Exception:
                pass

        # LLM 不可用时降级：直接按权威分推荐
        if top:
            return (
                f"检测到 {len(conflicts)} 处信息冲突。"
                f"根据权威性加权（文档{top['doc_id']} 权威分最高 {top['authority_score']}），"
                f"推荐以文档{top['doc_id']}（版本:{top['version'] or '未知'}）为准，"
                f"其余说法仅供参考。"
            )
        return f"检测到 {len(conflicts)} 处信息冲突，但缺少足够元数据进行权威性判断，建议人工核实。"
