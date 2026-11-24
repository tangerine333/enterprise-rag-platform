# ============================================================
# 黄金测试集生成器 — 600 条分层测试集（Easy:Medium:Hard = 4:4:2）
# 确定性生成（固定种子），可复现。产出 eval/golden_set.jsonl
#
# 用法：python eval/generate_golden_set.py
# ============================================================
from __future__ import annotations

import json
import random
from pathlib import Path

# ------------------------------------------------------------
# 语料：模拟企业知识库文档（doc_id -> 文档主题/部门/版本）
# 与 eval/run_eval.py 的检索指标配套（evidence_doc_ids 指向这里）
# ------------------------------------------------------------
DOCS = {
    1: {"title": "产品X技术白皮书", "department": "技术部", "version": "v2.0"},
    2: {"title": "HNSW索引与向量检索指南", "department": "技术部", "version": "v1.1"},
    3: {"title": "员工入职与离职管理办法", "department": "人力部", "version": "v3.0"},
    4: {"title": "产品Y功能规格说明", "department": "产品部", "version": "v1.0"},
    5: {"title": "数据安全与合规制度", "department": "信息安全", "version": "v2.4"},
    6: {"title": "合同审批与归档流程", "department": "法务部", "version": "v4.0"},
    7: {"title": "客户投诉处理规范", "department": "客服部", "version": "v2.1"},
    8: {"title": "新员工入职材料清单", "department": "人力部", "version": "v1.5"},
    9: {"title": "项目延期审批权限表", "department": "项目管理委员会", "version": "v3.2"},
    10: {"title": "退款政策（旧版）", "department": "客服部", "version": "v1.0"},
    11: {"title": "退款政策（现行）", "department": "客服部", "version": "v3.0"},
    12: {"title": "服务器采购与硬件标准", "department": "技术部", "version": "v1.3"},
    13: {"title": "差旅与费用报销制度", "department": "财务部", "version": "v2.2"},
    14: {"title": "信息资产分级与授权规范", "department": "信息安全", "version": "v2.0"},
    15: {"title": "绩效考核与晋升管理办法", "department": "人力部", "version": "v3.1"},
    16: {"title": "销售提成与奖金发放规则", "department": "财务部", "version": "v1.8"},
    17: {"title": "第三方供应商准入与审计", "department": "采购部", "version": "v2.0"},
    18: {"title": "办公网络与VPN使用规范", "department": "技术部", "version": "v1.4"},
    19: {"title": "知识库内容贡献激励制度", "department": "运营部", "version": "v1.0"},
    20: {"title": "应急预案与灾备演练流程", "department": "运维部", "version": "v2.5"},
}

# ------------------------------------------------------------
# 每类意图的事实池（用于拼装问题与标准答案）
# ------------------------------------------------------------
FACT_POOL = {
    "factual": [
        ("产品X的最大QPS上限", "10000", [1]),
        ("产品Y的QPS上限", "5000", [4]),
        ("合同文档的加密存储要求", "需加密存储，访问审批留痕，保存期限10年", [5, 6]),
        ("退款时效（现行）", "15天", [11]),
        ("新员工入职需要准备的材料", "身份证复印件、学历证明、离职证明、体检报告、银行卡信息", [8]),
        ("HNSW索引的时间复杂度", "O(log N)", [2]),
        ("项目延期7-30天的审批人", "部门总监", [9]),
        ("差旅住宿报销上限", "一线城市每晚500元，其他城市350元", [13]),
        ("服务器采购的最低配置", "32核CPU、128G内存、NVMe SSD", [12]),
        ("VPN连接的强制要求", "双因子认证，会话超时30分钟", [18]),
        ("供应商准入审计周期", "每年一次现场审计", [17]),
        ("绩效考核的周期", "季度考核+年度综合评定", [15]),
    ],
    "conceptual": [
        ("HNSW索引的原理", "分层可导航小世界图算法，通过多层图结构加速近似最近邻检索，时间复杂度O(log N)", [2]),
        ("什么是Small-to-Big分块", "子块用于精确检索、父块用于生成时回溯完整上下文的分块策略", [2, 1]),
        ("什么是数据分级", "按敏感程度将信息资产分为公开/内部/机密/绝密四级并分级授权", [14]),
        ("什么是应急预案", "为应对系统故障或灾难提前制定的响应与恢复流程", [20]),
        ("什么是双因子认证", "在密码之外增加第二重身份验证（如短信验证码/硬件令牌）", [18]),
        ("什么是权限最小化原则", "只授予完成工作所需的最小权限", [14, 5]),
    ],
    "comparative": [
        ("产品X和产品Y在QPS上的区别", "产品X最大QPS为10000，产品Y为5000，X是Y的2倍", [1, 4]),
        ("旧版与新版退款政策的区别", "旧版退款时效7天，新版为15天，以新版为准", [10, 11]),
        ("一线城市与其他城市差旅报销标准的区别", "住宿上限一线城市500元/晚，其他城市350元/晚", [13]),
        ("机密级与绝密级信息的授权区别", "机密级需部门负责人审批，绝密级需信息安全委员会审批", [14]),
    ],
    "procedural": [
        ("合同审批流程", "发起申请→部门负责人审批→法务审核→财务审核→归档，共5步", [6]),
        ("客户投诉处理流程", "受理登记→分级→指派负责人→处理回访→关闭归档，重大投诉24小时内上报", [7]),
        ("新员工入职办理流程", "提交材料→签订合同→开通权限→领取设备→入职培训", [8, 3]),
        ("差旅费用报销流程", "提交申请→主管审批→财务审核→打款，需附发票与行程单", [13]),
        ("供应商准入流程", "资质初审→现场审计→试合作→正式入库", [17]),
        ("灾备演练流程", "制定方案→通知相关方→执行演练→复盘改进", [20]),
    ],
}


def _generate() -> list[dict]:
    rng = random.Random(20240101)  # 固定种子，可复现
    items: list[dict] = []
    _id = 0

    # 难度配额：easy 240 / medium 240 / hard 120
    plan = [("easy", 240), ("medium", 240), ("hard", 120)]

    for difficulty, quota in plan:
        intents = list(FACT_POOL.keys())
        for _ in range(quota):
            intent = intents[_id % len(intents)]
            pool = FACT_POOL[intent]
            topic, answer, evidence = pool[_id % len(pool)]

            if difficulty == "easy":
                question = _question_by_intent(intent, topic)
                ev = evidence[:1]
            elif difficulty == "medium":
                question = _question_by_intent(intent, topic, with_detail=True)
                # medium 允许跨文档整合
                ev = evidence[:2] if len(evidence) >= 2 and _id % 2 == 0 else evidence[:1]
            else:
                question = _question_by_intent(intent, topic, with_detail=True, cross=True)
                # hard 跨文档综合推理
                ev = (evidence + [evidence[0] % 20 + 1])[:3] if _id % 2 == 0 else evidence[:2]

            ev = sorted(set(d for d in ev if 1 <= d <= 20))
            items.append({
                "id": f"{difficulty[:3]}-{_id:04d}",
                "difficulty": difficulty,
                "intent": intent,
                "question": question,
                "ground_truth": answer,
                "evidence_doc_ids": ev,
                "note": "",
            })
            _id += 1

    return items


def _question_by_intent(intent: str, topic: str, with_detail: bool = False, cross: bool = False) -> str:
    suffix = "？请给出具体数值。" if with_detail else "？"
    if cross:
        suffix = "？请结合相关制度给出结论。"
    if intent == "factual":
        return f"{topic}是多少{suffix}"
    if intent == "conceptual":
        t = topic[3:] if topic.startswith("什么是") else topic
        return f"请解释「{t}」{suffix}"
    if intent == "comparative":
        return f"请说明{topic}{suffix}"
    return f"{topic}有哪些步骤{suffix}"


def main():
    items = _generate()
    # 校验配额
    from collections import Counter
    dist = Counter(i["difficulty"] for i in items)
    print(f"生成 {len(items)} 条，难度分布: {dict(dist)}")

    out = Path(__file__).parent / "golden_set.jsonl"
    with open(out, "w", encoding="utf-8") as f:
        for it in items:
            f.write(json.dumps(it, ensure_ascii=False) + "\n")
    print(f"已写入 {out}")


if __name__ == "__main__":
    main()
