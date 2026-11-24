# ============================================================
# 演示数据种子 — 一键导入示例企业文档
# 覆盖黄金测试集 evidence 文档 1-11，含新旧两版退款政策（演示冲突消解）
#
# 用法：python scripts/seed_demo_data.py [--api-base http://localhost:8000]
# ============================================================
from __future__ import annotations

import argparse
import asyncio
import sys

# Windows 控制台默认 GBK，打印 emoji/中文符号会崩，强制 UTF-8
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

import httpx

DOCS = [
    {"name": "产品X技术白皮书.md", "version": "v2.0", "department": "技术部", "content": """# 产品X技术白皮书
## 性能指标
产品X的最大QPS上限为10000，设计架构支持水平扩展。
在压力测试中，产品X的峰值QPS达到12000，但官方建议生产环境不超过10000。
## 扩展架构
单实例QPS上限为10000。需要支持更高吞吐时，可通过水平扩展部署多个实例并配合负载均衡。
产品X支持Kubernetes部署，可配置HPA自动扩缩容。
"""},
    {"name": "HNSW索引与向量检索指南.md", "version": "v1.1", "department": "技术部", "content": """# HNSW索引与向量检索指南
## 什么是HNSW
HNSW（Hierarchical Navigable Small World）是一种基于图的近似最近邻搜索算法，通过分层结构加速检索。
## 复杂度
HNSW的时间复杂度为O(log N)，空间复杂度为O(N)，适用于百万级向量检索。
HNSW由Yu.A. Malkov和D.Y. Yashunin在2016年提出。
"""},
    {"name": "产品Y功能规格说明.md", "version": "v1.0", "department": "产品部", "content": """# 产品Y功能规格说明
## 性能指标
产品Y的QPS上限为5000，适用于中小规模业务场景。
## 与产品X对比
产品X最大QPS为10000，产品Y为5000，产品X的吞吐是产品Y的2倍。
"""},
    {"name": "数据安全与合规制度.md", "version": "v2.4", "department": "信息安全", "content": """# 数据安全与合规制度
## 合同文档存储要求
合同文档需加密存储，访问需审批留痕，保存期限10年。
## 数据分级
按敏感程度将信息资产分为公开/内部/机密/绝密四级并分级授权。
## 权限最小化
只授予完成工作所需的最小权限。
"""},
    {"name": "合同审批与归档流程.md", "version": "v4.0", "department": "法务部", "content": """# 合同审批与归档流程
## 审批步骤
合同审批共5步：发起申请→部门负责人审批→法务审核→财务审核→归档。
## 归档要求
合同文档需加密存储，访问需审批留痕，保存期限10年。
"""},
    {"name": "退款政策-旧版.md", "version": "v1.0", "department": "客服部", "content": """# 退款政策（旧版）
## 退款时效
用户提交退款申请后，退款时效为7天。
"""},
    {"name": "退款政策-现行.md", "version": "v3.0", "department": "客服部", "content": """# 退款政策（现行）
## 退款时效
用户提交退款申请后，退款时效为15天，7个工作日内到账。
"""},
    {"name": "客户投诉处理规范.md", "version": "v2.1", "department": "客服部", "content": """# 客户投诉处理规范
## 处理流程
客户投诉处理流程：受理登记→分级（普通/重大）→指派负责人→处理并回访→关闭归档。
## 升级规则
重大投诉需在24小时内上报。
"""},
    {"name": "新员工入职材料清单.md", "version": "v1.5", "department": "人力部", "content": """# 新员工入职材料清单
新员工入职需要准备的材料：身份证复印件、学历证明、离职证明、体检报告、银行卡信息。
"""},
    {"name": "项目延期审批权限表.md", "version": "v3.2", "department": "项目管理委员会", "content": """# 项目延期审批权限表
| 延期时长 | 审批人 |
| --- | --- |
| 小于7天 | 项目经理 |
| 7到30天 | 部门总监 |
| 大于30天 | 项目管理委员会 |
"""},
    {"name": "差旅与费用报销制度.md", "version": "v2.2", "department": "财务部", "content": """# 差旅与费用报销制度
## 住宿标准
一线城市住宿报销上限为每晚500元，其他城市350元。
## 报销流程
差旅费用报销流程：提交申请→主管审批→财务审核→打款，需附发票与行程单。
"""},
]


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--api-base", default="http://localhost:8000")
    args = parser.parse_args()

    async with httpx.AsyncClient(timeout=120) as client:
        for doc in DOCS:
            files = {"file": (doc["name"], doc["content"].encode("utf-8"), "text/markdown")}
            data = {"version": doc["version"], "department": doc["department"]}
            try:
                r = await client.post(f"{args.api_base}/api/v1/documents/upload", files=files, data=data)
                if r.status_code == 200:
                    j = r.json()
                    print(f"✓ {doc['name']}: {j.get('parse_status')} ({j.get('chunk_count')} chunks)")
                else:
                    print(f"✗ {doc['name']}: {r.status_code} {r.text[:100]}")
            except Exception as e:
                print(f"✗ {doc['name']}: {e}")

    print("\n完成。可到 Streamlit 或 Swagger 试问：")
    print("  - 退款时效是几天？（演示冲突消解：旧版7天 vs 现行15天）")
    print("  - 产品X和产品Y的QPS有什么区别？（演示多查询并行）")


if __name__ == "__main__":
    asyncio.run(main())
