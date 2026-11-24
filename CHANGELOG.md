# Changelog

## v1.0.0 — 2026-11-24

首个可运行版本：完成从文档导入、解析、双索引构建到 Agentic 检索与冲突消解的全链路。

### Added

- **Agentic 检索回路**：基于 LangGraph 的有状态检索决策图（意图分类 → 路由召回 → 充分性判断 → 查询改写，最多 3 轮，净增量 < 20% 提前终止）
- **多格式解析路由**：文本 PDF → PyMuPDF；复杂版面 → MinerU CLI；扫描页/图片 → PaddleOCR；Office/Markdown 结构化解析，重型依赖缺失时自动降级
- **Small-to-Big 分块 + 表格双索引**：子块 256 token 精确检索 / 父块 1024 token 回溯；表格行级（携带表头）与表级（LLM 语义摘要）双索引
- **冲突消解与可信度建模**：LLM 冲突检测 → 四维权威性加权（版本/部门/时间/反馈）→ 多方呈现 + 推荐结论；点赞/点踩按 EMA（α=0.9）回流并写入审计日志
- **评测闭环**：600 条分层黄金集（table/conflict/general 子集）、Recall@K/MRR/冗余、要点覆盖率、RAGAS 6 维、badcase 四段归因，结果落盘 `eval/results/`
- **可观测性**：Prometheus 指标 + Grafana 看板 + 点踩率/P95/补偿积压阈值告警
- **工程化**：权限组前置过滤、SHA-256 增量去重、删除/更新向 Qdrant/ES 传播、失败补偿任务与 reconcile 接口、Redis 多级缓存与 30s 熔断降级
- **测试与 CI**：55 个离线用例 + GitHub Actions

### Fixed

- 删除文档后 Qdrant/ES 残留切片仍可被召回
- 缓存 key 未包含权限组，存在跨权限命中风险
- `/answer` 的 Prometheus 指标埋点位于 `return` 之后，指标从未被记录
- Grafana 数据源 uid 与看板引用不一致，面板无法加载
