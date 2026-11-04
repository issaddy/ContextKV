# ContextKV 课题索引

| 内容 | 入口 |
|---|---|
| 实施任务与研究边界 | [task_spec.md](task_spec.md) |
| 当前执行机制 | [架构](../architecture.md)、[reading_runtime.md](reading_runtime.md) |
| 历史审查基线 | [review_20261104.md](review_20261104.md) |
| 当前运行入口 | [final_runbook.md](final_runbook.md) |
| 历史数值摘要与资产状态 | [reference_20261103.json](reference_20261103.json)、[budget_loop_20261103.json](budget_loop_20261103.json) |

当前机制以架构为准；详细接口见 [任务与来源](task_sources.md)、[支持搜索](support_and_search.md)、[执行账本](reading_runtime.md)。

正式比较固定 Qwen3-32B 和冻结 P0 共享 A：c4 执行 E0—E4，c1 仅执行 E1/E4，每格沿完整 350 题声明队列和自己的实际 P3 历史运行。保留 P0/P1、隔离/共享阅读、`whole_plan` 及 Native/固定 TTL 的实际对照，参数集中在运行手册。

旧逐题答案、奖励与成本原始记录已丢弃，历史摘要不作为当前验收证据。训练、推理及 GPU 测量暂停；代码实现状态与尚未采集的研究结果分别说明。
