# ContextKV 课题索引

| 内容 | 入口 |
|---|---|
| 实施任务与研究边界 | [task_spec.md](task_spec.md) |
| 当前执行机制 | [架构](../architecture.md)、[reading_runtime.md](reading_runtime.md) |
| 根因与实现要求 | [review_20261103.md](review_20261103.md) |
| 当前运行入口 | [final_runbook.md](final_runbook.md) |
| 当前重评数值 | [reference_20261103.json](reference_20261103.json)、[budget_loop_20261103.json](budget_loop_20261103.json) |

当前实现围绕完整结论预算与来源链、共享数值阅读、有限归约、实际历史和目标可见性、
B 显式动作收益、C Native/TTL 前缀保护，以及全部声明任务的官方评分和完整成本账本。
历史一次性批次脚本、旧 sweep 配方及重复状态快照已清理；旧代码和旧文档可从 Git 历史追溯。

A 入口是 `training.a_workflow`、`training.preference_rollout` 和
`bench.evidence_policy_comparison`；B 入口是 `training.plan_gain_rollout`、
`training.plan_gain_completion` 和 `bench.rolling_study_workflow`；C 入口是
`profiling.retention_trace` 和 `bench.capacity_workflow`。所有训练、评分和成本记录
绑定各自实际 reader 与来源。
