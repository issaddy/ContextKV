# ContextKV 课题索引

| 内容 | 入口 |
|---|---|
| 实施任务与研究边界 | [task_spec.md](task_spec.md) |
| 当前执行机制 | [架构](../architecture.md)、[reading_runtime.md](reading_runtime.md) |
| 当前审查与实施阶段 | [review_20261104.md](review_20261104.md) |
| 当前运行入口 | [final_runbook.md](final_runbook.md) |
| 当前重评数值 | [reference_20261103.json](reference_20261103.json)、[budget_loop_20261103.json](budget_loop_20261103.json) |

当前实现连接完整结论预算与来源链、共享数值阅读、有限归约、实际历史、B 动作收益及 C 来源前缀保护。Qwen3-32B 权重与解码固定。P0/P1 与隔离/共享阅读协议用于归因，完整联合入口读取冻结 P0 共享 A 配置。

A 的实际候选采集和比较使用 `training.preference_rollout`、`bench.evidence_policy_comparison`。B 使用 `training.plan_gain_workflow --mode observed_actions`，以真实动作配对学习最终质量差和动作前原生状态条件化的绝对分支成本。C 使用 `profiling.retention_trace`，按同窗口 Native/TTL 配对估值，并在保护集合中去重未来读取收益和共享页费用。

`bench.capacity_workflow --full-queue` 接入冻结 A、B 收益与 C 资产，执行完整 350 题的固定 B / 收益 B × native / value，在 c1/c4 下分别报告各自 P3、完整分母质量、响应时延和成本。`whole_plan` 及原两轮容量诊断作为实际对照保留。运行参数见运行手册，实测状态统一见数值记录。
