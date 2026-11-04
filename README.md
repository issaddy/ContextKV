# ContextKV

面向资源受限多轮文档问答的上下文管理与 KV 缓存协同优化研究。

Qwen3-32B 权重与解码参数冻结，不做微调或合并模型。

系统在实际问答历史、条件与版本基础上选择原文，按固定窗口和累计成本执行
READ/REDUCE/ANSWER，并基于原生缓存状态决定来源前缀保护。最终结论保留完整来源链，
数值需求共享阅读，确定性运算在 CPU 完成；模型归约按有限分片执行。

| 主线 | 当前入口 |
|---|---|
| A：实际历史、选材与阅读协议 | `training.a_workflow`、`training.preference_rollout`、`bench.evidence_policy_comparison` |
| B：显式阅读动作与完整成本 | `controller.action_policy`、`controller.rolling_policy`、`training.plan_gain_rollout`、`training.plan_gain_completion` |
| C：原生缓存与来源前缀保护 | `controller.prefix_retention`、`profiling.retention_trace`、`bench.capacity_workflow` |
| 评价与总成本 | `bench.study_scoring`、`bench.reassess_saved`、`profiling.study_cost_ledger` |

保留共享数值阅读 / `numeric_need_isolation`、显式动作 / `whole_plan` 固定计划、
Native / 固定 TTL，以及 P0/P1 规划器 × 阅读协议 的真实对照。四格先使用冻结实际输入，
协议对照为 `task_first + numeric_need_isolation` 与 `source_first + 共享提取`，随后沿各自真实生成历史执行 P3。质量使用全部声明任务的官方分母，失败与未完成任务的
已付调用、辅助计算和物理驻留费用一起记录。

```bash
uv sync --extra retrieval --extra evaluation --extra tokenizer --locked
uv run python -m bench.run_sessions --config configs/demo.yaml demo
uv run ruff check controller adapters bench profiling tests scripts engine_patches training
```

CPU 测试保留核心阅读、数值归约、来源链、目标可见性、动作决策和前缀接口冒烟。
CI 仅按需手动触发，提交代码和创建 PR 不自动运行测试。
实际模型运行由本次配置明确声明模型、端点、reader 身份、缓存设置、cohort 和输出目录。

| 文档 | 内容 |
|---|---|
| [课题索引](docs/study/README.md) | 当前机制与入口 |
| [任务要求](docs/study/task_spec.md) | 研究边界 |
| [接续说明](docs/study/review_20261103.md) | 根因、纠正与目标 |
| [运行手册](docs/study/final_runbook.md) | 当前复现入口 |
| [架构](docs/architecture.md) | 服务与状态链 |
| [数据与评价](docs/data_and_evaluation.md) | 官方任务、完整分母和成本口径 |
| [当前重评](docs/study/reference_20261103.json) | 重评数值 |
| [预算阅读记录](docs/study/budget_loop_20261103.json) | 当前机制数值 |
| [部署](docs/deployment.md) | 原生引擎与 HTTP 接入 |

`controller/` 为运行核心，`adapters/` 提供资料与检索适配，`engine_patches/` 对接原生缓存；
`bench/`、`profiling/`、`training/` 分别组织评价、成本采集与训练。
