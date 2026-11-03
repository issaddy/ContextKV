# ContextKV 课题索引

当前审查基线为 `702ebe9`；原始实施任务书保存在 [task_spec.md](task_spec.md)。

## 单一状态入口

| 内容 | 文档 |
|---|---|
| 当前机制 | [架构](../architecture.md) |
| 最新已运行实验、训练规模与批次位置 | [final_report.md](final_report.md) |
| 本轮根因、代码修正和后续实现任务 | [review_20261103.md](review_20261103.md) |
| 既有模型与新训练目标的接续 | [reader_policy_readiness.md](reader_policy_readiness.md) |
| 运行与产物命令 | [final_runbook.md](final_runbook.md) |

R1、R2 和 P1 prototype22 均已完成实际训练。B 开发选择为 B3，C 开发选择为 native。公开开发候选均未达到当时的 P95 条件，实验冻结使用 P0R0；这一记录见封存报告。

本轮新增的评分修正改变了 MTRAGEval 条件分量和 2Wiki 官方分母，已有结果需按相同声明题目离线重评。归约与训练协议修正需要新的实际运行和训练资产；原始实验记录继续保存原数值。

## 实施入口

- A：`adapters/history_state.py`、`evidence_pool.py`、`support_model.py`、`controller/evidence_planner.py`、`training/`。
- B：`controller/reading.py`、`execution.py`、`claim_graph.py`、`plan_policy.py`、`plan_profiles.py`、`rolling_policy.py`。
- C：`controller/prefix_retention.py`、`prefix_view.py`、`engine_patches/`。
- 评价：`bench/study.py`、`study_metrics.py`、`study_scoring.py`、`public_multihop.py`。

数据、权重与原始运行产物保存在既有实验环境 `sjy-0518`，具体路径和模型身份以封存报告及各资产 manifest 为准。
