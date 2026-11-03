# 当前课题运行入口

运行使用明确配置的模型、端点、reader 身份与原生缓存配置。主机目录、GPU、服务端口和
输出目录由本次运行配置声明；不复用旧批次的固定路径或截止时间。

| 任务 | 当前入口 | 必须保留的对照与产物 |
|---|---|---|
| 结论预算、共享数值与有限归约 | `bench.reading_mechanisms` | 共享阅读 / `--numeric-need-isolation`；实际 messages、完整来源叶子与阶段账本 |
| 原生来源前缀机制 | `bench.prefix_read_mechanisms` | 实际 READ epoch、前缀身份与 owner 观察 |
| Reader 审计、会话队列及训练准备 | `training.a_workflow` 的 `audit`、`cohort`、`prepare-labels`、`reader-stage`、`reader-prepare` | 实际 messages 生成目标，完整原文独立核验；READ/REDUCE/ANSWER 分阶段 |
| Reader 训练 | `training.train_reader` | reader/checkpoint、来源分区、训练 token 与成本 |
| P0/P1 × R0/新 reader | `training.a_workflow compare`、`bench.evidence_policy_comparison` | 冻结实际输入四格、各自真实 P3、实际来源与历史 |
| 证据策略真实候选收益 | `training.preference_rollout` | 实际最终回答、官方质量与完整成本；绑定当前 reader |
| B 动作收益 | `training.plan_gain_rollout`、`training.plan_gain_completion`、`training.collect_plan_gains` | DIRECT/READ/REDUCE/GAP 的真实增量执行、当前完整状态、独立最终答案与成对费用 |
| B 训练与滚动比较 | `training.plan_gain_workflow`、`bench.rolling_study_workflow` | 显式动作与 `whole_plan` 固定计划对照分别登记；保留已付成本与绝对分支预算 |
| C 固定轨迹配对 | `profiling.retention_trace` | 同模型、同输入、同 DAG 下 Native/固定 TTL/学习保护；共享页与 idle 驻留成本 |
| 实际并发容量 | `bench.capacity_workflow` | 相同质量与完整响应条件下的原生/保护对照；连续设备和进程采样 |
| 官方评分与离线重评 | `bench.study_scoring`、`bench.reassess_saved` | 全声明题目分母，失败、未完成、未评分及其已付费用 |
| 全成本账本 | `profiling.study_cost_ledger` | 原始调用/采样引用、窗口并集、共享份额与未知覆盖 |

配置入口为 `configs/study.yaml`、`study_matrix.yaml`、`evidence_policy_comparison.yaml`；
模型 scope 位于 `configs/model_scopes/`。Reader、策略身份、来源与费用由本次阶段
manifest 记录。当前数值入口为 [reference_20261103.json](reference_20261103.json)
和 [budget_loop_20261103.json](budget_loop_20261103.json)。

机制重放示例（显式指定实际服务配置和新输出目录）：

```bash
python -m bench.reading_mechanisms --config ACTUAL-RUN.yaml \
  --read-layout source_first --cases pair correction long multihop \
  --output-dir NEW-OUTPUT/shared
python -m bench.reading_mechanisms --config ACTUAL-RUN.yaml \
  --read-layout source_first --numeric-need-isolation --cases long \
  --output-dir NEW-OUTPUT/isolated
```

Reader 准备与四格比较示例：

```bash
python -m training.a_workflow reader-prepare --config READER-STAGE.yaml --output NEW-OUTPUT/reader
python -m training.a_workflow compare --comparison-config COMPARISON.yaml \
  --policy-asset CURRENT-P1-ASSET --matrix MATRIX.yaml --dataset DATASET \
  --split dev --phase both --output NEW-OUTPUT/comparison
```

所有实验按运行前声明的 cohort、reader 身份和资源范围读取结果。质量未测得时保留未知，
完成子集另外报告；训练 loss、输入投影成功率、预测成本和容量观测不能替代正式答案评分。
