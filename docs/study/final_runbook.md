# 当前课题运行入口

运行冻结同一 Qwen3-32B 模型权重与解码参数，显式配置端点、协议身份与原生缓存。主机目录、GPU、服务端口和
输出目录由本次运行配置声明；不复用旧批次的固定路径或截止时间。

| 任务 | 当前入口 | 必须保留的对照与产物 |
|---|---|---|
| 结论预算、共享数值与有限归约 | `bench.reading_mechanisms` | 共享阅读 / `--numeric-need-isolation`；实际 messages、完整来源叶子与阶段账本 |
| 原生来源前缀机制 | `bench.prefix_read_mechanisms` | 实际 READ epoch、前缀身份与 owner 观察 |
| 阅读协议审计与会话队列 | `training.reader_audit` 的 `export`、`review`；`training.a_workflow` 的 `audit`、`cohort`、`prepare-labels` | 实际 messages 与完整原文独立核验，无 SFT 输出 |
| P0/P1 规划器 × 阅读协议 | `training.a_workflow compare`、`bench.evidence_policy_comparison` | task_first+隔离数值 / source_first+共享提取；固定实际输入四格与各自真实 P3 |
| 证据策略真实候选收益 | `training.preference_rollout` | 实际最终回答、官方质量与完整成本；绑定当前 reader |
| B 动作收益 | `training.plan_gain_rollout`、`training.plan_gain_completion`、`training.collect_plan_gains` | DIRECT/READ/REDUCE/GAP 的真实增量执行、当前完整状态、独立最终答案与成对费用 |
| B 训练与滚动比较 | `training.plan_gain_workflow`、`bench.rolling_study_workflow` | 显式动作与 `whole_plan` 固定计划对照分别登记；保留已付成本与绝对分支预算 |
| C 固定轨迹配对 | `profiling.retention_trace` | 同模型、同输入、同 DAG 下 Native/固定 TTL/学习保护；共享页与 idle 驻留成本 |
| 实际并发容量 | `bench.capacity_workflow` | 相同质量与完整响应条件下的原生/保护对照；连续设备和进程采样 |
| 官方评分与离线重评 | `bench.study_scoring`、`bench.reassess_saved` | 全声明题目分母，失败、未完成、未评分及其已付费用 |
| 全成本账本 | `profiling.study_cost_ledger` | 原始调用/采样引用、窗口并集、共享份额与未知覆盖 |

配置入口为 `configs/study.yaml`、`study_matrix.yaml`、`evidence_policy_comparison.yaml`；
P1 支持策略配置为 `configs/evidence_policy_fixed_qwen.yaml`。
审计配置为 `configs/reader_audit.yaml` 的 `reader_audit:{base_scope,data,review,resource_sample_seconds}`。
会话队列使用 `evidence_cohort`；P1 支持策略直接绑定固定生成配置。
辅助学习的 200 train / 40 dev 状态统一使用
`runs/study/budget-loop-20261103/reader-cohorts/full/cohort_manifest.json`。
task/support 配置必填 `training.cohort_manifest`；P1 采集必填
`collection.cohort_manifest` 和 `cohort_split`，开发采集显式选择 `dev`。
B 固定流程观测使用 `execution.state_observation`，从真实请求开始计量并记录 READ/REDUCE 决策状态；
不加载 gain 模型，不从旧 SQLite 补造物理状态。A 冻结后显式启用。
`training.collect_plan_gains --observations OBS --actual-turns TURNS --output PACKETS`
导出后，`training.plan_gain_workflow --mode observed_actions --observed-states PACKETS`
沿用源 cohort 身份。每个源 task 保留各合法动作首次可配对的观测，不按未来质量选点。
B 的 `training.plan_gain_rollout`、`training.plan_gain_workflow` 均必填
`--cohort-manifest` 指向同一文件；不再使用 `--partition-seed` 或 `--dev-fraction`
重新划分状态，实际采集数量从声明分区内选择并单独记录。
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

辅助训练复用现有核验标签与 BGE 资产，配置 `configs/auxiliary_training.yaml`。
运行时以 `CUDA_VISIBLE_DEVICES` 选择物理设备，配置内使用该进程的 `cuda:0`：

```bash
python -m training.train_models --config configs/auxiliary_training.yaml --stage task
python -m training.train_models --config configs/auxiliary_training.yaml --stage task-eval
python -m training.build_examples --config configs/auxiliary_training.yaml
python -m training.train_models --config configs/auxiliary_training.yaml --stage support
```

`build_examples` 使用固定 Qwen 服务生成支持关系标签。P1 采集将
`collection.task_asset` 设为实际训练后的 task 资产、`parent_support_asset` 设为实际 support 资产，
按同一 manifest 分别采集 train/dev；查询资产身份随奖励保存。

固定模型协议审计与四格比较示例：

```bash
python -m training.reader_audit export --config ACTUAL-AUDIT.yaml
python -m training.reader_audit review --config ACTUAL-AUDIT.yaml
python -m training.a_workflow compare --comparison-config COMPARISON.yaml \
  --policy-asset CURRENT-P1-ASSET --matrix MATRIX.yaml --dataset DATASET \
  --split dev --phase both --output NEW-OUTPUT/comparison
```

所有实验按运行前声明的 cohort、reader 身份和资源范围读取结果。质量未测得时保留未知，
完成子集另外报告；输入投影成功率、预测成本和容量观测不能替代正式答案评分。
旧 Qwen LoRA/合并模型已退出活动入口，既往已付费用列为弃用支出，不进入受控改进结果。

正式主验收沿用 `study_matrix.yaml` 既有四数据集的 development_weights 和指标。
已从既有 `runs/study/cohorts.json` 固定 350 题、77 会话至 `runs/study/formal-dev-queue.matrix.json`，
运行及评分直接使用其中 IDs，再在相同清单上重跑 baseline 与候选；辅助学习的
200/40 划分不能替代正式开发队列。既有 a-public-dev 27 题、dynamic 14 题和 B 的
6 题仅为阶段参照，当前尚无统一主验收队列的完整实测参照。默认关闭大型消融 sweep，
保留 B0/Btoken/B3/B6/B7/Lmap 对照入口和 c1/c4 资源配置。
20 道 constructed long_dev 仅按参考正确性评分（correctness=1，归一化为 1）；
公开 Loong test 保留官方 loong_reference_score/100，二者通过 metric_by_split 明确区分。

C 的配对重放使用同一请求序列、完整 DAG 和至少 180 秒的观察窗。`replay` 与
`retention_pair_workflow` 默认固定 Native、TTL15、TTL60、TTL180 四臂；
`--horizon-seconds` 只决定共同观察长度，`--ttl-seconds` 只决定保护期限。
窗口选择要求原始轨迹中存在观察窗之后的真实请求边界，保留其间全部压力；不足时记录
不可观察，不改用更短窗口。每个窗口仅执行一次 Native，分别与三个 TTL 作差，不能把
三个对照行中的 Native 成本再次相加。

每臂的 `observation_window` 和物理/KV 积分覆盖相同长度，并持续采样到窗口结束，
包括无重访时的空闲驻留。`revisit-outcomes.json` 按实际原文因果页需求与原生回执
区分无重访、GPU 命中、Host 回载和重算；混合访问同时保留各类页数和事件数。
训练行中的 `observation_seconds` 是共同计费窗口，`ttl_seconds` / `horizon_seconds`
是策略保护期限。
