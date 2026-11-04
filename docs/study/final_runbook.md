# 当前课题运行入口

训练、模型推理与 GPU 测量按用户要求暂停。旧实验原始记录已丢弃；恢复运行后按本次配置重新采集答案、奖励及成本。

要求见 [任务书](task_spec.md)，机制见 [架构](../architecture.md)。以下命令描述实际入口。

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
`runs/study/auxiliary/cohorts/full/cohort_manifest.json`。
task/support 配置必填 `training.cohort_manifest`；P1 采集必填
`collection.cohort_manifest` 和 `cohort_split`，开发采集显式选择 `dev`。
B 新资产绑定 `infer_bridge_ready_or_one_gap_one_terminal_read_v3` 及 GAP 限额。反事实先执行首动作，优先使用已就绪终端证据，必要时一次 GAP，随后至多一次终端 READ、一次 ANSWER；直接回答分支不续接。线上执行一个动作后重新观察。旧一步资产不用于这个监督目标，仅重新采集目标实际变化的状态。

B 的 observed 资产另绑定 `observed_feature_schema=pre_action_budget_support_state_v2`，使用实际已付资源和剩余额度；旧特征资产不能直接用于该决策。

GAP 离线与线上只排除实际已读来源，统一新增 passage 额度；采集和拟合必须绑定相同协议。

B 固定流程观测使用 `execution.state_observation`，从真实请求开始计量并记录 READ/REDUCE 决策状态；
不加载 gain 模型，不从旧 SQLite 补造物理状态。A 冻结后显式启用。
四格固定历史入口设置 `policy_comparison.audited_history:{cohort_manifest,cohort_split,actual_rollouts}`，
各规划器对同一审核任务与真实 QA 执行自己的检索。P1 source-first 格可同时写 B 观测；原候选池重放入口另留作隔离对照。
`training.collect_plan_gains --observations OBS --actual-turns TURNS --output PACKETS`
导出后，`training.plan_gain_workflow --mode observed_actions --observed-states PACKETS`
沿用源 cohort 身份。每个源 task 保留各合法动作首次可配对的观测，不按未来质量选点。
真实检索后的初始状态也可配对 DIRECT 与首次 READ；无需先完成一次 READ。
进度分别记录去重源任务数和决策快照数，不将多个动作快照重复计为训练任务。
`observed_actions` 沿用源动作的输出份额，不扫描 `read-outputs`；有效观测数单独记录，声明分母仍为 200/40。
新成本拟合读取各分支 `run.json.pre_action_quote`，质量与成本分别归一化；`cold` 数据只能说明冷启动费用，`observed` 保留实际分支起点，不声称恢复源缓存。
B 的 `training.plan_gain_rollout`、`training.plan_gain_workflow` 均必填
`--cohort-manifest` 指向同一文件；不再使用 `--partition-seed` 或 `--dev-fraction`
重新划分状态，实际采集数量从声明分区内选择并单独记录。
模型 scope 位于 `configs/model_scopes/`。Reader、策略身份、来源与费用由本次阶段
manifest 记录。

机制重放示例（显式指定实际服务配置和新输出目录）：

```bash
python -m bench.reading_mechanisms --config ACTUAL-RUN.yaml \
  --read-layout source_first --cases pair correction long multihop \
  --output-dir NEW-OUTPUT/shared
python -m bench.reading_mechanisms --config ACTUAL-RUN.yaml \
  --read-layout source_first --numeric-need-isolation --cases long \
  --output-dir NEW-OUTPUT/isolated
```

辅助训练配置为 `configs/auxiliary_training.yaml`；BGE 资产按配置复用，标签由以下入口重新准备。
先准备原始任务，再采集实际生成历史、声明会话隔离的 cohort、生成教师标注配置，最后标注和训练。以下命令供恢复 GPU 工作后执行，当前不运行：

```bash
python -m training.a_workflow prepare-data --config configs/auxiliary_training.yaml \
  --output runs/study/data
python -m training.task_rollout --config configs/auxiliary_training.yaml \
  --min-conversations 40 --min-turns 300 --concurrency 4
python -m training.a_workflow cohort --config configs/auxiliary_training.yaml \
  --output runs/study/auxiliary/cohorts
python -m training.a_workflow prepare-labels --config configs/auxiliary_training.yaml \
  --output runs/study/auxiliary/task-labels
python -m training.task_labeling --config runs/study/auxiliary/task-labels/task_labeling.yaml
```

采集量是示例，cohort manifest 记录实际数量与 shortfall，不预先宣称达到 200/40。
`task_cohort_labeling.existing_labels_path: null` 表示从零标注；只有存在可复用的真实标签时才指定文件。
当前配置 `task_model.initial_adapter_path: null` 与 `training.support_parent_asset: null` 从显式 BGE base 初始化；设为非空父路径时只复用真实存在的资产，不自动回退。有效学生与支持资产须由本次训练及开发评价确认。
配置默认将实际历史写入 `runs/study/actual-rollout/sessions.sqlite3`。训练与评价共用
`runs/study/auxiliary/task-labels/task_labels.jsonl`，由 cohort manifest 隔离 train/dev。

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

当前辅助配置将新 support 输出接为 P1 父资产与 P0 对照，将新 task 输出同时接入 P1 奖励采集和比较的查询编码器。初始化真实历史的辅助 rollout 使用显式无查询 adapter 的 BGE 基座；若采用已有资产，修改对应显式路径后重新绑定采集身份。

已有候选答案的纯 CPU 诊断随 `python -m training.train_evidence_policy prepare --config CONFIG.yaml` 写入 `preference_manifest.json.candidate_quality_diagnostics`。该命令只读取已生成答案、评分和成本资产，按真实历史/预算/reader/scorer/query 分组，给出候选质量上限、最高预测并列时的遗憾区间和声明任务缺失列表，不产生模型训练或推理。输入资产不可用时不产生诊断结果。

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

正式主验收沿用 `study_matrix.yaml` 既有四数据集的 development_weights 和指标。
正式开发队列为 `runs/study/formal-dev-queue.matrix.json`，声明 350 题、77 会话；
运行及评分直接使用其中 IDs，再在相同清单上重跑 baseline 与候选；辅助学习的
200/40 划分用于辅助训练，不能替代正式开发队列。
冻结 A 采用 P0 共享协议；模型、支持策略、查询资产、阅读协议与预算由
本次冻结 A 的 manifest 及实际运行配置绑定。
P1 效用输入使用零上下文 `evidence_set_only`，不使用实际 READ 后长度。
B 从冻结 A 的真实观察采集动作分支；失败尝试中已完成步骤的观察通过真实 Attempt ID
进入配对，capture 最终状态仅作出处，不作为模型特征或未来质量筛选条件。

默认关闭大型消融 sweep，
保留 B0/Btoken/B3/B6/B7/Lmap 对照入口和 c1/c4 资源配置。
20 道 constructed long_dev 使用非官方 `constructed_numeric_exact_v1`：模型仅从实际问题和答案
抽取最终计数、总额、单位与一致性，不接触 gold；代码按整数、Decimal 和 credits 单位精确比较，
计数与总额同时匹配才得 1 分；
公开 Loong test 保留官方 loong_reference_score/100，二者通过 metric_by_split 明确区分。

C 使用 `paired_native_retention_v5` 及 `target_demand_rule=exact_complete_causal_prefix_and_body_ranges_v1`，旧标签不能直接重命名为新目标。训练行必须保存独立会话集合，train/dev 集合互斥；同窗口各 TTL 及共享会话的窗口为重复测量。粗特征为访问状态、原生驻留、压力与并发组合。只有相同完整来源需求重访才标注整次下一 READ 的 target 收益；部分页复用仍记录在原生观察与整窗成本中。新增 owner 的目标收益按完整来源需求去重，容量与驻留按共享页增量；配对外部请求代价进入选择，正 collateral 不再额外记为可叠加收益。完整集合的真实净效果仍由同轨 Native/value 对照和物理积分验收。

C 的两次容量构造共用原有 TTL15/60/180、owner、页并集和需求去重目标，不新增 TTL 网格或原生接口。

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
候选前缀必须含完整来源正文页；纯系统或元数据前缀不作为“无重访”训练样本。
单前缀配对分别积分 ready 页和传输目标已分配空间：H2D 计入 GPU，D2H 计入 Host。
`profiling.retention_trace reassess --replay-root COMPLETED_REPLAY` 从已完成重放的原始回执重算派生费用，
保留原请求、采样窗口和全部已付成本，不产生模型调用。

冻结 A 后，联合开发验收使用 `bench.capacity_workflow --full-queue`。主负载 c4 默认执行 E0—E4，c1 默认只执行 E1/E4。E0_native 显式保留 PR3 原行分片、task-first、隔离数值和 Native；P0 支持/查询资产与公共正确性代码共用。
必须提供 `--arrival-plan`：JSON 索引格式为 `{"format":"contextkv_dataset_workload_index_v1","datasets":{"DATASET":"/absolute/fixed-arrivals.json"}}`，每个数据集严格覆盖声明队列。
使用已有请求轨迹离线导出固定相对到达（不产生模型调用）：

```sh
python -m bench.study_workloads --tasks TASKS.jsonl --records RECORDED-REQUESTS.jsonl \
  --output FIXED-ARRIVALS.json --name DECLARED-NAME --concurrency 4
```

旧请求记录已丢弃时，直接冻结新声明的共同到达映射，无需模型调用：

```sh
python -m bench.study_workloads --tasks TASKS.jsonl --arrivals DECLARED-ARRIVALS.json \
  --output FIXED-ARRIVALS.json --name DECLARED-NAME --concurrency 4
```

`DECLARED-ARRIVALS.json` 必须显式列出每道任务的相对到达秒数，例如 `{"session:1":0,"session:2":30}`；该示例不是已选定的正式负载。保存任务与到达映射哈希，E0—E4 复用同一声明；原生观测仍由未来真实执行产生。

来源记录必须包含所有声明任务的请求开始时间；未记录的上游到达不能反推或补成零。各方案复用同一固定计划及哈希，按各自真实 P3 历史完成依赖；等待前一轮和等待并发槽均计入完整响应。
`--complete-slo-seconds` 设置事先声明的完整响应 SLO，goodput 按达标完整回答/观察秒数计算，无阈值时不生成 SLO 数值。

```sh
python -m bench.capacity_workflow --full-queue \
  --config "$FROZEN_A_STUDY" --matrix "$JOINT_MATRIX" --arrival-plan "$ARRIVAL_PLAN" \
  --gain-asset "$B_GAIN" --online-gpu-devices "$ONLINE_GPUS" --reader-base-url "$READER_URL" \
  --value-model "$C_VALUE" --concurrency 4 --output "$JOINT_OUTPUT"
```

E1—E4 四格共用冻结 A 的模型、支持策略、查询资产、阅读协议和预算；E0_native 仅覆盖登记基线的阅读与来源协议。可用
`--methods B6_native` 等参数将指定方法分配到独立 reader 服务，只生成所选方法的矩阵和任务。
`B6_native` 无需 gain/value；`B6_value` 仅需 `--value-model`；没有 S3 方法时须显式提供
`--reader-profile-id`。S3 方法需要 `--gain-asset`，value 方法需要 `--value-model`。
默认 c4 执行 E0_native/B6_native/S3_native/B6_value/S3_value 并要求两种资产；c1 默认只执行 B6_native/S3_value，每格使用同一完整队列。
固定 B 可独立执行，例如：

```sh
python -m bench.capacity_workflow --full-queue --methods B6_native \
  --config "$FROZEN_A_STUDY" --matrix "$JOINT_MATRIX" --arrival-plan "$ARRIVAL_PLAN" \
  --reader-profile-id "$READER_PROFILE" --reader-base-url "$READER_URL" \
  --online-gpu-devices "$ONLINE_GPUS" --concurrency 1 4 --output "$JOINT_OUTPUT"
```

收益 B 的 reader 身份由实际运行配置与实测收益资产绑定。`--online-gpu-devices` 接受逗号分隔的物理卡号。
可通过 `--online-process-roots` 指定逗号分隔的 PID，默认按 reader 和 helper 的端口发现
进程并加入协调进程。`--reader-base-url` 显式指定实际服务端点。联合矩阵沿用正式 350 题清单，
其中检索器设备配置与本次物理卡号一致。GAP 限额沿用冻结观测配置，
缺省为 96 个单元和 10 个 passage，可用 `--gap-max-units`、`--gap-max-passages` 显式指定。
`--lambda-time` 覆盖冻结配置中的动作成本权重，实际值写入 `capacity-plan.json`。
分设备结果用 `profiling.final_study_index --baseline-commit "$E0_SOURCE_COMMIT" --joint-scope joint-frozen-A/inputs.json`
汇总，逐个 `--artifact NAME=WORKFLOW.json` 指定各设备结果。共同输入、实际 worker 配置、
cohort、reader 与预算确定对照条件，各格分别列出实际启用的 B/C 资产。
报告保留全分母质量、完成率、已交付完整响应 P95、逐数据集 SLO goodput 与成本出处。比较身份同时核对到达计划哈希和 SLO；提供 `--cost-manifest`/`--physical` 生成成本命令时，必须通过 `--gpu-devices` 声明实际设备，不默认包含整机。

联合汇总的 `dataset_protocols` 来自矩阵已有 `split_kind` 与分区指标声明；Markdown 将公开基准、参考历史诊断和构造机制任务分表。每轨原指标、完整题目分母、已评分子集与未知区间同时呈现，混合开发 Q 的登记权重保持不变。

联合 Markdown 对照表直接呈现完整分母质量差与 P95 相对变化（候选/基线−1，负值为降低），并列出候选质量是否不低于基线。相对延迟只在比较身份一致、双方完整交付且全部已评分时提供；质量判断为点估计，资源变化仍读取实际成本账本。

每格按自己的 P3 历史执行原 350 题，使用声明的开发指标与权重。未交付回答计入
完整分母并贡献零分，已交付但未评分的回答保留质量区间。报告另列原生已付调用
索引和整批物理成本账本。评分依赖沿用冻结配置。`--source-manifest` 用于原有两轮容量诊断。

C 的离线 `retention_trace export` 需显式提供 `--failed-conversations`（零失败为 `[]`）。
同测量段内未导出的成功与失败请求墙钟区间进入压力索引，回放不使用与之重叠的窗口。
`profiling.retention_coverage --replay-root C_OUTPUT --output COVERAGE.json` 只读已有回执，
分别汇总干预目标与整窗重复来源页的 GPU 命中、Host 回载和重算，不产生模型调用。

B 的 DIRECT/READ/REDUCE/GAP 分支从实际动作前状态报价，分别拟合绝对分支成本，
同一批次以动作减 DIRECT 计算增量成本和实际质量差，已付共享成本只计一次。
REDUCE 只读已观察材料；源 packet 的 legal_actions 保留，补充的反事实候选单独登记并在执行前重新报价。

C 固定 DAG 回放解析原始输出文本，新的生成响应用于实际 token、耗时和费用；完整窗口
预测用两臂窗口成本差验证，下一次 READ 的集合收益按事件及共享页去重。
