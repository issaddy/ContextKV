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
四格固定历史入口设置 `policy_comparison.audited_history:{cohort_manifest,cohort_split,actual_rollouts}`，
各规划器对同一审核任务与真实 QA 执行自己的检索。P1 source-first 格可同时写 B 观测；原候选池重放入口另留作隔离对照。
`training.collect_plan_gains --observations OBS --actual-turns TURNS --output PACKETS`
导出后，`training.plan_gain_workflow --mode observed_actions --observed-states PACKETS`
沿用源 cohort 身份。每个源 task 保留各合法动作首次可配对的观测，不按未来质量选点。
`observed_actions` 沿用源动作的输出份额，不扫描 `read-outputs`；当前 178/36 个可调度状态的生成预算上限设为 3041，声明分母仍为 200/40。
新成本拟合读取各分支 `run.json.pre_action_quote`，质量与成本分别归一化；`cold` 数据只能说明冷启动费用，`observed` 保留实际分支起点，不声称恢复源缓存。
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
200/40 划分用于辅助训练。正式 c4 参照已评分，350 题交付 300 个答案，全队列
Q 为 0.44509019347880296。报告位于
`runs/study/budget-loop-20261103/formal-dev-c4/reference-final/report.json`。

| 开发轨道 | 答案 / 声明题数 | 全队列 Q | 权重 |
|---|---:|---:|---:|
| MultiDoc2Dial | 183 / 209 | 0.5822966507177033 | 0.30 |
| MTRAG | 41 / 41 | 0.32660479305396806 | 0.25 |
| VersionQA | 76 / 80 | 0.94375 | 0.20 |
| Constructed long | 0 / 20 | 0 | 0.25 |

固定历史四格均按声明的 40 题计算 RB_llm。P0 隔离 32 答、0.605，共享 34 答、0.610。
旧 P1 隔离 36 答、0.5175，共享 35 答、0.4325。旧 P1 已被拒绝。修正后 P1 共享格交付 36/40、Q 为 0.5325，仍低于 P0 的 0.610；
已选择并冻结 P0 共享协议进入 B/C。
效用训练已去除实际 READ 后长度带来的输入泄漏，训练和推理统一使用零上下文
`evidence_set_only`。新资产 `models/study-support-fixed-qwen-set-only` 沿用 1400 对偏好、
147 个状态和 3 个 epoch，已完成训练，完整阶段耗时 1485.07 秒；40 题验证已完成。旧 B capture 完成的 115 个状态保留为
旧模型的成本记录，不用于新模型训练。B 小批次配对已完成，完整状态配对待开发观测采集结束后执行。
`selected-a-followthrough/selection.json` 已记录 P0 的冻结资产、配置与选择证据；
P0 train 观测已完成 178 个状态：173 个交付答案、5 个失败，共 590 条实际观察；失败前的有效观察也进入配对。
dev 观测仍在采集，随后执行 B 配对。修正后 P1 的隔离协议固定历史格在 B 之后执行。

`constructed_numeric_exact_v1` 已重评 21 个已有长文答案。抽取器只看到实际问题和答案，
由代码精确比较整数计数、Decimal 总额和单位。正式筛选前长文运行交付 16/20 答案，
7 题正确，全分母 Q 为 0.35。新 per-Need 长文运行已完成生成与评分，
见 `runs/study/budget-loop-20261103/numeric-selected-p3-c4`。20 题交付 15 个答案，
11 题计数与总额均精确匹配，全分母 Q 为 0.55。五次失败包括 1 次 READ 输出截断、
3 次单位不兼容和 1 次量值冲突。case002 的最终答案为 35 项、595 credits，仍有误。
冻结旧 READ 的单 Need 局部重放曾得到正确的 630 credits 和 35 个父结论。
P0 共享协议的 MultiDoc2Dial 已评分，交付 205/209 个答案，全队列 Q 为 0.6626794258373205，
参照为 0.5822966507177033。该批次实际加载 607e02d，含一个数值 Need 的筛选步骤；
其消息早于后续的通用单位提示修正，协议审计见 `formal-dev-c4-grounded/P0-multidoc2dial-protocol-audit.json`。
MTRAG 已评分 41/41，完整队列 Q 为 0.5074440554339312，参照为 0.32660479305396806。
最终联合实验采用冻结后的协议。

原 350 题参照维持 300 个答案、Q 为 0.44509019347880296。参照的原生账本已核实
2487 次调用和 11941836 token。另有 1 条 PLANNED 未提交记录，其模型 token 为 0，
前置物理成本仍计入。本阶段累计清理 3289295076 字节重复产物，原始账本保留。
P0 共享候选已完成完整 350 题对照：341 答、Q 为 0.6575398416096789，较参照提高
21.2449648130876 个百分点；包含失败步骤的阅读器原生 token 为 7520143，减少 37.0269%。
两臂的窗口、历史、输出、累计步骤和 token 配额一致。报告为
`formal-dev-c4-grounded/P0-complete-queue-report/report.json`。这是冻结前 P0 的完整队列结果；
阶段质量指标已超过 5 个百分点，最终 A/B/C 联合实验及完整 P95 验收仍待完成。

默认关闭大型消融 sweep，
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

冻结 A 后，联合开发验收使用 `bench.capacity_workflow --full-queue`。

```sh
python -m bench.capacity_workflow --full-queue \
  --config "$FROZEN_A_STUDY" --matrix "$JOINT_MATRIX" \
  --gain-asset "$B_GAIN" --online-gpu-devices "$ONLINE_GPUS" --reader-base-url "$READER_URL" \
  --value-model "$C_VALUE" --concurrency 1 4 --output "$JOINT_OUTPUT"
```

完整队列四格共用冻结 A 的模型、支持策略、查询资产、阅读协议和预算。可用
`--methods B6_native` 等参数将指定方法分配到独立 reader 服务。默认执行四格，每格使用同一完整队列。
收益 B 的 reader 身份由实际运行配置与实测收益资产绑定。`--online-gpu-devices` 接受逗号分隔的物理卡号。
可通过 `--online-process-roots` 指定逗号分隔的 PID，默认按 reader 和 helper 的端口发现
进程并加入协调进程。`--reader-base-url` 显式指定实际服务端点。联合矩阵沿用正式 350 题清单，
其中检索器设备配置与本次物理卡号一致。GAP 限额沿用冻结观测配置，
缺省为 96 个单元和 10 个 passage，可用 `--gap-max-units`、`--gap-max-passages` 显式指定。
`--lambda-time` 覆盖冻结配置中的动作成本权重，实际值写入 `capacity-plan.json`。
分设备结果用 `profiling.final_study_index --joint-scope joint-frozen-A/inputs.json`
汇总，逐个 `--artifact NAME=WORKFLOW.json` 指定各设备结果。共同输入、实际 worker 配置、
cohort、reader 和 B/C 资产身份进入对照，报告保留全分母质量、完成率、已交付完整响应 P95 与成本出处。

每格按自己的 P3 历史执行原 350 题，使用声明的开发指标与权重。未交付回答计入
完整分母并贡献零分，已交付但未评分的回答保留质量区间。报告另列原生已付调用
索引和整批物理成本账本。评分依赖沿用冻结配置。`--source-manifest` 用于原有两轮容量诊断。

C 的离线 `retention_trace export` 需显式提供 `--failed-conversations`（零失败为 `[]`）。
同测量段内未导出的成功与失败请求墙钟区间进入压力索引，回放不使用与之重叠的窗口。

C 已在完成隔离对照的 GPU4 reader 上启动 train/dev 各两窗，共计划 2388 次真实生成；`retention-preparation/paired-plan.json`
记录同一180秒观察窗、Native/TTL15/60/180四臂及运行资源。训练轨迹来自已有真实证据候选，
开发轨迹来自P0自身P3；其与最终冻结A的联合队列分别计量。训练轨迹中624次READ输入
已按当前协议重新编译，全部与原始输入hash一致，审计未调用生成模型。

B 已完成最早两个训练状态的小批次：5 个回答分支、3 组实际配对（6 次 reader、5 次 judge），
补检索的一组质量差为 +0.16，其余两组为0。原始记录在 `b-selected-a-pilot`。
当前动作协议允许对至少两条 lookup/procedure 原事实做一次可选 summarize；归约仍只读已观察材料。
源 packet 的 legal_actions 保留，采集清单另记本次补充的反事实候选，实际执行前重新报价。

补充REDUCE后的两题采集已完成6个回答分支、4组配对（8次reader、6次judge）。
本次REDUCE质量差为+0.24，GAP两组为+0.4/0，READ为0，均保留实际新增成本。
正式B采集保留200/40声明分母，当前train有178个审核可用状态。失败尝试中的有效观察
通过真实Attempt ID进入配对，capture最终状态仅作出处。
C固定DAG回放解析原始输出文本，新的生成响应只用于实际token、耗时和费用；完整窗口
预测用两臂窗口成本差验证，下一次READ的集合收益继续按事件及共享页去重。
