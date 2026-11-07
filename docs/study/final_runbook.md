# 当前课题运行入口

`test-0523` 使用容器默认 `python3`，Qwen3-32B 权重保持固定。主链已接通；当前工作是补齐 B 的 GAP 监督，并完成冻结 ranked A 的联合比较。完整 350 题质量与 P95 尚未验收。

正式运行位于 `runs/study-20261106-resume/final-joint-6888cf8-001`，代码冻结为 `6888cf8`。固定 B 的 c1、c4 两组已启动，各依次运行 Native、ContextPilot 和 LlamaIndex Native；分别使用 GPU 0/1、3/5，每组一张 reader 卡和一张 helper＋BGE 卡。收益 B 两组使用同样的两卡配额、350 题／77 会话、到达序列和生成条件，待监督补齐后启动。`inputs/inputs.json`、各组 `capacity-plan.json` 记录实际配置；模型和大日志留在容器。

长文共享提取已得到 36 条、666 kg，多跳得到 1997。旧中断参照的 64 个答案已完成官方重评，完整声明分母与费用保留。

Host 固定 token 配对已验证真实回载与重算：同一请求下，Native 回载 128、重算 129 tokens；独立前缀保留回载 256、重算 1 token。业务 epoch 退役后 owner 仍保留 Host 前缀。该结果验证生命周期，主实验保留策略仍为 Native；成功与初次接口失败的费用分别保存。

已完成的 c1 MultiDoc2Dial：support A＋固定 B＋Native 交付 196/209，完整分母 RB_llm 为 0.688038；LlamaIndex Native 交付 209/209，为 0.806220。两者 MT-RAG 各 41 个答案已完整评分，harmonic 分别为 0.524363、0.557842。DeBERTa 位置上限适配后复用既有答案补齐 algorithmic；两项截断的 RAGAS 输出另以 4096 上限补齐，新旧 NLI 输入相同、旧文本为新输出前缀，其余分量未改。原调用和补齐费用分别保留。新批次显式设置 `scoring.ragas_json_schema: true`、`scoring.ragas_max_tokens: 4096`，只作用于 RAGAS 结构输出，RB 仍为 2048。c1 尚未运行的 versionqa/loong 和外部 c4 仍须补齐。

30 个预声明、空历史相同的首轮题已完成 ranked 固定输入对照：RB_llm 为 0.846667，原 support 为 0.776667（12 胜、13 平、5 负），与 LlamaIndex 均分相同。对照通过 `bench.evidence_policy_comparison frozen --cells P0_source_first_shared` 仅设置 `selection.mode: ranked`，原记录池、TaskState、预算和历史逐条一致，两种选材均由固定 B 直接 ANSWER。该结果不包含初始检索，不能当作完整 P3 或端到端提速。30 个请求费用窗口完整；批次另按真实末采样时刻报告可观测积分，后续代码已修正结束戳与停止采样的顺序。

ranked 的 c1 MultiDoc2Dial 完整 P3 已交付并评分 209/209，RB_llm 为 0.788038，比原 support 的完整分母结果高 0.10；LlamaIndex 为 0.806220。两次项目方法的在线批次物理窗口均完整：ranked 3727.844 秒、618329.213 GPU GiB·s、172118.183 Host GiB·s；support 14022.922 秒、2315683.243 GPU GiB·s、617252.311 Host GiB·s，包含其失败请求。这里报告的是该公开轨道的整批成本，完整 350 题目标尚未验收。

旧 support A 的 ContextPilot 队列、support Native 与 ranked 资格比较队列均已结束扩跑；原答案、失败和完整已付费用保留在各自运行目录。这些历史配置单列，不并入当前正式四格。

当前实现已补齐来源版本过滤、必需数值归约、选择报价到执行的传递，以及直接 ANSWER 的来源编排。Native 对尚未分配 KV 的排队请求也已正确报价；原失败费用保留。

最新 READ 提示合并重复事实引用，清单逐项保留必要限定；`union/group` 只去重完整相同的事实。一次原截断步骤回放保留原来源、顺序、schema、采样和 2048 上限，输出由 length/2048 变为 stop/1315，解析出 13 条事实；稳定来源消息与任务 payload 均相同。

ranked B 的小批次已完成：声明 8/4 个源任务，实际历史 48/24 轮。初始已知超窗动作从拟合中排除，原失败费用保留；复用 11 个真实 READ 后状态，合并得到 18/10 个训练／开发配对，分别来自 14/8 个状态和 7/4 个源任务。当前训练含 READ、REDUCE，开发集还有 1 个未训练 GAP；预测覆盖为 9/10，已覆盖动作 MAE 为 0.157019，整体 MAE 留空。原生分支未重跑；旧工作流只因开发评估遇到未训练 GAP 而报错的部分已用 CPU 补齐。数据、来源及成本摘要在 `runs/study-20261106-resume/b-ranked-3abfde2/summary.json`，模型仍留在实验容器。

GAP 补充采集位于 `b-ranked-gap-native-4t3`：按原始训练问题的跨轮指代和来源切换预选 4 个第三轮目标，共 12 个真实 P3 轮次。原开发会话保持隔离；是否形成 GAP 监督由实际观测决定，不构造缺口或修改问题。

要求见 [任务书](task_spec.md)，机制见 [架构](../architecture.md)。以下命令描述实际入口。

后续运行使用容器默认 `python3`，不再新建虚拟环境；此前批次保留其真实解释器记录。跨机器仅同步代码、配置和轻量结果／成本记录，训练模型及大日志留在实验容器。

运行冻结同一 Qwen3-32B 模型权重与解码参数，显式配置端点、协议身份与原生缓存。主机目录、GPU、服务端口和
输出目录由本次运行配置声明；不复用旧批次的固定路径或截止时间。

本轮运行链：`DocumentQA.ask → StudyEvidenceProvider → Controller → RequestCompiler/ReadingExecutor → SessionStore`。API 的 `/documents` 登记资料，`/sessions/{id}/query` 用 `document_set_id` 引用；`bench.study` 共用 ask。新资料的编码和排序在实际检索阶段计费。

引擎准备复用容器当前 Python/CUDA，不安装或替换 PyTorch：

```bash
export CONTEXTKV_MODEL_SCOPE_PATH=/absolute/path/qwen3-scope.json
bash scripts/bootstrap_sglang.sh
source runs/engine-runtime.env
CUDA_VISIBLE_DEVICES=0 bash scripts/launch_sglang.sh --host 127.0.0.1 --port 30000
```

scope 的 `model_path` 指向 `/data/models/qwen3-32b`，模型/tokenizer 身份沿用真实文件。默认集成为官方 SGLang 0.5.18 源码及项目补丁，CUDA 内核来自现有环境。`engine-source.json`、`engine-environment.txt` 记录实际版本；不要求升级主机驱动。

本次使用 `source runs/study-20261106-resume/default-python/runtime.env`。默认解释器为 `/opt/sglang/bin/python3`；`CUDA_HOME` 使用该运行目录内的 `cuda12-toolkit/root/usr/local/cuda-12.9`，`LD_PRELOAD` 选择实际挂载的 `libcuda.so.560.35.03`，FlashInfer 与 SGLang JIT 使用对应 cu129 工具链。BF16/FA3 显式设置上游 `SGLANG_ENABLE_JIT_DEEPGEMM=0`，补丁使该开关在可选扩展导入前生效；共享 DeepGEMM 包未卸载。`NO_PROXY` 覆盖本机服务地址。旧虚拟环境路径只属于之前批次，当前服务及采集均已使用默认解释器。

BGE 检索、发布重排器及历史编码器在本环境显式使用 `sdpa_backend: math`；对应配置分别位于 retriever YAML 和 `task_model`。两套索引的语料、向量哈希及离线费用见 `runs/study-20261106-resume/offline-index-build`。此后端选择不改变 Qwen 权重或推理设置。

作者 ContextPilot 准备与启用：

```bash
python3 scripts/prepare_contextpilot.py --checkout vendor/contextpilot
```

study 配置的 `execution.cache_compilation` 取 `{mode: contextpilot, checkout: vendor/contextpilot}`；Native 对照取 `{mode: native}`。`prefix_retention` 默认 `{mode: native}`，与编排开关独立。

LlamaIndex 的统一执行入口与最终编译矩阵已接线；外部参照实测和完整队列比较仍须完成。首次外部运行的 4 次尝试在来源映射处失败，未生成答案，费用保留在 `external-contextpilot-fixed-001`。映射已按上游实际渲染保留的字符修正：正文尾空白被上游移除时，来源结束坐标相应缩短，仍逐字核对原文。B 恢复沿用原 SQLite 和已完成问答，通过已有 `--request-namespace` 为新尝试分配 ID；旧失败记录保存在原测量段，实际历史与已付成本继续保留。导出时保留 `measurements/<segment>/generation_manifest.json`，模型身份由实际捕获资产派生，原 digest 和来源 hash 留在 `capture_resident_binding`。下表的入口存在不表示对应实验已完成。

| 任务 | 当前入口 | 必须保留的对照与产物 |
|---|---|---|
| 结论预算、共享数值与有限归约 | `bench.reading_mechanisms` | 共享阅读 / `--numeric-need-isolation`；实际 messages、完整来源叶子与阶段账本 |
| 原生来源前缀机制 | `bench.prefix_read_mechanisms` | 实际 READ epoch、前缀身份与 owner 观察 |
| 阅读协议审计与会话队列 | `training.reader_audit` 的 `export`、`review`；`training.a_workflow` 的 `cohort`、`prepare-labels` | 实际 messages 与完整原文独立核验，无 SFT 输出 |
| P0/P1 规划器 × 阅读协议 | `training.a_workflow compare`、`bench.evidence_policy_comparison` | task_first+隔离数值 / source_first+共享提取；固定实际输入四格与各自真实 P3 |
| 证据策略真实候选收益 | `training.preference_rollout` | 实际最终回答、官方质量与完整成本；绑定当前 reader |
| B 动作收益 | `training.plan_gain_rollout`、`training.plan_gain_completion`、`training.collect_plan_gains` | DIRECT/READ/REDUCE/GAP 的真实增量执行、当前完整状态、独立最终答案与成对费用 |
| B 训练与滚动比较 | `training.plan_gain_workflow`、`bench.rolling_study_workflow` | 显式动作与 `whole_plan` 固定计划对照分别登记；保留已付成本与绝对分支预算 |
| C 固定轨迹配对 | `profiling.retention_trace` | 同模型、同输入、同 DAG 下 Native/固定 TTL/学习保护；共享页与 idle 驻留成本 |
| 完整联合与外部比较 | `bench.capacity_workflow --full-queue` | c1/c4 编译四格、LlamaIndex Native 参照、跨轮事实复用消融；固定到达、连续设备/进程采样与全部在线费用 |
| 官方评分与离线重评 | `bench.study_scoring`、`bench.reassess_saved` | 全声明题目分母，失败、未完成、未评分及其已付费用 |
| 全成本账本 | `profiling.study_cost_ledger` | 原始调用/采样引用、窗口并集、共享份额与未知覆盖 |

配置入口为 `configs/study.yaml`、`study_matrix.yaml`、`evidence_policy_comparison.yaml`；
P1 支持策略配置为 `configs/evidence_policy_fixed_qwen.yaml`。
`bench.study prepare` 的公开 VersionQA 资产从 `assets.versionqa.source/questions` 指定的原文和 CSV 构建，复用 `bench.versionqa` 与 BM25/BGE 索引入口，不再引用历史运行目录；其公开测试题与构造机制开发题分别登记。
审计配置为 `configs/reader_audit.yaml` 的 `reader_audit:{base_scope,data,review,resource_sample_seconds}`。
会话队列使用 `evidence_cohort`；P1 支持策略直接绑定固定生成配置。
辅助学习配置中的 200 train / 40 dev 规模使用
`runs/study/auxiliary/cohorts/full/cohort_manifest.json`；实际运行指向本批预声明的 cohort。
task/support 配置必填 `training.cohort_manifest`；P1 采集必填
`collection.cohort_manifest` 和 `cohort_split`，开发采集显式选择 `dev`。
B 新资产绑定 `infer_bridge_ready_or_one_gap_one_terminal_read_v4` 及 GAP 限额。反事实先执行首动作，优先使用已就绪终端证据，必要时一次 GAP，随后至多一次终端 READ、一次 ANSWER；直接回答分支不续接。线上与反事实 READ 共用相关父结论绑定，线上执行一个动作后重新观察。成本资产同时记录 `prefix_control_scope=native_excludes_prefix_control`，价值 C 的首步控制预测另加一次。当前协议的监督使用真实 P3 状态重新配对采集。

B 的 observed 资产另绑定 `observed_feature_schema=pre_action_budget_support_state_v2`，使用实际已付资源和剩余额度；旧特征资产不能直接用于该决策。

当前 B 资产还必须携带 `input_protocol`：`merged_source_windows_actual_coverage_reused_facts_compiled_choice`、实际窗口粒度和完整 reader fingerprint。采集、分支、配对及加载全链核对该身份；旧资产缺少绑定时须按当前协议重采/重拟合，不改名复用。实际 seed、已读覆盖、编译选中排列及报价的 `pre_action_input` 一并保留；事实复用消融仍使用相同 B 模型，实际状态反映关闭复用。

动作收益资产的 `evidence_selection_contract` 来自实际 A manifest，绑定选材模式和顺序，以及 `support` 使用的效用、保留集合重排和支持模型记录身份。观测包和实际分支使用同一合同，开发集及线上提供器须与拟合资产匹配。语料、索引、B6/B7 和服务地址独立记录。只缺该合同的旧资产可用保留了实际分支 manifest 的配对记录重新拟合，旧观测包可从原 capture 重新导出。

GAP 离线与线上只排除实际已读来源，统一新增 passage 额度；采集和拟合必须绑定相同协议。

B 固定流程观测使用 `execution.state_observation`，从真实请求开始计量并记录 READ/REDUCE 决策状态；
不加载 gain 模型，不从旧 SQLite 补造物理状态。A 冻结后显式启用。
四格固定历史入口设置 `policy_comparison.audited_history:{cohort_manifest,cohort_split,actual_rollouts}`，
各规划器对同一审核任务与真实 QA 执行自己的检索。P1 source-first 格可同时写 B 观测；原候选池重放入口另留作隔离对照。
`training.collect_plan_gains --observations OBS --actual-turns TURNS --output PACKETS`
导出后，`training.plan_gain_workflow --mode observed_actions --observed-states PACKETS`
沿用源 cohort 身份。主线直接读取真实 P3 capture 的 TaskState/ReadingOutcome，不要求另行准备 `--task-labels`、task/student 或 P1 标签；`whole_plan` 固定历史对照仍需审计标签。cohort 预声明 task、会话与 train/dev 分区，真实 QA 历史在采集后由 snapshot 和已完成原生回执核验，只有固定历史对照才额外校验预先提供的历史 hash。每个源 task 保留各合法动作首次可配对的观测，不按未来质量选点。

已有实际 READ 分支可用 `training.collect_plan_gains --terminal-branches GAIN_ROOT --reader-config READER_YAML --output NEW_PACKETS` 导出终端状态。该入口使用原先保存的报价事件和物理样本切分已付成本，排除尚未发生的 ANSWER 费用；来源标为 `frozen_actual_rollout`，原 P3 与实际分支身份均保留。新配对沿用原 `actual-rollouts`、sources 和 cohort，使用新输出目录。
`bench.study` 的成功/失败记录保留原始 `reference`、原始数据分区和 observer cohort 绑定，仅供离线审计与评分。导出器将 packet 绑定到完整实际 capture 行的 hash；B 采集核验 native-train 原始正文、版本、窗口坐标与 raw grounding 引文，开发会话独立隔离。原始参考不会进入 A/B/C 的在线输入。正式 dev 队列不能作为 B 训练数据。
真实检索后的初始状态也可配对 DIRECT 与首次 READ。离线 DIRECT 在当前材料上强制 ANSWER，线上停止须通过实际覆盖或结论充分性检查。确认终止而未提交回答的真实分支计效用 0，包含生成前终止，原始评分保留为空。
进度分别记录去重源任务数和决策快照数，不将多个动作快照重复计为训练任务。
`observed_actions` 沿用源动作的输出份额，不扫描 `read-outputs`；有效观测数单独记录，声明分母来自所用 cohort。当前 B 小批次为 8/4 个源任务，48/24 个历史轮次不计作独立训练/开发状态。
新成本拟合读取各分支 `run.json.pre_action_quote`，质量与成本分别归一化；`cold` 数据只能说明冷启动费用，`observed` 保留实际分支起点，不声称恢复源缓存。
B 的 `training.plan_gain_rollout`、`training.plan_gain_workflow` 均必填
`--cohort-manifest` 指向同一文件；不再使用 `--partition-seed` 或 `--dev-fraction`
重新划分状态，实际采集数量从声明分区内选择并单独记录。
`plan_gain_workflow` 可将 `--train-states` 或 `--dev-states` 设为 0，只补所需分区；无有效训练标签时保留采集结果并跳过拟合，开发标签可随后独立补采。
模型 scope 位于 `configs/model_scopes/`。Reader、策略身份、来源与费用由本次阶段
manifest 记录。

B 收益的唯一默认质量目标为固定官方 RB_llm，使用与最终评价相同的真实对话和实际证据投影。`--official-checkout` 可指定已有官方源码目录，默认沿 study 的 `scoring.official_checkout`。`plan_gain_quality` 保存作者已归一化的 RB 值；不把它当作必要点覆盖率。P1 的 `necessary_points_grounding_conditions_versions_task_v3` 保留为独立四项诊断，旧 B 自定义四分结果仅作历史记录。

重评产生新标签和实际 scorer 身份，保留原 `before.json`、原始回答与执行成本，不回写原执行登记。实际观察到的评分条件误判随原文审计保留；不能据一次 judge 的分差声称正确性改进。

当前小批次资产位于 `runs/study-20261106-resume/b-pilot/gain-official-018b710/`：4 次既有官方调用经实际输入和采样核对后复用，新增 21 次评分，未重生成问答。拟合使用 9 个 train、4 个 dev 配对；训练覆盖 8 个 READ、1 个 GAP，REDUCE 无样本。开发保存候选上的模型选择为 2 次 DIRECT、2 次 READ，按同一实测质量与费用目标应为 3 次 DIRECT、1 次 READ；其中官方评分的条件误判另有原文审计。该诊断尚不能证明在线策略收益。`summary.json` 保存逐状态选择和费用，模型文件留在实验容器；后续比较绑定实际资产，旧自定义评分模型不部署。

机制重放示例（显式指定实际服务配置和新输出目录）：

```bash
python3 -m bench.reading_mechanisms --config ACTUAL-RUN.yaml \
  --read-layout source_first --cases pair correction long multihop \
  --output-dir NEW-OUTPUT/shared
python3 -m bench.reading_mechanisms --config ACTUAL-RUN.yaml \
  --read-layout source_first --numeric-need-isolation --cases long \
  --output-dir NEW-OUTPUT/isolated
```

辅助训练配置为 `configs/auxiliary_training.yaml`；BGE 资产按配置复用，标签由以下入口重新准备。
先准备原始任务，再采集实际生成历史、声明会话隔离的 cohort、生成教师标注配置，最后标注和训练。仅在对应任务或支持瓶颈需要辅助训练时执行：

```bash
python3 -m training.a_workflow prepare-data --config configs/auxiliary_training.yaml \
  --output runs/study/data
python3 -m training.task_rollout --config configs/auxiliary_training.yaml \
  --min-conversations 40 --min-turns 300 --concurrency 4
python3 -m training.a_workflow cohort --config configs/auxiliary_training.yaml \
  --output runs/study/auxiliary/cohorts
python3 -m training.a_workflow prepare-labels --config configs/auxiliary_training.yaml \
  --output runs/study/auxiliary/task-labels
python3 -m training.task_labeling --config runs/study/auxiliary/task-labels/task_labeling.yaml
```

采集量是示例，cohort manifest 记录实际数量与 shortfall，不预先宣称达到 200/40。
`task_cohort_labeling.existing_labels_path: null` 表示从零标注；只有存在可复用的真实标签时才指定文件。
实际 rollout 同时保存 `task_catalogue`、`task_reference_date` 和 `source_scope`，准备与标注直接使用这些观测，不从参考答案或金标补造版本目录。
`prepare-labels` 可在同一输出目录重跑，复用与当前状态一致的已有标签。`training.task_labeling --limit N` 每次最多新增 N 条，已完成条目不占该额度；全部完成后不再加载模型。
当前配置 `task_model.initial_adapter_path: null` 与 `training.support_parent_asset: null` 从显式 BGE base 初始化；设为非空父路径时只复用真实存在的资产，不自动回退。有效学生与支持资产须由本次训练及开发评价确认。
配置默认将实际历史写入 `runs/study/actual-rollout/sessions.sqlite3`。训练与评价共用
`runs/study/auxiliary/task-labels/task_labels.jsonl`，由 cohort manifest 隔离 train/dev。

运行时以 `CUDA_VISIBLE_DEVICES` 选择物理设备，配置内使用该进程的 `cuda:0`：

```bash
python3 -m training.train_models --config configs/auxiliary_training.yaml --stage task
python3 -m training.train_models --config configs/auxiliary_training.yaml --stage task-eval
python3 -m training.build_examples --config configs/auxiliary_training.yaml
python3 -m training.train_models --config configs/auxiliary_training.yaml --stage support
python3 -m training.train_models --config configs/auxiliary_training.yaml --stage support-eval
```

`support-eval` 读取显式 `support_eval_asset`、`support_eval_labels`、cohort 和输出目录，核对资产实际训练会话与开发会话隔离，报告支持标量误差、错误 complete、原文 span 选择及真实推理费用。这是独立会话上的教师标签一致性，原文正确性另审；不能据此替代完整问答质量。当前优先分析支持选择是否丢失有效材料，未启动支持学生训练。

`build_examples` 使用固定 Qwen 服务生成支持关系标签。P1 采集将
`collection.task_asset` 设为实际训练后的 task 资产、`parent_support_asset` 设为实际 support 资产，
按同一 manifest 分别采集 train/dev；查询资产身份随奖励保存。
P1 质量偏好可使用物理成本部分缺失的已评分答案；缺失费用保留未知，成本覆盖率单列，不将其作为实测成本收益。

当前辅助配置将新 support 输出接为 P1 父资产与 P0 对照，将新 task 输出同时接入 P1 奖励采集和比较的查询编码器。初始化真实历史的辅助 rollout 使用显式无查询 adapter 的 BGE 基座；若采用已有资产，修改对应显式路径后重新绑定采集身份。

已有候选答案的纯 CPU 诊断随 `python3 -m training.train_evidence_policy prepare --config CONFIG.yaml` 写入 `preference_manifest.json.candidate_quality_diagnostics`。该命令只读取已生成答案、评分和成本资产，按真实历史/预算/reader/scorer/query 分组，给出候选质量上限、最高预测并列时的遗憾区间和声明任务缺失列表，不产生模型训练或推理。输入资产不可用时不产生诊断结果。

固定模型协议审计与四格比较示例：

`reader_audit review` 使用 `observed_output_independent_review_v1`，只审核导出 traces 的实际输出；每条只产生审核调用，不重写答案。原始 token 数、输入、来源与阶段归因保留。旧目标生成、旧目标再审计和 `reader_data_only` 采集入口已移除。

```bash
python3 -m training.reader_audit export --config ACTUAL-AUDIT.yaml
python3 -m training.reader_audit review --config ACTUAL-AUDIT.yaml
python3 -m training.a_workflow compare --comparison-config COMPARISON.yaml \
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

C 使用 `paired_native_retention_v6`、`target_demand_rule=exact_complete_causal_prefix_and_body_ranges_v1` 与 `control_cost_rule=owner_call_only_fixed_decision_once_v1`。训练行保存独立会话集合，train/dev 集合互斥；同窗口各 TTL 及共享会话的窗口为重复测量。粗特征为访问状态、原生驻留、压力与并发组合。只有相同完整来源需求重访才标注整次下一 READ 的 target 收益；部分页复用仍记录在原生观察与整窗成本中。新增 owner 的目标收益按完整来源需求去重，容量与驻留按共享页增量；配对外部请求代价进入选择，正 collateral 不另计可叠加收益。单 owner 标签计量实际保留调用（ledger＋retain），inspect 与完整 callback 分别记录。完整集合的真实净效果仍由同轨 Native/value 对照和物理积分验收。

C 的两次容量构造共用原有 TTL15/60/180、owner、页并集和需求去重目标。配置使用单一 `protection_budget_tokens`（联合入口为 8192），GPU/Host 的 Native 物理配额与 `lambda_gpu` / `lambda_host` 驻留价格保留原含义。

`retention_pair_workflow` 先在同一捕获特征桶内选择两组完整、会话互不相交的训练窗口，并在独立开发轨迹中匹配同桶窗口，再核算 Native 与三个 TTL 的总调用预算。选择清单通过 `retention_trace replay --selection` 原样使用。拟合按重放时的实际特征要求每桶至少两个独立会话组；没有支持桶或开发覆盖时，工作流记录无可用资产状态。窗口数量与期限网格保持不变。

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

冻结 A 后，联合开发验收使用 `bench.capacity_workflow --full-queue`。默认同时执行 c1/c4，每个并发运行 E1 `B6_native`、E2 `S3_native`、E3 `B6_contextpilot`、E4 `S3_contextpilot`；四格为固定/收益 B × Native/ContextPilot 编译，`prefix_retention` 均为 Native。共同基线为实际存在的 `B6_native`，独立选中其他方法时使用所选首项；冻结与配对汇总读取同一参照。

每个并发还执行 `LlamaIndex_native` 和 `S3_contextpilot_no_fact_reuse`。外部方法通过统一 Controller 使用上游 history condense＋compact/refine，物化各数据集 `hybrid_rerank`、`query_mode: current`、无 query adapter/selector 的检索 YAML（`evidence_selection: ranked`、单查询融合、resolved rerank、`source_window: false`）；`--external-reranker` 显式选择发布版 checkpoint，默认 `models/bge-reranker-v2-m3`。其任务/支持规划关闭，Native 编译与保留相同。主消融与 E4 仅相差 `reading.fact_reuse: false`；仍保留实际历史和本轮共享提取。

E0_native（原 PR3 task-first/隔离数值）和 `B6_value` / `S3_value` 为显式可选对照，不进入默认主四格，不要求 C value 资产。
必须提供 `--arrival-plan`：JSON 索引格式为 `{"format":"contextkv_dataset_workload_index_v1","datasets":{"DATASET":"/absolute/fixed-arrivals.json"}}`，每个数据集严格覆盖声明队列。
使用已有请求轨迹离线导出固定相对到达（不产生模型调用）：

```sh
python3 -m bench.study_workloads --tasks TASKS.jsonl --records RECORDED-REQUESTS.jsonl \
  --output FIXED-ARRIVALS.json --name DECLARED-NAME --concurrency 4
```

旧请求记录已丢弃时，直接冻结新声明的共同到达映射，无需模型调用：

```sh
python3 -m bench.study_workloads --tasks TASKS.jsonl --arrivals DECLARED-ARRIVALS.json \
  --output FIXED-ARRIVALS.json --name DECLARED-NAME --concurrency 4
```

`DECLARED-ARRIVALS.json` 必须显式列出每道任务的相对到达秒数，例如 `{"session:1":0,"session:2":30}`；该示例不是已选定的正式负载。保存任务与到达映射哈希，所有方法复用同一声明；原生观测仍由未来真实执行产生。

来源记录必须包含所有声明任务的请求开始时间；未记录的上游到达不能反推或补成零。各方案复用同一固定计划及哈希，按各自真实 P3 历史完成依赖；等待前一轮和等待并发槽均计入完整响应。
`--complete-slo-seconds` 设置事先声明的完整响应 SLO，goodput 按达标完整回答/观察秒数计算，无阈值时不生成 SLO 数值。

```sh
python3 -m bench.capacity_workflow --full-queue \
  --config "$FROZEN_A_STUDY" --matrix "$JOINT_MATRIX" --arrival-plan "$ARRIVAL_PLAN" \
  --gain-asset "$B_GAIN" --online-gpu-devices "$ONLINE_GPUS" --reader-base-url "$READER_URL" \
  --contextpilot-checkout vendor/contextpilot --concurrency 1 4 --output "$JOINT_OUTPUT"
```

主四格共用冻结 A 的模型、支持策略、查询资产、阅读协议与预算；外部方法使用显式登记的发布版检索栈。可用 `--methods B6_native` 等参数将指定方法分配到独立 reader 服务，只生成所选方法的矩阵和任务。
`B6_native`、`B6_contextpilot` 和 `LlamaIndex_native` 均无需 gain/value；无 S3 方法时显式提供 `--reader-profile-id`。S3 方法只需当前协议 `--gain-asset`；显式选择 `*_value` 才需要 `--value-model`。ContextPilot 方法需要 `--contextpilot-checkout` 或冻结配置的 `execution.cache_compilation.checkout`；准备阶段验证真实 checkout HEAD 与清洁源码，并保存作者 Python 源码内容清单及摘要，而不是把固定 revision 字符串当作源码摘要。
默认六种方法在两个并发都使用相同 350 题、77 会话 IDs，各自实际 P3；未运行格保留缺失，不能用历史 E3/E4 的 value 结果充当新编译格。
固定 B 可独立执行，例如：

```sh
python3 -m bench.capacity_workflow --full-queue --methods B6_native \
  --config "$FROZEN_A_STUDY" --matrix "$JOINT_MATRIX" --arrival-plan "$ARRIVAL_PLAN" \
  --reader-profile-id "$READER_PROFILE" --reader-base-url "$READER_URL" \
  --online-gpu-devices "$ONLINE_GPUS" --concurrency 1 4 --output "$JOINT_OUTPUT"
```

收益 B 的 reader 身份由实际运行配置与实测收益资产绑定。`--online-gpu-devices` 接受逗号分隔的物理卡号。
可通过 `--online-process-roots` 指定逗号分隔的 PID，默认按 reader 和 helper 的端口发现
进程并加入协调进程。`--reader-base-url` 显式指定实际服务端点。联合矩阵沿用正式 350 题清单，
其中检索器设备配置与本次物理卡号一致。收益 B 的 GAP 限额继承实际收益资产；固定 B 沿用冻结观测配置，
缺省为 96 个单元和 10 个 passage。`--gap-max-units`、`--gap-max-passages` 可显式覆盖对应值；收益 B 仍须与采集协议一致。
`--lambda-time` 覆盖冻结配置中的动作成本权重，实际值写入 `capacity-plan.json`。
分设备结果用 `profiling.final_study_index --baseline-commit "$E0_SOURCE_COMMIT" --joint-scope joint-frozen-A/inputs.json`
汇总，逐个 `--artifact NAME=WORKFLOW.json` 指定各设备结果。共同输入、实际 worker 配置、
cohort、reader 与预算确定对照条件，各格分别列出实际启用的 B/C 资产。
报告保留全分母质量、完成率、已交付完整响应 P95、逐数据集 SLO goodput 与成本出处。比较身份同时核对到达计划哈希和 SLO；提供 `--cost-manifest`/`--physical` 生成成本命令时，必须通过 `--gpu-devices` 声明实际设备，不默认包含整机。

联合汇总的 `dataset_protocols` 来自矩阵已有 `split_kind` 与分区指标声明；Markdown 将公开基准、参考历史诊断和构造机制任务分表。每轨原指标、完整题目分母、已评分子集与未知区间同时呈现，混合开发 Q 的登记权重保持不变。

联合 Markdown 对照表直接呈现完整分母质量差与 P95 相对变化（候选/基线−1，负值为降低），并列出候选质量是否不低于基线。相对延迟只在比较身份一致、双方完整交付且全部已评分时提供；质量判断为点估计，资源变化仍读取实际成本账本。
`final_study_index` 同时输出 B 在 Native/ContextPilot 编译下、编译在固定/收益 B 下的直接对照，以及四格交互 `E4−E2−E3＋E1`；共同启用的 B 与作者编译身份须一致。四格保留均为 Native；TTL/value 可选结果不进入该交互。报告另列 E4/无事实复用及主线/外部参照，P95 交互使用四格完整队列各自的分位数。
每格执行前自动记录本机硬件和服务的原生缓存配置，随结果进入比较身份；只在声明的实际测量主机执行此工作流。资源容量、设备数量、GPU 型号/显存/驱动或原生缓存配置不同，以及观测缺失时，不报告可比差值。

每格按自己的 P3 历史执行原 350 题，使用声明的开发指标与权重。未交付回答计入
完整分母并贡献零分，已交付但未评分的回答保留质量区间。报告另列原生已付调用
索引和整批物理成本账本。评分依赖沿用冻结配置。`--source-manifest` 用于原有两轮容量诊断。

C 的离线 `retention_trace export` 需显式提供 `--failed-conversations`（零失败为 `[]`）。
同测量段内未导出的成功与失败请求墙钟区间进入压力索引，回放不使用与之重叠的窗口。
`profiling.retention_coverage --replay-root C_OUTPUT --output COVERAGE.json` 只读已有回执，
分别汇总干预目标与整窗重复来源页的 GPU 命中、Host 回载和重算，不产生模型调用。

B 的 DIRECT/READ/REDUCE/GAP 分支从实际动作前状态报价，分别拟合绝对分支成本，
同一批次以动作减 DIRECT 计算增量成本和实际质量差，已付共享成本只计一次。
REDUCE 只读已观察材料。源 packet 的 `legal_actions` 保留，补充的反事实候选单独登记并在执行前重新报价。所选来源读完后，任务必需的 `numeric_select` 和 CPU `count/sum/group` 按固定流程完成。收益资产的样本覆盖只约束可选 REDUCE。

完成记录的来源覆盖诊断采用注册前来源分母，合并既往与新增已读覆盖；GAP 新来源保留在实际 outcome，不修改注册分母。`whole_plan` 与 `observed_actions` 的资产分别声明实际特征，不交叉标记为动作模型。

C 固定 DAG 回放解析原始输出文本，新的生成响应用于实际 token、耗时和费用；完整窗口
预测用两臂窗口成本差验证，下一次 READ 的集合收益按事件及共享页去重。
