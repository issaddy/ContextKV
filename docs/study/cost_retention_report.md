# 成本、前缀复用与联合策略记录

本报告从容器内原始账本生成。每项结果保留读者 checkpoint、profile ID、实际提示词 SHA、sampling、引擎 fingerprint、cache config、设备 UUID、进程组与 resident model identity。训练、开发、冻结测试分别登记。

## 运行索引

| 任务 | 原始入口 | 输出 | 使用范围 |
| --- | --- | --- | --- |
| 全阶段物理采样 | `profiling.study_observer` | `manifest.json`、`samples.jsonl` | 观察器启动至结束，GPU 0–7 共用时间轴 |
| 完整阶段成本 | `profiling.plan_cost_workflow` | `workflow.json`、各阶段 `capture-entry.json` | 顺序执行的真实成本容量实验 |
| 当前 READ 重新执行与 C 配对 | `profiling.retention_pair_workflow` | `capture-stages.json`、`paired-workflow.json`、train/dev 配对目录 | 原任务实际检索原文、实际历史；新协议真正执行 |
| 成本总账 | `profiling.study_cost_ledger` | 显式指定的 ledger JSON | 重叠物理卡/PID采样并集；原始 job 积分不重复相加 |
| 联合策略 | 冻结的 rolling study workflow | 每轮 paid observation、StepRecords、QA 提交与模型调用记录 | 独立保留实际 reader / prompt / policy 范围 |

总账命令只接收显式文件或 glob，例：

```sh
.venv-sglang/bin/python -m profiling.study_cost_ledger \
  --manifest 'runs/study/*/capture-entry.json' \
  --manifest 'runs/study/*/training_cost*.json' \
  --manifest '<whole-study-observer>/manifest.json' \
  --physical '<whole-study-observer>/samples.jsonl' \
  --gpu-devices 0,1,2,3,4,5,6,7 \
  --output runs/study/full-cost-ledger.json
```

已知课题开始和结束的 client-monotonic 时间时，添加 `--window-start`、`--window-end`；早于任何采样的区间因此保留为 unknown。默认最大样本间隔为六秒，超过该间隔的洞不积分。

## 成本资产

| 字段 | 记录内容 |
| --- | --- |
| `selected_protocols` | 当前 READ prompt 的过滤开关；真实 ANSWER 控制起点选择 |
| `rows[].observed_input_range` | 实际测量输入范围 |
| `rows[].prediction_bin` | 注册输入桶内的统计预测范围，二元 list |
| `rows[].fit_method` | 同阶段、同提示词、同输出预算、同命中状态/物理范围的经验最大值 |
| `rows[].development_leave_one_observation_out` | 开发样本逐条留出误差；单样本桶为 null |
| `rows[].independent_held_out_error` | 独立留出误差，尚未测量时为 null |
| `rows[].success_count/failure_count` | 完整阶段语义处理成功/失败计数，QA quality 单独保留 |
| `native-assets.json` | 实际 first-token、decode、complete-control 各阶段拟合资产 |
| `step_output_priors[kind:cap]` | 实际输出长度与完整 control；ANSWER 另有真实 remaining-control |
| `claim_capacity` | 实际 tokenizer vocab 与固定 literal-key grammar 的可证明 claim 数上界 |

READ 附带的固定终端 ANSWER 从 `final_answer_compilation_start` 开始；rolling 成本探针从实际 `rolling_proposals_compilation_start` 开始。最终 rolling 资产默认选择后者，其他范围列在 observations ledger 的 excluded 项中，原记录定义不变。旧 READ 提示词通过 `--current-read-prompt` 排除；提示词未变的历史 ANSWER/REDUCE 继续按真实记录使用。

成本预测与观测分开报告。输入桶里的未测点使用 empirical estimate；其独立开发/测试误差据实际复测填写。物理成本仍绑定完整设备/进程/resident 模型范围；native token timing 的历史 resident 信息保留为来源审计。显式原文库、复制冻结 claims、加长引用名属于 cost-only 容量数据，不进入 QA、主质量、常规吞吐或 C value 训练。

## C 配对表

| 栏 | 数据来源 |
| --- | --- |
| train / dev 来源与首轮失败 | `capture-stages.json`、各 `capture-cases.json` |
| TTL 候选、观察边界与抽样权重 | `*-coverage.json`、`selection.json` |
| 真实输入一致性 | native/retained input hashes、DAG digest、sampling、原输出上限 |
| 原文需求复用 | causal body page IDs、partial/full、同/跨业务 session、真实访问间隔 |
| 原生命中与回载 | 每次 native quote、实际 receipt device-hit / host-loaded / computed tokens |
| 首次控制与释放 | retain、inspect/status、release 的实际时间 |
| 目标再访问收益 | `target_revisit_cost_difference_ms` |
| 其他来源与控制差 | `collateral_and_initial_control_cost_difference_ms` |
| 整个干预窗口收益 | `full_window_cost_difference_ms = Native − Retain` |
| 物理与逻辑容量 | 连续 GPU/Host GiB·s；独立保护集合/共享因果页边际 token·s |
| 资产与开发误差 | 只在 train fit；dev 预测、实际窗口差和未匹配桶分别报告 |

最多选择两组 train、两组 dev 配对。TTL 从预登记的 `2,5,8,15,30` 秒中选择两分区都具备后边界、且调用预算支持的最大值；窗口按固定 seed hash 抽样，不使用收益作选择依据。没有覆盖的 TTL 是 censored / unavailable，不记录成零复用概率。系统公共页不计为证据收益。收益保留正负号；原生策略占优时原样报告。

## 联合与消融表

| 读者与实际提示词 | 证据规划 | 前缀策略 | 质量 / 完成率 | response ms | GPU GiB·s | Host PSS GiB·s | 实际输入 / 输出 token | QA 提交次数 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 按冻结配置填写 | 固定 / rolling | native | 来自实际评测 | 实际账本 | 总账唯一物理范围 | 已观察进程范围 | 去重真实模型调用 | 每成功轮应为一次 |
| 同一输入任务分区 | 固定 / rolling | fixed TTL 资格 | 单列机制范围 | 实际账本 | 同上 | 同上 | 同上 | 同上 |
| 同一输入任务分区 | 固定 / rolling | train-fit value | 实际新提示词 joint 评测 | 实际账本 | 同上 | 同上 | 同上 | 同上 |

跨提示词泛化可以作为明确的实验条件：训练来源 SHA 和部署 SHA 同时列出，并用实际当前提示词的 joint 评测决定效果。固定 TTL 只提供独立 owner 生命周期资格；它不替代基于真实 paired 成本的 value 主结论。预分配 KV pool 的 VRAM 差为零时，仍报告实际保护逻辑容量与其他前缀 miss/回载代价。

## 全成本结算

从 ledger 分别引用：唯一 GPU 观察积分、唯一 PID PSS 观察积分、各卡 unknown intervals、原生逻辑 KV 积分、唯一实际模型 token、训练实报 tokens / wall / ranks、离线拟合 CPU、失败和重试成本。连续观察器和 stage/phase manifest 的重叠不重复累计；未建立实际阶段时间窗的旧请求份额不标成实测 per-step 费用。没有价格来源时 monetary total 为 null。

最终填写实际已完成数量、数据/模型/配置路径、冻结提交、验证范围、未覆盖测量区间及对应 ledger ID。原始文件继续保留在指定容器。
