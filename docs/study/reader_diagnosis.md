# 冻结实际历史的六格诊断

`bench.reader_diagnosis` 从 JSONL packets 执行 E0/E1/E2 × R0/R1。每个 packet 包含 `task_id`、`actual_history`（原始 `TurnRecord` JSON）、`e0_plan`、`candidate_pool`、`full_applicable_pool`、`annotations`、`scope_review`，可另带 `verified_task_state`、`reference`、`offline_reference`、`goal_conditions`、`original_preparation_usage`。契约使用现有 `ContextPlan`/`EvidencePool`/`TaskState`；oracle 注释类型见 `bench.evidence_diagnosis`。

```bash
python -m bench.reader_diagnosis \
  --packets runs/diagnosis/packets.jsonl \
  --r0-config configs/qwen3_32b.yaml --r0-url http://127.0.0.1:30005 --r0-gpu 5 \
  --r1-config models/study-reader-r1/r1_run_config.yaml --r1-url http://127.0.0.1:30004 --r1-gpu 4 \
  --body-tokens 2048 --output runs/diagnosis/body-2048
```

当前 R1 已完成真实 LoRA、合并并在30004/GPU4就绪；最新可比R0为30005/GPU5。
上述为执行入口。sixgrid-v1 已实际评分152格，16格缺失；14个状态中2个泛 topic 的 E1/E2 原文范围审核不足。
可比主表只使用其余12个共同状态，保留每个格子的实际 protocol 与原始缺失原因。
reader 未就绪时可省略其配置，矩阵保留 pending 单元。
输出目录首次创建 `protocol.json`，重复执行使用新的目录。`--r0-url/--r1-url` 可显式覆盖端口；`--r0-checkpoint/--r1-checkpoint` 连接既有权重登记。实际 run config、model scope、engine fingerprint 与 cache configuration 另存运行身份。

主 agent 从指定容器报告的共同12状态 `project_condition_version_score` 如下：

| 证据与任务绑定 | R0 | R1 |
|---|---:|---:|
| E0 / inferred | 0.8708 | 0.7333 |
| E0 / verified | 0.7333 | 0.7667 |
| E1 / inferred | 1.0000 | 1.0000 |
| E1 / verified | 0.9833 | 1.0000 |
| E2 / inferred | 1.0000 | 1.0000 |
| E2 / verified | 1.0000 | 0.9792 |

首轮 R1 在 E0/inferred 下低于 R0 13.75个百分点，没有得到质量改善。同一候选池的 E1/inferred 相对 E0 分别为 R0 增加约12.9个百分点、R1增加约26.7个百分点；在这批状态中，实际证据选集保留了主要改进空间。verified 任务替换的收益并不一致，不能据此单独归因为任务解析。R0 继续作为较强的当前候选；完整预定开发候选选择尚未完成。

这是一批小型构造型动态开发状态的工程主分，不是公开主 benchmark 结论，也不把 oracle 选材当作可部署计划器效果。后续先补精确 topic 目标及其来源审核；R1 下一轮至少应在匹配 E0 条件下达到 R0 当前的0.8708。指导要求的5—8个百分点提升，若以本批当前 R0 候选为局部诊断参照，对应0.9208—0.9508；正式验收仍使用完整预定 cohort 和相同在线预算。条件/版本错误减少30%及 P95 ≤1.2倍，需各自实际错误计数和负载成本，不能由本表代算。

E0 保留冻结时真实选中的单元，重新以实际 tokenizer 测量正文；超出目标正文预算就记录 infeasible。E1 只在原候选单元及其依赖内优化，E2 按已审核 witness 的 SourceSpan 边界物化原文。每个必要点的 witness 是必须共同出现的条件、前提、例外与版本跨度组，备选 witness 之间可择一。预算不足时保留实际覆盖与遗漏，不填答案质量分。E2 的全适用范围审核是否完成单独记录。

各格导入相同实际 QA 历史，统一正文、READ 输入输出、最终 ANSWER 预算、阅读模式和布局参数；默认沿用 E0 的原始设置，正文预算由命令行指定。`read_packing` 同样继承 E0，可显式声明 `--read-packing mixed|source_windows`；两 reader 使用相同装包方式。所有格使用 always_rebuild 和 native 前缀生命周期。默认每格 cold 初始化；`--cache-start observed` 保留并记录实际缓存初态。两个 reader 的 tokenizer、模板、采样设置与对应初始 READ token 输入必须一致。

阅读深度沿用 Incumbent：E0 初始计划没有 READ 时，E1/E2 同样直接 ANSWER，不新增 READ 或 REDUCE。E0 inferred 在材料与 READ 参数不变时保留原始初始 DAG、支持记录、AnswerPlan 与布局候选；换证据或 verified binding 才按原有阅读策略重建。`initial_plan_report` 记录初始 DAG、READ 输入、直接 ANSWER 输入及实际记录的 token 一致性；没有实际输入观测时标记未观测。

动态 packet 优先从真实 `planning_trace.context_plans` 按选中 plan_id 提取初始 E0，执行后的扩展计划与 outcome 单独保存。仅有旧运行 claim 索引而找不到初始计划时，单元标为 `initial_plan_unavailable`。既有 R0 gap READ 与 claim 索引不会直接接入 R1；REDUCE 根据本次 reader 的实际新结果生成，来源运行中的 gap 扩展另外列出。含 gap 的记录因此明确限定为初始计划诊断，不冒称完整扩展行为已被重放。

每格只运行一次当前问题的 `serve_turn`，内部仍走 READ/REDUCE/ANSWER，并验证只提交一轮新 QA。有 `verified_task_state` 时另执行 verified binding 列：问题、实际历史与证据保持不变，替换任务对象、条件、版本和必要点。当前来源引用随实际选集更新。

每格目录保存计划、原始 StepRecord、原生回执、答复和 `results.unscored.jsonl`，以及兼容 `bench.study_scoring` 的 `official.jsonl`、`audit.jsonl`、`references.jsonl`。每个 reader 另存物理 GPU/PSS 采样、原生 KV 池采样和运行身份；逐格成本包含请求完整墙钟、所有实际 reader token、prefill/设备命中/Host 回载、物理内存积分及峰值。失败生成同样留存原始输出、回执和已支付成本，然后传播异常。

计量范围是冻结证据后的计划构建与 READ/REDUCE/ANSWER 执行。oracle 审核和选择为共享离线准备，原始检索/任务/支持费用通过 `original_preparation_usage` 单列，不反复作为各格的新模型调用。历史导入和缓存初态准备记录为离线单元设置开销，controller 初始化独立记录。质量评分在生成后单独执行，矩阵不会因模拟测试或缺失 R1 宣称六格实验完成。

构造型动态开发集可从实际 E0 run 生成 packets：

```bash
python -m bench.dynamic_goal_packets \
  --run-dir runs/dynamic-goal/E0/dynamic_goal_dev/dev \
  --dynamic-tasks runs/study/evaluation/dynamic_goal_dev/tasks.jsonl \
  --version-tasks runs/study/evaluation/version_dev/tasks.jsonl \
  --version-references runs/study/evaluation/version_dev/references.jsonl \
  --run-config configs/qwen3_32b.yaml \
  --output runs/diagnosis/dynamic-packets.jsonl
```

builder 以真实 `question_policy.goal_id/goal_conditions` 选择当前对象、平台和版本，再核对 `version_dev` 已有原文跨度逐字匹配。版本比较使用双版本联合 witness；移除平台约束保留每个平台必要点。verified state 的对象与条件引用实际用户文本，不导入参考答案值。原始 P3 历史、候选池与 E0 计划保持原值，输入文件身份、实际 cohort 缺口和五类场景覆盖写入 manifest。

泛摘要 topic switch 没有最小完备原文标注时，packet 标为 `review_incomplete`，执行器只运行其 E0 与任务绑定替换；E1/E2 明确跳过，不用空 witness 充当完备 oracle。全部审核声明限定于构造型开发目标及其提供的适用来源，不扩展为 native 全语料事实审核。该入口不读取公开 test 数据，不执行训练，也不接入线上检索。
