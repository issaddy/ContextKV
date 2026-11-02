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
上述为执行入口，尚无完整六格质量结果。reader 未就绪时可省略其配置，矩阵保留 pending 单元。
输出目录首次创建 `protocol.json`，重复执行使用新的目录。`--r0-url/--r1-url` 可显式覆盖端口；`--r0-checkpoint/--r1-checkpoint` 连接既有权重登记。实际 run config、model scope、engine fingerprint 与 cache configuration 另存运行身份。

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
