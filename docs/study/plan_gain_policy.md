# 完整计划收益策略接口

`controller.plan_policy` 使用现有 `ReadingOutcome` 作为只读工作状态，通过其中的 `ContextPlan`、`TaskState`、`ReadResult`、claims、coverage 与 open_needs 读取当前事实和待补需求。它不创建第二套任务状态。

A 提供最多四个 `PlanCandidate`，语义为 `direct_answer`、`complementary_evidence`、`multihop` 或 `scope_scan`。每个候选引用完整 `ContextPlan`，保留 `ReadTask` DAG，并包含一个终端 ANSWER。`complete_plan` 可将现有计划转换为该接口；每个当前可执行步骤的 `PlanStep.variants` 保存实际编译的 `PromptCandidate`。互补证据作为整个计划估值，不把两个局部支持分相加为收益。

```python
policy = PlanPolicy(gain_model)
next_step, context_action, output_budget = policy.choose(
    plans, reading_outcome, native_state, reusable_prefixes)
```

`NextStep` 包含计划 ID、实际下一项 `ReadTask`、选中的 `PromptCandidate` 与复用 prefix IDs。执行器完成这一项后持久化现有 `StepRecord`/`ReadResult`，A 更新计划，再次调用 choose。最终 ANSWER 仍由现有外层事务提交一轮 QA。

`NativePlanningState` 提供已验证的 reader checkpoint/profile、当前 snapshot、真实 `CacheQuoteBatch`、已有 `CostModel`、逐候选物理占用预测及已支付成本。延迟单位为 ms，GPU/Host 积分单位为 GiB·ms，峰值为 GiB。已知输入用原生报价预测排队、prefill、回载、decode 与控制成本；`paid_cost` 包含本请求此前检索、辅助推理、执行及控制开销，避免重复加入 PromptCandidate 中旧的 READ 总额。

未生成的下游输入由 `native.future[(plan_id, step_id)]` 提供显式 profile 预测。它保留未来输入的预测范围和 calibration ID，不构造不存在的 input_ids 或原生报价。选择器仅对接下来两个执行步骤的上下文动作和预算选项展开前瞻，同时计入完整尾部。完成步骤不会再计费；每一步都重新读取原生状态并滚动选择。`last_evaluations` 保存全计划收益、完整成本、可行性、前瞻路径及预测来源。

收益目标为预测 ΔQ 减去完整延迟、GPU 积分与 Host 积分的预算代价，权重沿用 `ContextBudget.lambda_time/lambda_gpu/lambda_host`。累计 token、完整响应预算、物理峰值和积分预算为硬约束。全范围要求来自原工作状态的 `coverage_units`：已完成 READ 的覆盖与计划剩余 READ 的范围必须包含它；范围未满足的直接 ANSWER 不可选择。内部步骤输出预算受 READ 预算约束，终端 ANSWER 保持当前声明的回答预算。

C 的 `reusable_prefixes` 是候选 ID 到真实 prefix IDs 的映射。`ResourceForecast` 必须对应当前 quote_seq；前缀影响已经体现在原生延迟预测和 C 的物理占用预测中，B 不再重复减去一次“缓存节省”。

`training.train_plan_gain.build_gain_record` 只接收已经真实生成和评分的直接 ANSWER 与反事实回答。两者共享问题、实际 QA 历史、逻辑状态、reader checkpoint/profile、比较预算、数据分区与评分器；原始回答、原生回执、评分和全过程成本完整保留。监督标签精确等于 `quality_counterfactual - quality_direct`，不从支持分、来源规则或参考答案生成。

`train_plan_gain` 的显式训练入口拟合标准化线性 ridge 回归，保存 JSON 权重、训练状态清单、数据摘要、评分器及 reader 绑定；开发状态独立评估，测试答案不能进入收益监督。reader 更新后必须形成对应的新收益数据和资产。没有已训练且绑定当前 reader 的收益资产时，不启用该策略，现有固定策略继续作为对照。该收益策略的启用状态由实际反事实数据和模型资产 manifest 记录。

实际运行入口为 `Controller(..., rolling_policy=RollingPlanRuntime(...))`，配置入口为 `RunConfig.plan_gain`。默认值为 `None`。启用参数包含 `mode: rolling_gain`、实际 `reader_checkpoint` 与 `reader_profile_id`、`gain_asset`、`execution_profile_asset`、内部 READ 的 `output_budgets`、物理采样的 `gpu_devices/process_ids/sampling_interval_seconds`，以及可选 `limits`。资产加载核对真实模型 fingerprint、cache configuration、采样配方、评分训练分区与来源。该路径当前比较 READ 输出预算，最终 ANSWER 使用声明的回答预算。

`controller.rolling_policy` 将至多三个现有证据计划与一个当前状态直接 ANSWER 组成有限候选；提前声明已知操作的 REDUCE 尾部。每轮编译当前可执行输入、批量取得真实 quote、调用 `choose`，随后只执行选中的一个逻辑 READ/REDUCE。执行结果写入现有 `StepRecord`、`ReadResult` 和 `ReadingOutcome` 后再次选择。内部 REDUCE 输入超出已报价窗口时，该输入不可执行；它不会产生未计价的隐含子调用。步骤 ID 由任务、实际原文、条件和 READ 布局确定，不同集合的 `read:0` 不会覆盖同一持久化记录。

实际 OpenNeed 经已有定向检索接口取得 `EvidencePool`，新来源、依赖闭包和父 claim 形成新的多跳候选。检索调用及其模型 token 纳入累计账本；新候选仍须通过完整计划选择和剩余预算。已读来源保持原始坐标，已读事实与新原文可共同支持下一步。全范围 coverage 不随候选切换而丢失。

`controller.plan_profiles` 仅在实际测量范围内提供资源和未来阶段成本。未来模型输入使用声明的输入上限，未来缓存命中按零复用状态查表；测量未达到该上限时，完整成本不可用。CPU REDUCE 的 profile 同时绑定实际 operation 与 claim 数量范围。execution 资产存在已绑定实际 tokenizer/原生必需字段协议的 `claim_capacity` 时，未来模型前驱按 `claim_count_ceiling` 计算输出 claim 上界；已完成前驱使用真实 claim 数，未来确定性 CPU 前驱按真实实现的单个输出 claim 计上界。没有容量证明时保留输出 token 数的保守上界。`profiling.plan_cost_fit` 从完整阶段窗口及物理采样构建资产，保留覆盖范围和排除记录。

`rolling_plan_decision` 记录状态、全部候选的收益和成本、native quote、已支付 token/延迟/物理积分、未来阶段 profile、最终决策及实际物理边界样本。最终 ANSWER 的 `answer_phase_window` 同时写入 fresh `TurnResponse` 与 `ledger_committed` 事件，保留实际开始定义、commit 时间与响应控制尾部终点；幂等与恢复回放不创建新的测量窗口。固定策略从最终回答编译开始计阶段窗口，rolling 从最后一轮候选编译开始；两者的 profile 分开。rolling 当前 ANSWER 资源预测扣除同一阶段已进入 paid 账本的实测积分，扣减窗口、边界样本和逐候选明细均保留。

`python -m training.collect_plan_gains --pairs <pairs.jsonl> --output <new-gains.jsonl>` 可导入已生成、已评分的直接回答与候选回答。每个 packet 保存现有 `PlanCandidate/ReadingOutcome/SessionSnapshot`，指定原始指标及有协议出处的归一化规则，并指向两个完整 run JSON。run 必须在执行前记录 `frozen_state_key`、真实 reader/profile、实际 plan ID、分区及最终执行预算。导入器核对真实回执与所有成本，保留原始 run、评分、文件 hash 和来源清单；它不执行回答或评分，也不把六格尚未评分的输出当收益数据。

资源资产的 `resource_scope` 绑定实际 GPU index/UUID、外部进程 roots、PSS 口径和原生端点；采样要求各声明进程均有真实 PSS 记录。`resident_model_manifest` 与 canonical digest 保存采集时实际加载的检索/辅助模型。provider 加载完成后，`Controller.bind_resident_models(provider.manifest)` 核对这些常驻资产；独立 reader 用显式 `bind_resident_models(None)`。因此 reader-only 采样与包含 query/BGE 常驻模型的完整在线资源口径具有明确绑定。
