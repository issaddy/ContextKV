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

`train_plan_gain` 的显式训练入口拟合标准化线性 ridge 回归，保存 JSON 权重、训练状态清单、数据摘要、评分器及 reader 绑定；开发状态独立评估，测试答案不能进入收益监督。reader 更新后必须形成对应的新收益数据和资产。没有已训练且绑定当前 reader 的收益资产时，不启用该策略，现有固定策略继续作为对照。本轮只实现代码和单元测试，没有训练或真实收益结论。
