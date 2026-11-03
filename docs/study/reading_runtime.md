# 预算化阅读运行链

`controller.reading.build_read_plan` 将 `EvidenceSet` 编译为 READ 步骤。传入实际
`template` 和 `snapshot` 后，装包按完整提示的 tokenizer IDs 计量，包括当前问题、
工作历史、条件、来源标识及输出预留。超长单元按 Unicode 字符切开；切片保留原始
来源的字符区间，并获得独立 span ID。`all_supplied_documents` 使用完整来源目录
产生的集合，覆盖表逐项记录实际执行的窗口。
各 READ 包展开本包记录的依赖闭包，重复携带所需表头、定义原文，并按实际输入
计量。覆盖表按 unit ID 记录首次完成阅读；重复依赖保留在每步输入账本。

服务入口支持两种计划注入：`Controller.serve_turn(context_plans=...)`，以及在线
检索器返回的 `EvidenceSelection.context_plans`。控制器先按多集合候选报价选择
语义计划，在同一 attempt 内执行 READ/REDUCE，随后编译并选择最终 ANSWER。
最终用户答案完成后提交一条 QA，业务会话版本增加一次。

READ 使用原生 `sampling_params.json_schema`，输出事实、支持、冲突和 OpenNeed；
ANSWER 使用现有引用 regex。每个步骤只使用一种约束。READ 的抽取事实引用本步
span ID；REDUCE 的结论引用输入 claim ID。原文事实、派生结论和完整叶子谱系由
`ClaimGraph` 保留，模型接收紧凑的事实投影。

`count`、`sum`、`union`、`group` 在 CPU 上确定性执行。计数按规范化实体、
条件和版本对齐；同一 Need 内的实体对齐不依赖模型选择的关系措辞。求和同时
归一可换算单位，并保存重复来源和参与计算的输入。
`compare`、`infer` 调用同一个冻结生成阅读器；输入超过窗口时先分组，再层次合并。
每个模型步骤与最终 ANSWER 共同消费请求的累计 token 和步骤预算。

高密度装包同时预测 JSON 事实输出的 token 数：合同字段与来源 ID 使用实际
tokenizer，内容额度由 `output_tokens_per_fact` 配置（开发起点 64）。行数与需求数
决定所需事实额度；输入或输出达到本步分配时结束当前包。Lmap 传入 `None`，按
普通摘要窗口装包。`reading_mode` 的三个消融直接改变执行路径：`coverage_only`
只传递覆盖；`claims` 用模型汇总；`typed_reduce` 执行确定性运算。

多跳通过 `Controller(..., extend_open_needs=callback)` 接入。回调接收
`(plan, open_needs, claims)`，返回 `EvidencePool`，其中 `source_blocks` 提供新增
原文，`units` 提供下一跳阅读单元。下一跳 READ 接收 OpenNeed 所关联的父 claim；
最终结论可以沿这些连接回到各跳来源。关系缺口先进行扩展，再执行最终 REDUCE。
每次扩展将已读来源、新来源和依赖共同计入原 passage 上限。扩展集合超出预算时，
保留 OpenNeed 并生成 partial 答案计划。
扩展返回 `EvidencePool.selected_set` 时，执行器按 need ID 更新扩展需求的支持，
保留已完成需求，并将实际阅读集合和支持图写入最终 TurnRecord。

SQLite 的 `turn_plans` 保存所用计划，`execution_steps` 保存各步骤的任务、输入、
原生 epoch、输出、解析结果和计时。每次原生生成使用独立步骤请求 ID 和临时
epoch；完整结果持久化后登记退役 outbox，并解除该步骤的原生引用。物理缓存页
继续由原生缓存管理。READ 响应丢失时，同一外层请求可根据已保存步骤和原生
`status` 恢复结果，再完成最终一次提交。
`reading_progress` 与后继计划原子保存已扩展 OpenNeed、预算终止需求及辅助用量。
重试恢复这些进度，已完成的 gap 评分只计一次。`reading_elapsed_ms` 累积各次实际
执行段，包含失败调用等待与恢复工作，不包含两次请求之间的空闲时间。
每个实际模型步骤在生成前取得原生只读报价，保存完整 cache quote、active/queued
并发观测和 cache config hash；prepare、quote、parse、retire 均计入 step wall time。
确定性 CPU 汇总记录运行耗时，失败保留 FAILED 步骤后向调用方报告原始错误。

`TurnResponse` 分开记录最终用户首 token 与内部 READ 首 token，并返回 READ/
REDUCE 数、累计输入输出、逐步骤阶段耗时、辅助模型时间，以及累计 device hit、
Host load 和实际 prefill。最终 ANSWER 的原始字段继续表示该次生成；`total_*`
字段表示完整请求的模型工作量。
`reader_input/output_tokens` 表示 READ/REDUCE/ANSWER 工作量，
`auxiliary_input/output_tokens` 表示任务、检索、支持和后续 gap 的辅助模型调用；
`total_*` 为两者之和。辅助调用未测得的 token 保留空值。`reading_ms` 是包含 gap
辅助调用的执行段 wall time 总和；阶段账单将 gap 开销单列，成本规划只计一次。

`profiling.reading_measurements` 从明确标为 dev 的步骤账本导出标准 Measurement：
各 READ/READ_SUMMARY/REDUCE profile 分文件保存，真实 receipt 提供 hit/Host prefix，
coverage_only 使用独立 READ_COVERAGE profile。生成前 quote 提供 active+1 并发
及 queued 旁账。客户端首 token、首至末 token chunk、step wall 减末 token
三个阶段使用同一时钟并精确加总；原生 decode 继续保留在旁账。缺失客户端末
token 观测的旧记录留在 step-ledger，不补为估算校准点。

当前机制入口为 `bench.reading_mechanisms`，从实际 READ/REDUCE/ANSWER 账本检查
共享数值阅读、完整来源谱系、有限归约及完整成本。HTTP query 通过 `extend_for_plan`
接入 gap 检索；最终 AnswerPlan 的 source spans 和 citation_ids 绑定实际最终候选。

CPU 核心冒烟由 `tests/test_reading_runtime.py`、`test_shared_numeric_reading.py` 和
`test_conclusion_allocation.py` 保留。实际模型质量和成本由 study 实验记录。
