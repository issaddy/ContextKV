# 任务、历史与来源实现

`adapters/history_state.py` 通过 `TaskInterpreter(model, HistoryIndex(...))` 将当前问题和实际提交的 QA 投影成 `TaskState`。`HistoryIndex` 使用冻结文档编码器，把每条原文消息按 384 token 分窗，保留 Unicode 字符偏移；活动需求指针、最近两轮的代表窗口和语义召回共同形成至多 16 个候选。不同会话中相同的轮次编号按消息原文区分。模型决定历史跨度、条件操作与目标、需求操作、任务回访和版本意图。

条件更新器执行 KEEP、ADD、REPLACE、REMOVE、COMPARE。替换操作保留 `supersedes`，从当前活动条件移除目标；组合与比较保留两个分支。检索查询从原文跨度装配，删除已替换条件值所占的字符区间，随后加入新条件与当前问题。实际 QA 原文留在持久历史中。`TeacherTaskModel` 使用固定 Qwen3-32B 的结构化输出生成训练标签和首轮集成预测。

## 在线跨度 wire 与持久化坐标

当前教师 wire 为 `owned_task_graph_v7_visible_token_boundaries`，任务语义版本为
`task_need_semantics_v3_anchored_owned_graph`。引用格式为
`{"window_id":"w0","boundary_tokens":[i,j]}`：两个索引指向实际可见窗口中已包含的 token，
顺序无关，由窗口映射恢复原消息 owner 与半开字符坐标 `TextRef(start,end)`。
Need 必须有实际锚点，Condition 至少有一个实际限制锚点。协议不枚举字符对。
教师调用保留原始 wire、语义版本及实际可见的 `task_catalogue`，持久化状态仍使用原文字符坐标。
协议探针入口为 `profiling.task_protocol`；编译开销与模型生成成本分别记录。

`adapters/evidence_pool.py` 的 `SourceDirectory` 保存来源版本、文档族、生效区间、发布日期、适用条件、表格与显式依赖。阅读单元按原文行和 tokenizer offset mapping 划分，长行按实际 token 分割。每个单元保留来源跨度及原始 passage ID。候选搜索采用当前查询 top50、每个 Need top20、历史来源 top12 和显式版本来源的轮转配额，初始 96 个主来源；依赖闭包随后展开。`full_scope()` 直接展开指定完整来源范围。

在线接口为 `ServingRetriever.configure_study(interpreter, tokenizer)`，随后 `retrieve_pool(snapshot, query, request_id)` 返回 `(TaskState, EvidencePool, RetrievalState)`。`candidate_search()` 返回 Dense/BM25 的完整排序池；`source_fetch()` 和 `archive_lookup()` 提供来源访问。离线 `controller_context_provider(..., source_index=directory)` 与在线共享 `retrieval_state()` 投影。

## 训练入口和资产

`training.task_data.build_task_data(config)` 导出 MultiDoc2Dial 原生训练分区。导出的会话、问题和来源数量由本次数据 manifest 记录。作者 QA 写入 `reference_history`；`turns.jsonl` 由真实 rollout 导入，使用 `actual_history` 与 `rollout_policy`。真实历史标注统一经 `training.a_workflow prepare-labels` 审核入口准备，再运行输出的 `task_labeling.yaml`；`existing_labels_path: null` 明确从零标注。实际采集数及 shortfall 由 cohort manifest 记录，train/dev 按会话隔离。完整准备顺序见 [运行手册](final_runbook.md)。

`training.task_model.train_task(config)` 已接通以下学习、解码和评价路径：

- 多锚点 Need 分组，Need—subject、Need—condition、Need—predecessor 关联与 scope。
- Condition key/value/predicate 学习分组、独立 predicate，以及 target_refs/supersedes 多标签。
- 从教师可见目录选择 family/version/resolved，source_turns 多标签；followup/correction/revisit 恢复任务。

targets、head、loss、decode、严格 v3 资产与 eval 使用同一图契约。mandatory 状态及目录先占用 8192-token 输入预算，历史窗口使用剩余额度。Need 前驱可指向完整跨轮 memory，条件只从选定 base 中选取。`source_scope` 保持调用方给定范围，不学习新的自由文本。查询对比目标使用已核对的 `verified_grounding`；文档编码器冻结。

当前辅助配置的 `task_model.initial_adapter_path: null`、`training.support_parent_asset: null` 表示分别从显式 BGE base 初始化；非空路径必须真实存在，不自动回退。代码输出 `adapter/`、`task-heads.pt`、`task-model.json` 与损失记录，`TaskModel` 同时提供任务预测和查询 `encode()`。训练与推理仍暂停，接口完整不代表已得到有效训练资产或质量提升。

配置键：`training.data_dir`、`training.rollout_paths`、`training.task_labels`、`training.task_output`、`training.device`、`training.dtype`、`training.adapter_lr`、`training.head_lr`、`training.epochs`；`task_model.base_path`、`task_model.initial_adapter_path`、`task_model.lora_rank`、`task_model.history_unit_tokens`、`task_model.history_candidates`。

GPU 训练和持出集模型指标由实际训练运行清单记录。
