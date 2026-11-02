# 任务、历史与来源实现

`adapters/history_state.py` 通过 `TaskInterpreter(model, HistoryIndex(...))` 将当前问题和实际提交的 QA 投影成 `TaskState`。`HistoryIndex` 使用冻结文档编码器，把每条原文消息按 384 token 分窗，保留 Unicode 字符偏移；活动需求指针、最近两轮的代表窗口和语义召回共同形成至多 16 个候选。不同会话中相同的轮次编号按消息原文区分。模型决定历史跨度、条件操作与目标、需求操作、任务回访和版本意图。

条件更新器执行 KEEP、ADD、REPLACE、REMOVE、COMPARE。替换操作保留 `supersedes`，从当前活动条件移除目标；组合与比较保留两个分支。检索查询从原文跨度装配，删除已替换条件值所占的字符区间，随后加入新条件与当前问题。实际 QA 原文留在持久历史中。`TeacherTaskModel` 使用固定 Qwen3-32B 的结构化输出生成训练标签和首轮集成预测。

`adapters/evidence_pool.py` 的 `SourceDirectory` 保存来源版本、文档族、生效区间、发布日期、适用条件、表格与显式依赖。阅读单元按原文行和 tokenizer offset mapping 划分，长行按实际 token 分割。每个单元保留来源跨度及原始 passage ID。候选搜索采用当前查询 top50、每个 Need top20、历史来源 top12 和显式版本来源的轮转配额，初始 96 个主来源；依赖闭包随后展开。`full_scope()` 直接展开指定完整来源范围。

在线接口为 `ServingRetriever.configure_study(interpreter, tokenizer)`，随后 `retrieve_pool(snapshot, query, request_id)` 返回 `(TaskState, EvidencePool, RetrievalState)`。`candidate_search()` 返回 Dense/BM25 的完整排序池；`source_fetch()` 和 `archive_lookup()` 提供来源访问。离线 `controller_context_provider(..., source_index=directory)` 与在线共享 `retrieval_state()` 投影。

## 训练入口和资产

`training.task_data.build_task_data(config)` 导出 MultiDoc2Dial 原生训练分区。2026-11-02 本地运行得到 3474 个作者会话、22806 个问题及 488 个完整来源，记录在 `runs/study/data/task_data_manifest.json`。作者 QA 写入 `reference_history`；`turns.jsonl` 由真实 rollout 导入，使用 `actual_history` 与 `rollout_policy`。`label_actual_rollouts()` 对真实历史进行教师标注，并为选中的历史跨度生成成对移除测量请求。

`training.task_model.train_task(config)` 从现有 BGE-M3 查询 LoRA 初始化，训练历史、BIO 跨度、条件操作、条件关系、条件指针、需求操作、版本和任务指针头。对比检索目标使用依据实际历史核对过的 `verified_grounding`。文档编码器冻结；超过编码窗口的文档按完整窗口编码并聚合。输出 `adapter/`、`task-heads.pt`、`task-model.json` 与逐步损失。`TaskModel` 同时提供任务预测和已有 `DenseIndex` 所需的查询 `encode()`。

配置键：`training.data_dir`、`training.rollout_paths`、`training.task_labels`、`training.task_output`、`training.device`、`training.dtype`、`training.adapter_lr`、`training.head_lr`、`training.epochs`；`task_model.base_path`、`task_model.initial_adapter_path`、`task_model.lora_rank`、`task_model.history_unit_tokens`、`task_model.history_candidates`。

本地机制与相关回归测试共 92 项通过。GPU 训练和持出集模型指标由训练运行清单记录。
