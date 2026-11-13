# ContextKV 架构与机制

生成模型固定 Qwen3-32B。在线链路由共享查询编码、共享证据编码、结构化计算、价值停止和 SGLang 执行组成。`DocumentQA` 是 HTTP 和 benchmark 的共同入口。

## 模块职责

| 模块 | 职责与输出 | 主要代码 |
|---|---|---|
| 公共入口 | 接收问题、注册文档、定位会话，交给同一执行器 | `controller/api.py`、`qa.py` |
| 查询与证据 | 生成任务/Need 表示，限定合法范围，召回、重排、字段绑定和选择证据；输出检索状态 | `adapters/shared_encoding.py`、`shared_heads.py`、`shared_retrieval.py` |
| 来源与文档计算 | 维护原文地址、文档与版本关系，执行投影、筛选、连接及 CPU 聚合；输出完整结论和来源 | `adapters/corpus.py`、`document_compute.py`、`version_structure.py` |
| 决策与执行 | 持有本轮预算、累计费用和执行进度，决定扩展/计算/回答，编译请求并提交结果 | `controller/cooperative.py`、`value_policy.py`、`cooperative_request.py` |
| KV 执行 | 持有实际 KV 页、GPU/Host 驻留和私有覆盖层，选择原生或 InfoFlow 路径，返回真实执行成本 | `engine_patches/contextkv_compute.py`、`infoflow.py`、`infoflow_attention.py` |
| 数据与会话 | 定义共用数据结构，保存已提交历史、计算结果和注册文档 | `controller/schema.py`、`conversation_store.py` |
| 常规对照 | 提供普通 RAG 与 LlamaIndex 的检索、上下文组织和生成流程 | `adapters/providers.py`、`rag_dialogue.py`、`llamaindex_dialogue.py` |
| 离线训练 | 准备监督、采集实际答案收益、训练辅助模型和建立文档索引；产出权重与索引 | `training/` |
| 评测 | 按 benchmark 选题和重放、调用公共入口、官方评分并汇总最终质量与成本 | `bench/study.py`、`study_scoring.py`、`judge.py` |

在线模块使用 `adapters/` 中的编码器和任务头；`training/` 调用这些实现生成资产。评分客户端只属于 `bench/`。模型权重格式与训练入口保持不变。

调用方向为：HTTP／benchmark → DocumentQA → CooperativeController → 证据与文档计算 → 请求编译 → SGLang。检索状态只携带数据；控制器负责流程，SGLang 负责实际缓存资源。

## 查询、证据与计算

Qwen3-Embedding-0.6B 同次前向提供会话查询向量、Need 向量及 token 表示。任务头关联主体、条件、版本和前驱 Need；RCEM 将会话查询对齐到基础编码器的独立查询空间。文档向量由冻结基础编码器生成。任务与历史使用原消息坐标。

文档范围、版本和适用条件先确定合法来源，再执行 BM25 与稠密召回，通过 RRF 合并问题和各 Need 的候选。VersionIndex 保存版本、章节及字段对应关系；比较请求取得前后成对的原文依据。

Jina-reranker-v3.5 每批证据前向提供段落与 token 表示。字段头输出源跨度、Need、角色、条件关联和连接键；集合效用头在已有表示上选择互补证据。扩展候选时只编码新增材料。

基线与 ContextKV 共用 Jina 的分包、marker 投影及跨批打分函数。检索状态只保存数据，控制器显式传入已提交的计算事实，并调用检索器扩展候选；表格外正文的划分由 DocumentIndex 负责。

结构化索引保留原文表头、行列、单元格、单位与限定条件。DocumentProgram 执行投影、筛选、连接和聚合，数值运算使用 CPU。表格总体中的来源分为已绑定、明确不适用和尚未判定，完整性按整个合法目录计算。

字段头未完成的绑定由固定 Qwen 使用表 schema 和少量样例一次完成。非表正文只投影本题需要的字段，模型同时判断记录适用性，返回原文跨度；相同来源的 count/sum 共用这次投影。表格和正文结果合并成同一总体后聚合。空总体结果同样关联已处理的原文范围。

## 继续、停止与最终窗口

StopValueHead 复用证据表示与当前进度、预算、缓存和排队特征，学习停止与继续扩展后的实际最终质量及完整成本。训练使用多步回报。全量统计由总体覆盖与运算完成决定结束条件。

普通语义问答在证据充分时直接生成答案；额外主模型调用用于必要的字段绑定或语义投影。完成的结构计算按需求、条件和来源地址跨轮复用。

最终窗口先计入系统、问题、历史、完整计算结论和输出份额，再在剩余额度内选原文。ANSWER 只携带结果、必要限定和短来源引用，SQLite 中保存可复用计算及源地址。每个实际动作仅编译和提交一个请求。

## 上下文与 KV 协同

请求把稳定原文块地址、实际 token 区间以及已知后续证据需求一起提交 SGLang。引擎在同一个请求内选择原生精确执行或 InfoFlow 复用：

1. 原生 RadixAttention/HiCache 管理精确前缀与 GPU/Host 驻留。
2. 独立原文块生成可共享 KV；首次构建费用计入实际执行。
3. InfoFlow 恢复全局位置，使用生成模型的查询注意力信号选取重算 token。
4. 重算结果写入请求私有稀疏覆盖层；共享基块继续供其他请求复用。
5. 调度器结合证据需求、驻留、搬运和重算成本安排后续工作与保留。

上下文选择和结构计算改变后续原文需求；引擎反馈实际获取成本、缓存就绪度和排队状态，进入下一次价值决策。计价包括冷建、打分、搬运、重算及质量损失项，避免用局部命中代替完整收益。

## 并发、会话和结果

辅助模型对就绪输入合批。SGLang 对预填充、打分和重算按工作段调度。一个会话内顺序提交，不同会话共享并发额度；单轮失败记录已付费用，下一轮继续使用此前真实提交的历史。

ConversationStore 保存注册文档、已提交 QA、可复用计算和每轮最终状态。会话快照只读取历史和 corpus 标识，注册文档按文档集加载。benchmark 默认保留每题答案、来源、官方分数、完整响应和缓存计数，以及批次 GPU/Host 占用积分。评分所需原文是临时文件，评分完成后清理。
