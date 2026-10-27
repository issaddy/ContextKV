# 机制、研究依据与课题增量

本文说明各机制如何服务于ContextKV及其适配工作。完整论文和项目链接见[参考文献清单](references.md)，实际结果见[数据与评价](data_and_evaluation.md)，工程状态见[实现状态](implementation_plan.md)。

## 机制与复用关系

| 机制 | 主要依据与固定来源 | ContextKV的适配与扩展 |
|---|---|---|
| 上下文对齐与复用 | [ContextPilot `b4c8032`](https://github.com/EfficientContext/ContextPilot/tree/b4c80327f6cfae2286839413731fcb6f53578b2f)、[TokenPilot / LightRSI `9f0f193`](https://github.com/zjunlp/LightRSI/tree/9f0f19308a30445438779efbf3108a7d965a55c7) | 在不可变会话快照上计算重叠与位置距离；用实际正文可见性、版本及共同引用范围组织APPEND/REBUILD |
| 多轮任务状态 | CHIQ、HAConvDR、QuReTeC、SOM-DST、Context-aware DST、JSON Merge Patch | 单次解释维护开放任务字段、需求和命名条件；保留/更新/删除原子提交，按状态变化位置找回因果历史 |
| 检索、融合与选材 | [BGE-M3 `5617a9f`](https://huggingface.co/BAAI/bge-m3/tree/5617a9f61b028005a4858fdac845db406aefb181)、[BGE-reranker `953dc6f`](https://huggingface.co/BAAI/bge-reranker-v2-m3/tree/953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e)、BM25S、RRF、xQuAD、结果折叠 | 共享索引与评分池，保留主任务锚点，按需求覆盖选择不同内容组；可选联合任务/来源决策 |
| 原文结构与范围 | Contextual Retrieval、ParentDocumentRetriever、Docling、SPIRE、[CRAFT](third_party/CRAFT_NOTICE.md)、TabRAG | 传递来源、章节、依赖与原文窗口；表格行列及作者列表保持来源坐标，实际阅读材料进入共同A/R |
| 原生执行与完整成本 | [SGLang `3f36fc0`](https://github.com/sgl-project/sglang/tree/3f36fc08331f64cd63c4fbbdf39cc402f38bb38b)、阶段测量/插值、经验输出长度、Landlord、WAIT/TIE等调度分析 | 同批GPU/Host只读报价；分开计算容量预留与服务长度；结合等待、恢复、Prefill、Decode及分动作控制开销选择方案 |
| 缓存生命周期 | SGLang Radix/HiCache与会话软引用、[EfficientAgent `b21b12a`](https://github.com/KunmingSHAO/efficientagent_release/tree/b21b12a07a3ccf473c82acf00820b1aa2073044b) | 业务epoch映射原生引用，持久提交后退役；共享引用、outbox和重启恢复；可选业务工作集Host准入 |
| 引用与输出 | SGLang XGrammar、原生NGRAM、FullCite、Lost in the Middle | 当前合法引用范围进入原生解码；完整原文与历史参与回答，解释服务复用原生片段草稿 |
| 评价与工程 | MT-RAG/MultiDoc2Dial、原生基线、来源核验工具、FastAPI、SQLite事务 | 统一CLI/HTTP链路和P1/P2/P3协议，保存实际输入、回答、阶段成本与来源，支持恢复和离线复算 |

## 关键适配

**A：从已见内容到当前可见证据。** ContextPilot原索引操作会修改状态；本项目将对齐算法独立为纯候选构造。距离为`1 − overlap / max(context_lengths) + 0.001 × mean_shared_position_difference`，共有材料优先对齐，剩余材料保留检索顺序。精确正文共享、原文跨度和版本身份共同决定复用资格；归档记录用于回访恢复。

**A：从查询改写到完整信息传递。** 当前问题、完整任务和子需求共同参与检索及选材，实际QA提供对话先行词，外部原文支持回答事实。研究中区分任务误解、材料不足、材料不适用和生成使用错误。实现细节见[多轮意图与证据机制](research/source_grounded_query_resolution.md)，配置见[混合检索](guides/hybrid_retrieval.md)。

**B：从局部Prefill收益到完整执行成本。** 成本模型复用真实阶段测量，容量依据原生预算，预计服务长度来自已完成历史。控制成本按动作记录；误差带描述预测差异，动作按预计完整成本选择。原理和公式见[联合决策](research/joint_optimization_model.md)，队列与容量记账见[等待估计](research/admission_wait_model.md)。

**C：从文本版本到原生引用生命周期。** 应用层管理业务版本和提交，SGLang管理物理匹配、锁、分配、传输与淘汰。EfficientAgent的工作集准入思想适配到原生BackupKV，并扩展为每请求累计成功准入页预算；实现见[Host准入](research/host_admission_design.md)。

上述工作形成场景适配、算法优化与跨模块协同的课题工作量。各模块的默认配置和效果取舍由[实现状态](implementation_plan.md)及[评价记录](data_and_evaluation.md)维护。

## 来源与许可

表中的固定来源定位实际研究和适配版本；更多调研方法统一收录于[参考文献清单](references.md)。直接复制的函数、提示和资产保留对应NOTICE与许可证，见[第三方说明](third_party/)及各资产目录。模型、数据与代码分别保留发布来源。
