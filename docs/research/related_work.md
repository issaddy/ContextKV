# 复用基础、方法原理与适配范围

这里保留直接支撑当前实现的固定来源。论文结果属于上游；ContextKV 的实际效果统一见[数据与评价](../data_and_evaluation.md)。移除的逐轮调研与实验纪要可从 Git 历史恢复，原始运行数据仍留在 `runs/`。

| 工作 | 固定来源 | 当前用途 |
|---|---|---|
| SGLang | [`3f36fc0`，Apache-2.0](https://github.com/sgl-project/sglang/tree/3f36fc08331f64cd63c4fbbdf39cc402f38bb38b) | 直接使用生成、UnifiedRadixCache、HiCache、准入/调度、软引用与传输；补丁提供只读报价、版本 guard、完成 receipt 和退役接口 |
| ContextPilot | [`b4c8032`，Apache-2.0](https://github.com/EfficientContext/ContextPilot/tree/b4c80327f6cfae2286839413731fcb6f53578b2f) | 独立适配重叠距离、前缀对齐与引用注释；没有移植有副作用的全局索引服务 |
| LightRSI / TokenPilot | [`9f0f193`，MIT](https://github.com/zjunlp/LightRSI/tree/9f0f19308a30445438779efbf3108a7d965a55c7) | 稳定前缀、计划与提交分离、生命周期门控原则；没有移植 TypeScript 运行时或摘要器 |
| EfficientAgent | [`b21b12a`，Apache-2.0](https://github.com/KunmingSHAO/efficientagent_release/tree/b21b12a07a3ccf473c82acf00820b1aa2073044b) | 工作集压力与 Host 备份准入的可选适配；保留 SGLang 原生 BackupKV，默认关闭 |
| BGE-M3 / BGE-reranker-v2-m3 | [M3 `5617a9f`](https://huggingface.co/BAAI/bge-m3/tree/5617a9f61b028005a4858fdac845db406aefb181) / [reranker `953dc6f`](https://huggingface.co/BAAI/bge-reranker-v2-m3/tree/953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e) | 固定权重及官方接口；BM25S、RRF 和重排组成可选真实检索链 |
| CHIQ / HAConvDR | [CHIQ，EMNLP 2024](https://aclanthology.org/2024.emnlp-main.135/) / [HAConvDR，ACL Findings 2024](https://aclanthology.org/2024.findings-acl.792/) | 复用先解歧再改写、减少无关历史的原则；当前为单次条件调用的 Qwen3 意图解析，没有移植其训练或完整多阶段流程 |
| SOM-DST | [ACL 2020论文](https://aclanthology.org/2020.acl-main.53/) / [作者实现](https://github.com/clovaai/som-dst) | 适配状态保留、更新、删除的机制；使用开放文本字段和已有Qwen3，不移植领域ontology或训练模型；操作预测与条件粒度仍需验证 |
| UvA，SemEval 2026 Task 8 | [作者论文](https://arxiv.org/abs/2606.11945) | 多轮改写→检索→重排→回答的同场景参考；检索排名改善不能替代回答质量验证，不继承其比赛结果 |
| Qwen3-Reranker-0.6B | [`fd9fb1d`，Apache-2.0](https://huggingface.co/Qwen/Qwen3-Reranker-0.6B/tree/fd9fb1d26c07223ced488065909faf522e29cc7d) | 官方模板及末位 yes/no 打分适配；已有同候选池对照，未替代 BGE 默认设置 |
| MiniCheck | [官方代码 `b58b9fa`，Apache-2.0](https://github.com/Liyan06/MiniCheck/tree/b58b9fa69acbd1015ec970fa65dd752413a053d2) | 可选离线声明/来源支持评分；不作为在线正确性保证 |
| CRAFT | [本地来源与 MIT 许可](../third_party/CRAFT_NOTICE.md) | 直接复用表格行文本函数，扩展 query 焦点与来源坐标；默认不启用额外表格注释 |

## A：从已见内容转为实际可见证据

ContextPilot 的 `reorder`、`deduplicate` 会更新索引、seen_docs 和统计，因此不能直接放进 A/R 的只读候选比较。`adapters/contextpilot.py` 按公开算法适配：

`distance = 1 - overlap / max(context_lengths) + 0.001 * mean_shared_position_difference`

匹配前缀中的共有材料提前，剩余材料保持检索顺序，依赖使用稳定顺序。只有仍在当前候选中具有正文 span 的正确版本，才能用复用注释省略正文；“归档过”不等于当前可见。证据先于变化的问题出现，给原生前缀缓存留下实际机会。默认 `stable_key` 与可选 `contextpilot` 共用证据、预算和版本规则。

本项目补充了稳定身份、版本冲突、条件依赖、明确回访、实际历史与执行账本分离，以及每轮相同的可引用范围。旧回答保留为历史，不自动获得当前证据资格。[FullCite](https://arxiv.org/html/2606.07130v1)区分来源归属、支持定位与 claim-citation 忠实性；本项目沿用这一区分，没有声称合法 ID 就能证明语义支持，也没有新增在线修订器。

多轮意图解析将同一独立问题贯穿检索、来源组织和 A/R 规划，并随回答持久提交。历史名称需要原文支持，用户的修正与新条件优先；不把助手旧回答当新事实。意图关系用于解释当前任务，不能单凭一个关系标签宣布原生缓存失效。实现与开销边界见[多轮解析](source_grounded_query_resolution.md)。

表格与 SDK 证据均是普通原文适配：显式行列、参数、初始化和依赖关系进入共同 A/R 输入，不私下为某个基线增加材料。使用见[表格关系](table_relations.md)、[SDK 证据](../guides/api_evidence.md)、[混合检索](../guides/hybrid_retrieval.md)。

最终材料选择参考[Elasticsearch结果折叠](https://www.elastic.co/docs/reference/elasticsearch/rest-apis/collapse-search-results)的最高排名组代表原则，在已有评分池内顺序选取不同内容组。组键为完整标题/正文，加上已知版本、父级范围和依赖；既有显式条件融合中的原任务锚点先保留，再填其余组。单视图分组保留原排名与前五段的不同内容组，空出的名额补入下一组。组不代表来源身份相等或独立佐证。无重复的普通轮次选择不变；新增材料适用性和输入代价仍需评价。将任务范围评分推广到所有补入轮次的检查未改善错产品材料，且会挤掉已覆盖的信息，该扩展已撤回；不为此继续调RRF权重。

[MMR原论文](https://www.cs.cmu.edu/afs/cs/Web/People/jgc/publication/MMR_DiversityBased_Reranking_SIGIR_1998.pdf)将相关性与已选内容相似度作加权取舍；本次使用精确分组，不引入相似度权重调参。[2026年10月预印本](https://arxiv.org/html/2610.09412v1)进一步讨论冗余程度和所需证据数量对多样化收益的影响，但部分触发器使用标注中的证据需求，不能当作现成在线能力。[AdaPCR](https://arxiv.org/html/2507.04069v1)以候选片段组合和回答似然训练体现互补性；其额外检索与训练不在当前实现中。上述原理用于选择有实际缺口的改进，不继承论文收益或额外建立模型支线。

## B：复用引擎真实状态和标准成本建模

原生报价负责实际 GPU/Host 前缀、共享硬锁和请求预算；控制器复用 lookup、阶段非负拟合、已完成历史的输出长度统计。容量预留依据引擎规则，服务时间依据经验长度，两者不混用。成本较低者被选中，误差只作诊断；没有为展示追加比例设置偏置。详见[联合原理](joint_optimization_model.md)、[原生等待估计](admission_wait_model.md)及[成本接口](../guides/stage_cost_model.md)。

标准固定阈值、重建、周期、稳定布局与冗余比例作为可运行参照保留。`warm_idle` 只是受 ReCAP 的缓存兼容思路启发的门控，没有持久注意力图，不能称完整复现。当前不扩大该方向。

## C：复用原生软引用，不建立影子缓存

固定 SGLang 的 [session_ref_tracker.py](https://github.com/sgl-project/sglang/blob/3f36fc08331f64cd63c4fbbdf39cc402f38bb38b/python/sglang/srt/mem_cache/unified_cache/session_ref_tracker.py) 管理软保护。每次请求仍传完整目标输入；撤销引用不保证物理缓存立即释放。ContextKV 将业务版本映射为 epoch，持久提交后再退役旧引用，失败保留 outbox 并恢复。

EfficientAgent 的 vLLM/LMCache connector 不能直接接到 SGLang。现有可选适配在合法 BackupKV action 的分配和 DMA 之前检查工作集压力与每请求累计页预算，沿用原生锁、祖先链、备份和 ACK。已有验证同时保留节省保留与增加自身重算的取舍，故当前不默认启用。见[Host 准入机制](host_admission_design.md)和[归属说明](../third_party/EFFICIENTAGENT_NOTICE.md)。

## 许可与可复现性

本轮引用资格直接复用固定 SGLang 的 [XGrammar regex backend](https://github.com/sgl-project/sglang/blob/3f36fc08331f64cd63c4fbbdf39cc402f38bb38b/python/sglang/srt/constrained/xgrammar_backend.py)，把本轮允许的 Evidence IDs 编成标准引用格式的正则约束。它限制标识符，不证明引用支持论断；原生对照使用同样的原生能力。

当前来源导航只保留ID、地址、标题和已声明章节，原有词项重合/IDF摘录已经移除；这类摘录不能可靠表达版本、平台和操作适用性。完整原文仍共同进入A/R。[Lost in the Middle](https://arxiv.org/abs/2307.03172) 提供长上下文位置敏感性的研究依据，不能替代本项目的质量验证，也不表示本实现复现了该论文方法。已有定点结果说明定位提示能促使部分回答，但仍可能发生适用对象泛化；不把它单独列为已证实的质量贡献。

直接复制的函数、提示与资产保留原许可证和 NOTICE；按公开原理独立实现的短算法注明来源和差异。模型权重、数据许可与代码许可分别记录，不将第三方数据和模型打入 Python 发行包。未进入当前服务的训练、复杂来源选择和离线评分工具保留复用价值，不能视作默认运行步骤或本次实测贡献。
