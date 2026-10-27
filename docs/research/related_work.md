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
| Docling | [文档层级](https://docling-project.github.io/docling/concepts/docling_document/) / [结构化分块与序列化](https://docling-project.github.io/docling/concepts/chunking/) | 采用保留原文层级、阅读顺序与来源坐标的原则；已有作者HTML时用标准库读取列表，不引入其转换/OCR流水线 |

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

## 本轮根因分析与方法取舍（2026-10-27）

现有反例中任务状态已正确，错误仍重复出现。根因不能继续全部归到查询改写：检索评分衡量相关性，子需求排名覆盖不保证流程兼容；片段常跨章节截断适用条件；模型还可能把历史答案或附近的另一平台说明当成当前事实。当前证据不足和证据充分但使用错误需要分开分析。

| 调研工作 | 可利用的原理 | 本项目取舍 |
|---|---|---|
| [CANARD，EMNLP 2019](https://aclanthology.org/D19-1605/) | 利用历史问答恢复省略与指代，不仅拼接旧问题 | 传递完整实际QA作为话语上下文，事实仍由当前来源支持；不引入训练 |
| [uva-irlab-conv，SemEval 2026 Task 8](https://arxiv.org/html/2606.11945v1) | 改写压缩可能丢失约束，后续列表重排重新接收对话历史 | 在现有解释器和联合选材中补齐实际QA；按预算采用近期及显式相关轮次，未完整复现其全历史/检索骨干。其检索强、生成仍有不足的结果也支持分别评价证据排序与回答有效性 |
| [Sufficient Context，ICLR 2025](https://arxiv.org/abs/2411.06037) | 区分相关材料与足以回答的材料，再分析生成错误 | 采用分层归因；不增加逐轮充分性judge或“一项不足就全部拒答”的闸门 |
| [Contextual Retrieval](https://www.anthropic.com/research/contextual-retrieval) | 将片段放回文档范围再构造检索表示，减少切块丢失的条件 | 优先利用原章节、标题、父级范围；派生说明仅辅助检索，原文保持事实来源。需要验证后再决定是否重建索引 |
| [QD-RAG，ACL SRW 2025](https://arxiv.org/abs/2507.00355) | 子问题召回、合并与已有重排 | 已复用；后续判断联合材料适用性，不能把更多子查询或排名覆盖当作答案收益 |
| [Lost in Conversation / Translation，2026预印本](https://arxiv.org/abs/2609.36700) | 分开定位多轮查询失真和已检索证据的综合失败 | 与当前反例一致的诊断框架；其大规模合成实验不替代本项目公开任务评测 |
| [CMT-RAG，2026预印本](https://arxiv.org/abs/2607.26470) | 子任务与证据依赖共同构成可回访的记忆 | 参考显式依赖；现有状态/原文图可先复用，不直接引入另一个训练骨干或庞大会话DAG |
| [SGLang Session Radix](https://docs.sglang.io/docs/advanced_features/session_radix_cache) / [HiCache](https://docs.sglang.io/docs/advanced_features/hicache_design) | 会话引用、原生淘汰与分层恢复；引用不是硬pin | 继续复用已验证的固定引擎；不为语义错误新增缓存策略或复制引擎管理器 |

公开benchmark接入和系统/执行比较边界已统一；当前从实际QA、阅读材料和对话行为的接口补齐信息，而非继续增设来源资格闸门。公开数据选用及精确发布规模见[评价](../data_and_evaluation.md)。这些方法的来源、原理和成本进入技术决策，论文收益不会直接写成ContextKV收益。

公开主比较后补充核对了[ICR（EMNLP 2025）](https://aclanthology.org/2025.emnlp-main.496/)与[Dialogue-RAG（ACL 2025）](https://aclanthology.org/2025.acl-long.1191/)。前者交替澄清与改写，后者以节点连接补齐省略；可供研究对象保持与低成本改写，但目前未移植其模型或声称已验证收益。原Cloud主比较中解析占约1.85秒，增加多轮LLM澄清必须有相应质量价值，不能仅因检索指标好就进入默认流程。[MLPerf End-to-End RAG（2026-08）](https://mlcommons.org/2026/08/endtoend-inference/)则提供完整多模型流水线计量的参考，其FRAMES多跳任务不能替代本项目的真实多轮主benchmark；不为套用它重新跑一套实验。当前优先复用既有日志进行任务理解、资料充分性和生成使用的分层归因。

解析解码直接复用固定SGLang的[NGRAM worker](https://github.com/sgl-project/sglang/blob/3f36fc08331f64cd63c4fbbdf39cc402f38bb38b/python/sglang/srt/speculative/ngram_worker.py)：CPU侧缓存已生成片段，目标模型对草稿运行验证，先应用语法mask再选取greedy目标token。本项目沿用其FA3/page64线性草稿能力，没有另建草稿模型或缓存器。[SuffixDecoding（NeurIPS 2025）](https://arxiv.org/abs/2411.04975)提供重复请求可利用片段复用的研究依据，当前采用的是引擎现成实现。已完成的有限诊断和部署参数见[评价](../data_and_evaluation.md#解释服务的原生解码优化)；原生验证不等于不同执行形状下保证浮点和输出逐字一致。

[LLMs Get Lost In Multi-Turn Conversation（2025）](https://arxiv.org/abs/2505.06120)分析早期假设与旧答案依赖的累积影响，重述任务只能部分缓解。当前系统已有任务重建与历史非事实提示；本项目观察到的重复SDK错误同时跨越APPEND和REBUILD，与此类问题相符，但不据此宣称已证明具体因果。后续改进应针对任务/来源适用性及生成使用，不能把这些失败简化为缓存动作错误或继续堆叠拒答规则。


## 任务投影与统一评测的补齐

[TurnFlow（EMNLP Findings 2026）](https://aclanthology.org/2026.findings-emnlp.825/)强调需要分别检查需求状态的识别、修订与最终执行。沿这一问题分解，本轮代码检查定位到一个确定的信息传递缺口：`frame_query`在requirements非空时舍弃完整task，可能丢失“除此之外”等限定语。现保留完整任务及明确子需求，单项需求查询仍各自聚焦；这是既有状态到检索接口的修复，没有新增推理阶段，也未声称解决了所有多轮误解。

[MultiDoc2Dial](https://aclanthology.org/2021.emnlp-main.498/)提供跨文档服务咨询，已复用作者语料、分区、对话和grounding，接到共同P3运行链；连续同角色发言的交流轮投影明确记录。[ChatRAG-Bench](https://huggingface.co/datasets/nvidia/ChatRAG-Bench)可补充给定历史下的多任务质量，[Loong](https://github.com/MozerWang/Loong)可补充确需多资料综合的长文本任务。它们按各自协议评价，公共数据不足的版本演化/资源竞争允许完整扩展，方法与评测不再局限于旧Cloud开发案例。公开覆盖、实现状态和分版本实际评分统一见[评价](../data_and_evaluation.md)。

## 结构证据与近期同场景方法

MultiDoc2Dial接入进一步复用[作者的章节分组规则](https://github.com/IBM/multidoc2dial/blob/2456a5609b447e1a12b716f4d658728f346f3041/scripts/data_preprocessor.py)，相邻同节或同标题成组；本项目保留连续原文offset、完整正文与父级标题，而非用grounding答案选段。488篇全文导出4110段，均能逐字恢复原文；该语料的BGE重排窗口使用8192，避免无意截断章节条件。两组共同使用这一适配，不把预处理差异当ContextKV独占收益。

本轮补充审阅[Howard-AI4PC（SemEval 2026）](https://aclanthology.org/2026.semeval-1.320/)和[AILS-NTUA（2026）](https://arxiv.org/html/2603.10524v1)：前者用混合检索、统一重排及贴近原文的回答组合，后者将原文证据定位与生成分解，并报告过检索反馈改写的负结果。可利用的是保留任务约束、区分召回缺口与选材/使用错误，以及原文定位；不直接叠加其多次改写、双候选及多judge流程。这里根据当前成本和反例作适配选择，不预设这些方法在本系统有效或无效。

[CORAL论文与公开实现](https://github.com/RUC-NLPIR/CORAL)另外提供开放领域、主题切换和引用标注的统一任务，适合补充目前服务咨询与技术资料的覆盖。该数据是自动构建，不能替代真实用户交互；本轮只核对论文、格式及发布入口，没有另启一套生成实验。
