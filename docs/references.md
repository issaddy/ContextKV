# 参考工作

保留当前机制的来源及与下一步瓶颈直接相关的工作。实现适配见[架构](architecture.md)，运行效果见[数据与评价](data_and_evaluation.md)。

## 机制选型与复用方向（2026-11-02收尾）

本轮结合当前源码与公开原文复核机制，补查2026年9—10月工作。选型以任务效果和完整成本为准，可复用有效算法、训练信号、数据组织与工程实现。模型、领域、负载和实现差异用于设计适配与实验。下表分别记录已采用机制、已做实验和待验证方向；本次实验结论对应具体适配、数据与预算，原方法的其他能力保留为可复用方向。

| 研究方向与原始工作 | 核心方法及与本课题的关系 | 当前采用状态与选型判断 |
|---|---|---|
| [SOM-DST](https://aclanthology.org/2020.acl-main.53/)、[ConvDR](https://arxiv.org/abs/2105.04166)、[HAConvDR](https://aclanthology.org/2024.findings-acl.792/) | 分别通过选择性状态覆盖、会话查询表示蒸馏、历史去噪与检索贡献监督处理多轮输入 | 已实现开放任务增量更新与实际QA查询适配；后两项的完整原论文训练流程未复现。保留会话整体对照后，默认采用实际QA查询适配＋Dense5稠密检索＋重建，结构化状态保持可选；按真实回答和完整成本选择。 |
| [Contextual Retrieval](https://www.anthropic.com/research/contextual-retrieval)、[SetR](https://aclanthology.org/2025.acl-long.861/)、[Sufficient Context](https://arxiv.org/html/2411.06037v3) | 文档背景补充、需求驱动集合选材、区分材料充分性与生成使用 | 当前使用原文结构/窗口及排名边际覆盖。133轮离线32B诊断及11题原文复核发现漏读条件、来源范围与标签不一致，诊断保留为分析工具。SetR训练和充分性判断器的任务适配仍是待验证方向。 |
| [Adaptive-k，EMNLP 2025](https://aclanthology.org/2025.emnlp-main.1017/)、[作者实现](https://github.com/megagonlabs/adaptive-k-retrieval) | 在原始余弦相似度序列上找最大下降，忽略末尾10%，补取5段；无需额外生成调用 | 已复用作者阈值函数，完成三方案441份回答。开发集评分缺失界限：Adaptive-k 0.833–0.841、固定稠密五段0.826–0.841；完整响应7.48/5.41秒，实际跨度覆盖72.9%/63.9%。实验确认扩大覆盖的收益及相应成本；本轮预算下选Dense5作为默认，自适应选材保留为不同覆盖需求下的候选。 |
| [CLEAR，EMNLP 2026](https://aclanthology.org/2026.findings-emnlp.802/) | 答案—片段蕴含蒸馏，并通过片段侧生成查询扩充召回 | 已试原文跨度监督BGE LoRA，属于较轻的支持性排序实验，未复现双头蕴含蒸馏。保留集跨度覆盖79.6%→86.9%，闭环开发回答RB_llm 0.826→0.802；后续监督需面向实际会话与回答效用。 |
| [Faithful Query Resolution，2026-10-10](https://arxiv.org/html/2610.15034) | 忠实解析与扩展查询分别表示，通过加权RRF融合；检索收益与交互收益分别验证 | 已有多视角融合实验，原论文iKAT流程待复现。可复用意图解析与查询扩展分工，同时评价排序和实际多轮回答。 |
| [ExecutionRAG，2026-10-15](https://arxiv.org/html/2610.21184) | 由初检建立未解决需求记录，逐项检索并记录支持资料；找回更多材料后仍观察到过早完成 | 当前只有需求列表及覆盖选材，尚无该迭代记录算法。优先借鉴逐需求诊断，增量检索应以实际缺口和完整调用成本决定是否值得开展。 |
| [LongRAG（Jiang等）](https://arxiv.org/html/2406.15319v3) | 完整文档/相关文档组作为长阅读单元，以MaxP聚合子块得分 | 已复核论文与作者仓库。本次完整父文档适配的133轮对照：跨度覆盖0.639→0.812，回答得分0.836→0.804、完整响应4.18→5.98秒，默认仍用片段阅读。论文的链接分组和两次答案抽取属于待验证组件，可分别适配。 |
| [REPLUG LSR](https://arxiv.org/html/2301.12652v4)、[RAG-DDR](https://arxiv.org/html/2410.13509) | 用冻结LM条件似然或最终回答奖励学习检索/模块选择 | 已采集1600训练会话6424对冻结32B答案token似然，按软分布监督BGE LoRA；完整294次生成及评分：公开RB_llm 0.809→0.819、区间跨零，完整响应增加0.34秒；本次训练权重作为实验资产保存。属于QA重排适配，未复现REPLUG生成集成或RAG-DDR整套训练。 |
| [ContextPilot，MLSys 2026](https://arxiv.org/html/2511.03475v3) | 层次上下文索引、按重叠对齐顺序、去重注释、跨请求调度 | 现有`adapters/contextpilot.py`为会话内原理适配，对照名`contextpilot`使用优先追加。另已接入固定v0.5.0作者索引/排序/去重及新版SGLang事件桥，13次真实生成验证了复用和淘汰联动；另完成133轮公开与14轮自建任务、两方法共294次生成及评分。当前样本每轮重建的质量和完整耗时优于作者追加方案；长上下文与容量转折仍需比较。 |
| [RAGCache](https://arxiv.org/html/2404.12457)、[CacheBlend](https://arxiv.org/abs/2405.16444)、[Cache-Craft](https://skejriwal44.github.io/docs/CacheCraft_SIGMOD_2025.pdf) | 分别研究成本/大小/频率驱动的分层缓存，及非前缀块KV的选择性重算 | 当前B采用真实阶段成本，C复用原生前缀分层缓存；未实现这些论文的替换或融合算法。完整响应与动态来源条件下的追加/重建增量须独立验证。 |
| [TokenPilot，2026-08修订](https://arxiv.org/html/2606.17016) | 稳定前缀、输入结构压缩、依据片段剩余任务效用成批清理 | 待验证。可将剩余任务效用用于历史QA和文档证据的保留，联合测量回答完整性与前缀复用，比较固定周期清理。 |
| [Continuum/Continnum，2026-09修订](https://arxiv.org/html/2511.02230)、[EfficientAgent，2026-09-27](https://arxiv.org/html/2609.33762) | 用回载/排队代价决定KV保留TTL；用并发复用工作集决定Host容量与写入准入 | 已有真实停顿实验及Host备份预算接口。TTL与容量条件准入算法待验证，可复用到会话回访，按GPU命中、Host回载和重算分别计量。 |
| [Tessera，2026-09](https://arxiv.org/html/2609.32999) | 检索需求、驻留位置与生成负载共同驱动缓存保留及路由，采用EPIC组合执行 | 待验证。需求驱动控制可先适配当前单生成服务，再分别评价跨节点池与非前缀组合执行。 |
| [RoleKV，EMNLP 2026](https://aclanthology.org/2026.emnlp-main.1982/)、[TOPAS，2026-08-26](https://arxiv.org/abs/2608.25523) | 分别按语义角色衰减管理驻留、按工作流剩余关键路径与前缀收益联合选择驻留和调度 | 本轮完成文献筛查，适配待验证。可将角色映射为指令、资料、历史QA，将工作流依赖映射为检索、生成和会话回访，比较其驻留与调度收益。 |

本课题机制是“实际任务与来源约束下的上下文组织、上下文策略选择、业务代际与原生KV引用协同”。状态更新、边际覆盖、事务outbox和Radix/HiCache均有已有方法基础；重点评价它们如何共同改善资源受限多轮文档问答。当前推荐固定重建与原生前缀复用，可选成本策略执行单轮选择；跨轮未来收益与直接质量—内存联合优化尚未实现。

研究比较按三个问题组织：①固定检索/生成预算下，实际会话中的需求支持与最终得分是否改善；②固定A及共同证据，成本选择相对追加、重建、固定周期、ContextPilot原理适配能否减少完整耗时，作者完整系统另列；③固定模型与总KV预算，改变上下文长度、并发与停顿，实测GPU/Host工作集及回载收益。整体部署共享一个32B带来的资源下降单列，不归因为B/C独立收益。

## 当前机制依据

- [SGLang](https://github.com/sgl-project/sglang)：结构化生成、前缀缓存和高性能大模型推理服务。
- [Session-Aware Radix Cache](https://docs.sglang.io/docs/advanced_features/session_radix_cache)：通过会话软引用管理可复用前缀的缓存生命周期。
- [HiCache](https://docs.sglang.io/docs/advanced_features/hicache_design)：GPU、Host及存储层之间的分层KV缓存管理。
- [SGLang NGRAM Worker](https://github.com/sgl-project/sglang/blob/3f36fc08331f64cd63c4fbbdf39cc402f38bb38b/python/sglang/srt/speculative/ngram_worker.py)：利用历史输出片段构造草稿并由目标模型验证。
- [SGLang确定性推理](https://github.com/sgl-project/sglang/blob/main/docs/docs/advanced_features/deterministic_inference.mdx)：说明temperature 0仍可能因GPU计算方式变化而产生不同输出，提供确定性模式；其代价与兼容性需在本项目实测。
- [ContextPilot](https://arxiv.org/abs/2511.03475)：通过上下文索引、重排、去重和注释促进长上下文复用。
- [Continuum](https://arxiv.org/abs/2511.02230)：结合任务暂停时间、KV保留时限与作业调度，衡量多轮代理的完成时间；可将暂停预测与保留时限适配到文档问答回访。
- [KVCache Cache in the Wild](https://www.usenix.org/conference/atc25/presentation/wang-jiahao)：实测服务负载的复用时间和概率分布，可用于设计贴近实际停顿分布的缓存实验。
- [CacheBlend](https://arxiv.org/abs/2405.16444)：为复用非前缀资料缓存选择性重算受上下文影响的KV；可与当前原生前缀缓存比较复用覆盖、回答质量和重算成本。
- [Tessera](https://arxiv.org/abs/2609.32999)：用检索需求、驻留状态和负载协同缓存及路由；其专门的组合执行为跨片段KV复用提供实现方向。
- [TokenPilot / LightRSI](https://arxiv.org/abs/2606.17016)：结合稳定前缀与上下文生命周期清理降低多轮执行成本。
- [EfficientAgent](https://arxiv.org/abs/2609.33762)：分析并发工作集、KV卸载和Host备份准入。
- [BM25S](https://github.com/xhluca/bm25s)：提供基于稀疏矩阵的高效BM25检索。
- [BGE-M3](https://huggingface.co/BAAI/bge-m3)：提供多语言、多粒度的文本检索表示。
- [BGE-reranker-v2-m3](https://huggingface.co/BAAI/bge-reranker-v2-m3)：对查询与候选文档进行交叉编码相关性打分。
- [Reciprocal Rank Fusion（RRF）](https://plg.uwaterloo.ca/~gvcormac/cormacksigir09-rrf.pdf)：依据名次倒数融合多个检索结果列表。
- [xQuAD](https://terrierteam.dcs.gla.ac.uk/publications/ecir2010_rodrygo_div.pdf)：通过查询方面覆盖实现搜索结果多样化。
- [DF-RAG](https://arxiv.org/abs/2601.17212)：以MMR权衡相关性和多样性，按问题选择多样性程度；可同时比较证据互补性、最终回答和额外调用成本。
- [How retriever redundancy and diversity impact RAG effectiveness](https://arxiv.org/abs/2608.13956)：在虚构问答中控制答案内容，比较重复、改写和不同文体的材料；可借鉴其控制变量方法，加入版本差异进行评价。
- [Contradictions in Context](https://arxiv.org/abs/2511.06668)：分析时间不同的相似医疗文献中的冲突，可用于细化来源时间、适用条件与正文相似度的评价。
- [SOM-DST](https://aclanthology.org/2020.acl-main.53/)：通过保留、更新和删除操作增量维护对话状态；原文分析指出状态操作预测与真实操作之间的准确率差距，是进一步改进的方向。
- [Correctable-DST](https://aclanthology.org/2022.emnlp-main.56/)：显式检测历史状态中过多、缺失和待更新的字段，缓解预测状态错误累积；可将监督信号适配到开放任务字段。
- [Beyond Continuity](https://arxiv.org/abs/2605.09268)：将话题切换与旧上下文过度继承分开衡量，并报告开放模型的状态惯性；受控拼接对话可与自然多轮任务互补评价。
- [Reset Is Not Recovery](https://aclanthology.org/2026.groundlm-1.53/)：研究虚假历史被撤回后模型仍受旧上下文影响；可用于设计条件纠正后实际回答与当前原文的一致性评价。
- [Getting Better Dialogue Context for Knowledge Identification](https://aclanthology.org/2022.sigdial-1.36/)：研究多文档对话中的历史选择与文档级话题切换；可比较主题关系的预测与历史选材收益。
- [IterCQR](https://aclanthology.org/2024.naacl-long.449/)：用检索反馈训练改写器；可在固定Qwen3-32B下复用反馈目标，并同时记录检索与回答收益。
- [GuideCQR](https://arxiv.org/abs/2407.12363)：用初检文档提取的线索辅助会话查询，并结合当前问题与历史过滤；其阅读理解模型和多阶段检索开销需另计。
- [ReformIR](https://arxiv.org/abs/2605.00560)：在固定重排预算内用排序反馈选择改写和候选，减少查询漂移；可适配到真实多轮历史下的查询选择。
- [MoNET](https://aclanthology.org/2023.findings-acl.33/)和[Granular Change Accuracy](https://aclanthology.org/2024.lrec-main.699/)：分别研究状态惯性及状态变化评价，提示应区分正确保留、更新和删除，而非只看最终整帧匹配。
- [Context-Aware Dialogue State Tracking](https://aclanthology.org/N19-1057/)：根据状态字段定位相关历史上下文。
- [Contextual Retrieval](https://www.anthropic.com/research/contextual-retrieval)：为片段补充文档背景以改善检索表示。
- [Docling](https://docling-project.github.io/docling/concepts/docling_document/)：保留文档层级、阅读顺序和来源坐标的结构化解析与分块。
- [SPIRE](https://arxiv.org/abs/2604.20849)：组织标题、父级结构及原文片段，保留证据的解释上下文。
- [CRAFT](https://arxiv.org/abs/2505.14984)：通过级联检索、表格行表示和相关行选择处理表格问答。
- [TabRAG](https://arxiv.org/abs/2511.06582)：将表格单元格、行列、单位和区域关系组织为检索表示。
- [WAIT / Nested WAIT](https://arxiv.org/abs/2504.11320)：借助流体模型与动态内存需求安排推理请求。
- [Landlord](https://www.cs.ucr.edu/~neal/publication/Young02Online.pdf)：结合对象大小和取回成本进行在线缓存替换。
- [XGrammar](https://github.com/mlc-ai/xgrammar)：使用语法和正则约束高效生成结构化输出。

- [JSON Merge Patch](https://www.rfc-editor.org/info/rfc7396/)：以省略、赋值和空值表达保留、更新与删除。
- [ParentDocumentRetriever](https://reference.langchain.com/python/langchain-classic/retrievers/parent_document_retriever/ParentDocumentRetriever)：以子片段召回并恢复父级阅读上下文。

## 需求、充分性与回答使用的后续参考

- [Sufficient Context](https://arxiv.org/abs/2411.06037)：区分检索相关性、材料充分性与生成阶段错误。
- [SELF-multi-RAG](https://aclanthology.org/2024.findings-emnlp.622/)：联合学习何时检索、怎样表示会话和评估所用来源；可复用检索时机和来源评价训练，联合计量回答、辅助调用与缓存成本。
- [CONQRR](https://aclanthology.org/2022.emnlp-main.679/)：用检索反馈训练会话问题改写；可与直接会话编码比较召回、实际回答及在线成本。
- [S2G-RAG](https://aclanthology.org/2026.acl-long.1185/)：用训练后的充分性和缺口判断器控制迭代检索，并通过原文句子指针限制证据增长；其多次模型调用和事实抽取需纳入完整成本。
- [Beam Retrieval](https://aclanthology.org/2024.naacl-long.96/)：多跳检索保留多个候选证据链，减少过早固定单一路径；候选链搜索与监督信号可分别适配现有BGE检索。
- [Learning to Plan and Generate Text with Citations](https://aclanthology.org/2024.acl-long.615/)：在生成前组织问题与来源的对应计划以改善归因；可复用计划表示、训练信号或多阶段生成流程。
- [Chain-of-Verification](https://aclanthology.org/2024.findings-acl.212/)：通过独立核查草稿问题减少幻觉；可测量固定32B下核查带来的质量增量与调用成本。
- [GigitAI / MTRAGEval](https://aclanthology.org/2026.semeval-1.389/)：在其任务设置中，简单生成与复杂RAG策略效果接近，欠明确问题仍较弱；可复用其比较思路，关注消歧需求及完整成本。
- [Preserving Evidence Anchors](https://aclanthology.org/2026.groundlm-1.62/)：检测查询改写是否丢失证据锚点并在必要时恢复；可结合当前条件选择适用锚点。
- [SAGA](https://aclanthology.org/2026.emnlp-main.702/)：分别表示对话事实与来源适用前提，并融合语义和结构检索；可将来源侧表示适配到本课题的跨领域资料。
- [VersionRAG](https://arxiv.org/abs/2510.08109)：按版本意图选择检索路径，并在VersionQA上评价文档演化；意图与版本分离、图结构可分别复用和比较。
- [TimelyRAG](https://aclanthology.org/2026.findings-emnlp.398/)：对修订内容重叠的文档结合语义与时间排序；可用来源生效时间构造版本排序特征。
- [MRAG / TempRAGEval](https://aclanthology.org/2025.findings-emnlp.167/)：将问题内容与时间约束分开并分别排序；公开数据带金证据句，可用于受控时间适用性评价；原论文开放检索规模约3,310万维基片段。
- [EviRerank](https://aclanthology.org/2026.findings-emnlp.704/)：按查询相关块自适应分配重排输入预算，并保留全局线索；可将预算分配与后续完整原文阅读分别评价。
- [Attribution–Compression Frontier](https://aclanthology.org/2026.groundlm-1.65/)：对比压缩后摘要与原文的可归因性；实验依赖自动评判，提示压缩效果需要同时验证原文支撑。
- [SetR](https://aclanthology.org/2025.acl-long.861/)：以信息需求为单位选择互补的证据集合，强调集合充分性而非各片段的独立相关性。
- [When Relevance Is Not Evidence](https://aclanthology.org/2026.akbc-1.7/)：分析主题相关排序与实际支持之间的差异；可复用其受控候选实验诊断支持性，再接入开放语料闭环。
- [Faithful Query Resolution](https://arxiv.org/abs/2610.15034)：区分忠实意图解析与扩大检索覆盖的查询扩展；检索排序收益与交互回答效果分别验证。
- [Explicit Query Rewriting](https://aclanthology.org/2022.emnlp-main.311/)：以编辑原问题的方式恢复对话中的省略和指代，并训练专用检索模型；可分别比较编辑表示与训练监督。
- [CompCQR](https://aclanthology.org/2026.findings-emnlp.1225/)：组合原问题、改写及假设答案形成多种检索视角，再融合与选择；[作者实现](https://github.com/YunahJang/CompCQR)每轮三次模型调用、最多325个检索变体，可复用视角组织或完整流程并计量各自成本。
- [Sub-Question Coverage](https://aclanthology.org/2025.naacl-long.301/)：分开评价问题各方面在材料和回答中的覆盖，避免仅以整题相关性衡量完整性。
- [RAGChecker](https://arxiv.org/abs/2408.08067)：通过参考答案与生成答案的事实主张及其支持关系，分别诊断材料覆盖和回答使用；模型判定仍需人工核对。
- [GroUSE](https://aclanthology.org/2025.coling-main.304/)和[ContextualJudgeBench](https://aclanthology.org/2025.acl-long.470/)：用针对证据约束与上下文的评价样例检查评判器，补充单一模型或单一参考答案的打分。
- [Correctness and Faithfulness in QA](https://aclanthology.org/2024.tacl-1.38/)：分析长回答使EM/F1失真的问题，分别评价满足信息需求和是否受资料支持。
- [Brevity and Sustainability](https://aclanthology.org/2025.findings-acl.1125/)：比较回答长度与能耗控制；其质量判断主要使用词面重叠，简短指令是否保留多部分需求仍需在本项目中实测。
- [ExecutionRAG](https://arxiv.org/abs/2610.21184)：维护未解决子问题及其支持证据以减少检索偏移，并分析证据找回后仍存在的过早完成与拒答。
- [Evidence-based Generation for Long Context QA](https://arxiv.org/abs/2610.26951)：研究多资料长上下文的答案与正文证据联合正确性、来源结构及片段不连续标识；比较时需区分已含正确来源与开放语料检索的证据访问条件。
- [Lost in the Middle](https://arxiv.org/abs/2307.03172)：研究信息位置对长上下文问答效果的影响。
- [Lost in Conversation or Translation](https://arxiv.org/abs/2609.36700)：用完整问题、拼接线索和分轮线索区分改写造成的检索变化与跨轮证据使用失败；可将这种分解用于实际生成历史的评价。
- [ContextualRetriever](https://aclanthology.org/2025.emnlp-main.602/)：在会话上下文中突出当前查询，通过改写意图监督和保留生成能力的训练学习检索表示，减少额外自回归改写。
- [Dialogue-RAG](https://aclanthology.org/2025.acl-long.1191/)：以话语节点连接与省略补全改善对话检索，并提供中英文对话RAG数据；可复用话语结构、补全监督和数据，在固定32B下比较。
- [ICR](https://aclanthology.org/2025.emnlp-main.496/)：通过迭代澄清消解多种可能解释；额外交互和调用成本需与直接回答的整体收益一起衡量。
- [HAConvDR](https://aclanthology.org/2024.findings-acl.792/)：结合历史去噪和历史轮次实际检索贡献构造监督，训练会话稠密检索；重点处理长历史噪声与话题切换。
- [Leveraging historical information to boost RAG in conversations](https://www.sciencedirect.com/science/article/pii/S0306457325003905)：比较复用历史检索结果、轮次依赖及历史证据的策略；历史来源在版本或条件变化后是否仍适用，需独立验证。
- [ConvDR](https://arxiv.org/abs/2105.04166)：继承文档编码，用独立问题的教师向量和检索排序损失训练会话查询表示，省去在线生成改写。
- [RCEM](https://aclanthology.org/2026.emnlp-industry.167/)：把会话向量对齐独立改写并约束原有表示空间，以减少分布变化下的退化；可作为现有对话适配器跨领域效果不足时的训练对照，仍需计入离线教师与训练成本。
- [CLEAR](https://aclanthology.org/2026.findings-emnlp.802/)：以训练期的答案—资料蕴含信号监督重排器，推理时不依赖答案；对本项目需换用符合固定模型约束的监督，并测量额外重排成本及版本、条件适用性。
- [REPLUG LSR](https://arxiv.org/abs/2301.12652)：冻结生成模型，以其条件似然监督检索器；可参考固定32B下的离线回答效用监督，需与原文支持和实际生成得分共同验证。
- [RAG-DDR](https://arxiv.org/abs/2410.13509)：通过模块输出扰动和最终回答奖励对齐RAG模块的数据偏好；本项目可参考离线奖励采集，保持32B生成权重固定，并计入训练数据采集成本。
- [DRAGged into Conflicts](https://arxiv.org/abs/2506.08500)：区分互补、时效、意见与错误信息等来源冲突。
- [MAGIC](https://aclanthology.org/2025.findings-emnlp.466/)：构造可解释的多跳资料冲突，报告模型容易漏检冲突或归错来源；可复用冲突构造方式补充动态版本会话。
- [Context-Driven Decomposition](https://aclanthology.org/2026.groundlm-1.21/)：区分模型先验、所给资料与冲突前提，分析正确答案背后的资料使用机制；可配合版本切换实验评价来源使用与成本。
- [When Knowledge Changes](https://arxiv.org/abs/2607.26843)：通过语料更新与变形测试检验RAG对无关变化的稳定性及对事实变化的响应，补充静态快照评价。
- [Re3](https://aclanthology.org/2026.acl-long.1180/)：训练时间感知检索与冲突仲裁来处理过时资料；可按用户询问的时间范围同时评价当前版本与历史版本。

## 评价数据

- [MT-RAG](https://arxiv.org/abs/2501.03468)：提供多领域、多轮对话的检索增强问答基准。
- [MTRAG-UN](https://github.com/IBM/mt-rag-benchmark/tree/main/mtragun-human)：覆盖歧义、非独立问题、不可答和部分可答的多轮RAG挑战任务。
- [MultiDoc2Dial](https://aclanthology.org/2021.emnlp-main.498/)：提供跨文档切换与服务咨询场景的多轮问答。
- [Doc2Dial](https://aclanthology.org/2020.emnlp-main.652/)：提供带文档范围和支持跨度的目标导向对话。
- [Loong](https://github.com/MozerWang/Loong)：评价长上下文中的多资料定位、比较、聚合与推理。
- [LongMemEval](https://github.com/xiaowu0162/LongMemEval)：评价长期会话记忆、时间推理与知识更新。
- [CORAL](https://github.com/RUC-NLPIR/CORAL)：提供知识密集、主题切换和引用标注的多轮问答数据。
- [RECOR](https://aclanthology.org/2026.findings-acl.129/)：提供多领域、多轮推理检索及文档级相关标注；其合成对话和检索任务与完整问答、动态版本评价应分别报告。
- [QReCC](https://aclanthology.org/2021.naacl-main.44/)：提供多轮问题、人工独立改写和回答，部分对话由单轮题构造；其开放语料约5,400万片段，分区实验应说明覆盖范围。
- [TopiOCQA](https://aclanthology.org/2022.tacl-1.27/)：开放域多轮问答包含主题切换及人工标注来源；作者指出答案跨度匹配对其抽象式回答并不充分，可补充评估状态切换和来源召回。
- [BurstGPT v2](https://github.com/HPMLL/BurstGPT/releases/tag/v2.0)：公开真实服务请求时间、会话标识与完整响应耗时；时间轨迹可与带正文的问答任务配合，分别测量负载和质量。
