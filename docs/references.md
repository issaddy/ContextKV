# 参考工作

保留当前机制的来源及与下一步瓶颈直接相关的工作。实现适配见[架构](architecture.md)，运行效果见[数据与评价](data_and_evaluation.md)。

## 当前机制依据

- [SGLang](https://github.com/sgl-project/sglang)：结构化生成、前缀缓存和高性能大模型推理服务。
- [Session-Aware Radix Cache](https://docs.sglang.io/docs/advanced_features/session_radix_cache)：通过会话软引用管理可复用前缀的缓存生命周期。
- [HiCache](https://docs.sglang.io/docs/advanced_features/hicache_design)：GPU、Host及存储层之间的分层KV缓存管理。
- [SGLang NGRAM Worker](https://github.com/sgl-project/sglang/blob/3f36fc08331f64cd63c4fbbdf39cc402f38bb38b/python/sglang/srt/speculative/ngram_worker.py)：利用历史输出片段构造草稿并由目标模型验证。
- [ContextPilot](https://arxiv.org/abs/2511.03475)：通过上下文索引、重排、去重和注释促进长上下文复用。
- [TokenPilot / LightRSI](https://arxiv.org/abs/2606.17016)：结合稳定前缀与上下文生命周期清理降低多轮执行成本。
- [EfficientAgent](https://arxiv.org/abs/2609.33762)：分析并发工作集、KV卸载和Host备份准入。
- [BM25S](https://github.com/xhluca/bm25s)：提供基于稀疏矩阵的高效BM25检索。
- [BGE-M3](https://huggingface.co/BAAI/bge-m3)：提供多语言、多粒度的文本检索表示。
- [BGE-reranker-v2-m3](https://huggingface.co/BAAI/bge-reranker-v2-m3)：对查询与候选文档进行交叉编码相关性打分。
- [Reciprocal Rank Fusion（RRF）](https://plg.uwaterloo.ca/~gvcormac/cormacksigir09-rrf.pdf)：依据名次倒数融合多个检索结果列表。
- [xQuAD](https://terrierteam.dcs.gla.ac.uk/publications/ecir2010_rodrygo_div.pdf)：通过查询方面覆盖实现搜索结果多样化。
- [SOM-DST](https://aclanthology.org/2020.acl-main.53/)：通过保留、更新和删除操作增量维护对话状态；原文分析指出状态操作预测与真实操作之间的准确率差距，是进一步改进的方向。
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
- [Faithful Query Resolution](https://arxiv.org/abs/2610.15034)：区分忠实意图解析与扩大检索覆盖的查询扩展；检索排序收益与交互回答效果分别验证。
- [S2G-RAG](https://aclanthology.org/2026.acl-long.1185/)：用训练后的充分性/缺口判断器指导迭代检索，并通过原文句子指针限制证据增长。
- [ExecutionRAG](https://arxiv.org/abs/2610.21184)：维护未解决子问题及其支持证据以减少检索偏移，并分析证据找回后仍存在的过早完成与拒答。
- [Evidence-based Generation for Long Context QA](https://arxiv.org/abs/2610.26951)：研究多资料长上下文的答案与正文证据联合正确性、来源结构及片段不连续标识；比较时需区分已含正确来源与开放语料检索的证据访问条件。
- [Lost in the Middle](https://arxiv.org/abs/2307.03172)：研究信息位置对长上下文问答效果的影响。
- [ContextualRetriever](https://aclanthology.org/2025.emnlp-main.602/)：在会话上下文中突出当前查询，通过改写意图监督和保留生成能力的训练学习检索表示，减少额外自回归改写。
- [HAConvDR](https://aclanthology.org/2024.findings-acl.792/)：结合历史去噪和历史轮次实际检索贡献构造监督，训练会话稠密检索；重点处理长历史噪声与话题切换。
- [ConvDR](https://arxiv.org/abs/2105.04166)：继承文档编码，用独立问题的教师向量和检索排序损失训练会话查询表示，省去在线生成改写。
- [DRAGged into Conflicts](https://arxiv.org/abs/2506.08500)：区分互补、时效、意见与错误信息等来源冲突。
- [When Knowledge Changes](https://arxiv.org/abs/2607.26843)：通过语料更新与变形测试检验RAG对无关变化的稳定性及对事实变化的响应，补充静态快照评价。

## 评价数据

- [MT-RAG](https://arxiv.org/abs/2501.03468)：提供多领域、多轮对话的检索增强问答基准。
- [MTRAG-UN](https://github.com/IBM/mt-rag-benchmark/tree/main/mtragun-human)：覆盖歧义、非独立问题、不可答和部分可答的多轮RAG挑战任务。
- [MultiDoc2Dial](https://aclanthology.org/2021.emnlp-main.498/)：提供跨文档切换与服务咨询场景的多轮问答。
- [Doc2Dial](https://aclanthology.org/2020.emnlp-main.652/)：提供带文档范围和支持跨度的目标导向对话。
- [Loong](https://github.com/MozerWang/Loong)：评价长上下文中的多资料定位、比较、聚合与推理。
- [LongMemEval](https://github.com/xiaowu0162/LongMemEval)：评价长期会话记忆、时间推理与知识更新。
- [CORAL](https://github.com/RUC-NLPIR/CORAL)：提供知识密集、主题切换和引用标注的多轮问答数据。
