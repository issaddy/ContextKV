# 当前方法的复用依据

| 相关工作 | ContextKV 采用的机制 | 项目实现 |
|---|---|---|
| [Qwen3 Embedding](https://github.com/QwenLM/Qwen3-Embedding/blob/main/README.md#L82) | 复用作者 `last_token_pool` 与 L2 归一化；左 padding 下取末 token。共享查询前向供任务头使用，文档编码保持冻结 base 空间 | `SharedQueryEncoder.encode_many`、`encode_documents_with_usage` |
| [RCEM §3，公式4–7](https://arxiv.org/html/2606.01697v2#S3) | 采用会话→改写查询的点 L2 对齐、普通查询自对齐和成对距离 SL（λ=1）；文档通过关闭查询适配器直接保持冻结空间。复用现有 LoRA 与共享任务头实现，未找到公开的作者训练源码 | `training/shared_training.py:task_loss` |
| [Jina-reranker-v3.5 作者源码](https://huggingface.co/jinaai/jina-reranker-v3.5/blob/main/modeling.py#L81) | 复用骨干、末层 marker 池化和原 projector，并接通作者 [`rerank` 跨批 query 加权 cosine](https://huggingface.co/jinaai/jina-reranker-v3.5/blob/main/modeling.py#L237)。项目按来源地址组织输入，字段与集合头共享这些表示 | `SharedEvidenceEncoder.forward_evidence`、`EvidenceEncoding.merge` |
| [OptiSet §3.2–3.3](https://arxiv.org/html/2601.05027v1#S3.SS2) | 借鉴相对无证据基线的生成器集合效用：本项目用真实答案 ΔQ=Q(S)−Q(∅)，以 MSE＋集合列表 KL 训练共享集合头。使用冻结 Jina 表示的语义投影、均值/最大值及成对交互预测有界增益，在依赖闭包内结合真实缓存驻留估计的获取成本选择集合。原文用 PPL 差与生成式 CE/KL。[作者仓库](https://github.com/liunian-Jay/OptiSet/blob/master/README.md#L48)目前仅说明与框架图，训练代码未发布 | `training/collect_shared_values.py`、`EvidenceHeads.loss` |
| [TableRAG（EMNLP 2025）](https://aclanthology.org/2025.emnlp-main.710/) | 借鉴作者[表结构入库](https://github.com/yxh-y/TableRAG/blob/main/offline_data_ingestion_and_query_interface/src/data_persistent.py#L90)和[SQL 执行](https://github.com/yxh-y/TableRAG/blob/main/offline_data_ingestion_and_query_interface/src/service.py#L32)思路，适配为保留原文地址的 CPU 投影、筛选、连接、聚合；语义模型绑定字段并完成必要推理 | `adapters/document_compute.py` |
| [VersionRAG](https://arxiv.org/abs/2510.08109) | 借鉴作者[版本关联](https://github.com/danielhuwiler/versionrag/blob/main/src/indexing/versionrag_indexer_graph.py#L45)和[检索前版本过滤](https://github.com/danielhuwiler/versionrag/blob/main/src/retrieval/versionrag_retriever_db.py#L180)，适配为版本、章节、字段对应及前后原文配对。查询共享头根据含有效期的目录把 as_of 请求解析为版本 ID，再限定检索范围 | `adapters/version_structure.py`、`adapters/shared_retrieval.py` |
| [Stop-RAG 作者训练源码](https://github.com/chosolbee/Stop-RAG/blob/main/src/train/stop_rag_train.py) | 核对 `_compute_tdlambda_labels`，各备份深度分别最大化，再做 full-width Q(λ) 混合；项目扩展为共享表示上的质量、完整成本和尾延迟预测；冷分支独立 cache scope，暖分支共享预建块 | `training/stop_value.py`、`controller/value_policy.py` |
| [InfoFlow KV 作者源码](https://github.com/tx467/InfoFlow-KV/tree/main/llm/models/qwen/kv_cache) | 核对 extractor、importance_scorer、recomputer：独立块、全局 RoPE、head 均值后的 query L2 norm、末两层平均与稀疏重算；在 SGLang 中接入请求私有页和同一 GPU/Host 页池 | `engine_patches/infoflow.py`、`engine_patches/infoflow_attention.py` |
| [SGLang](https://arxiv.org/abs/2312.07104)、[HiCache](https://docs.sglang.io/docs/advanced_features/hicache_design) | 原生执行、精确前缀和 GPU/Host 分层缓存 | `engine_patches/contextkv_compute.py`、SGLang 集成补丁 |

任务、字段、集合和停止头是本项目在共享表示上的训练实现。OptiSet 的效用思路落实为固定 Qwen3-32B 的真实答案结果；结构化计算保留原文地址。InfoFlow 的请求私有覆盖层、原生缓存融合和调度由本项目在 SGLang 内实现。

GPU 运行与实测结果见运行手册中的同版资产和 benchmark 入口。

## 评价依据

- [MultiDoc2Dial](https://github.com/IBM/multidoc2dial)：原始问题、文档和 F1/EM。
- [MT-RAG](https://github.com/IBM/mt-rag-benchmark)：原始任务及 MTRAGEval。
- [VersionQA](https://github.com/danielhuwiler/versionrag)：原始版本文档与作者二值正确性 rubric。
- [Loong](https://github.com/MozerWang/Loong)：作者参考评分 rubric 与 Perfect Rate。
- [LlamaIndex](https://github.com/run-llama/llama_index)：上游历史凝练与 compact/refine，生成使用相同固定 Qwen。

模型和源码使用各自发布许可证；SGLang 与 InfoFlow 的源码归属见 `engine_patches/NOTICE.md`，Loong rubric 归属见 `docs/third_party/LOONG_NOTICE.md`。
