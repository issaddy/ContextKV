# 参考文献与工程归属

本文档分区维护**论文直接引用文献（第 1 节）**、**论文相关工作综述文献（第 2 节，未直接复用）**与**仓库内部工程源码核对及第三方归属（第 3 节）**；专项归属另见 `engine_patches/NOTICE.md`、`docs/third_party/LOONG_NOTICE.md`。

## 1. 论文参考文献

### 基础模型与推理底座

- **Qwen3-32B**：Yang et al., *Qwen3 Technical Report*, 2025 ([arXiv:2505.09388](https://arxiv.org/abs/2505.09388)) — 全项目固定生成模型与离线裁判底座。
- **Qwen3-Embedding**：Zhang et al., *Qwen3 Embedding: Advancing Text Embedding and Reranking Through Foundation Models*, 2025 ([arXiv:2506.05176](https://arxiv.org/abs/2506.05176)) — 查询与文档编码骨干（`Qwen3-Embedding-0.6B`）；文档编码左 padding 取末 token 并 L2 归一化，共享查询前向取当前问题末 token 供检索与任务头共用（`adapters/shared_encoding.py`）。
- **Jina-reranker-v3 / v3.5**：Wang et al., *jina-reranker-v3*, 2025 ([arXiv:2509.25085](https://arxiv.org/abs/2509.25085))；Nasika et al., *jina-reranker-v3.5*, 2026 ([arXiv:2607.18152](https://arxiv.org/abs/2607.18152)) — 证据列表编码与重排骨干，复用末层 marker 池化、原 projector 与跨批 query 加权 cosine（`adapters/jina_encoding.py`、`adapters/shared_encoding.py`）。
- **SGLang & HiCache**：Zheng et al., *SGLang: Efficient Execution of Structured Language Model Programs*, **NeurIPS 2024**, 37:62557–62583 ([PDF](https://proceedings.neurips.cc/paper_files/paper/2024/file/724be4472168f31ba1c9ac630f15dec8-Paper-Conference.pdf) · [DOI](https://doi.org/10.52202/079017-2000))；SGLang Team, [HiCache 设计文档](https://docs.sglang.io/docs/advanced_features/hicache_design) — 原生执行、RadixAttention 精确前缀与 GPU/Host 分层缓存（`engine_patches/contextkv_compute.py`）。

### 经典检索、子模选择与微调方法

- **BM25 / BM25S**：Robertson & Zaragoza, *The Probabilistic Relevance Framework: BM25 and Beyond*, **FnTIR**, 2009, 3(4):333–389 ([DOI](https://doi.org/10.1561/1500000019))；Lù, *BM25S*, 2024 ([arXiv:2407.03618](https://arxiv.org/abs/2407.03618)) — 范围优先稀疏召回（`adapters/retrieval.py`）。
- **RRF**：Cormack, Clarke, & Buettcher, *Reciprocal Rank Fusion Outperforms Condorcet and Individual Rank Learning Methods*, **SIGIR 2009**, pp. 758–759 ([DOI](https://doi.org/10.1145/1571941.1572114)) — 融合 BM25 与稠密检索排序（`adapters/hybrid_retrieval.py`）。
- **MMR**：Carbonell & Goldstein, *The Use of MMR, Diversity-Based Reranking for Reordering Documents and Producing Summaries*, **SIGIR 1998**, pp. 335–336 ([DOI](https://doi.org/10.1145/290941.291025)) — 借鉴 §2 相关性与新颖度分解思想构造饱和新颖累积特征（`adapters/shared_heads.py`、`adapters/shared_encoding.py`）。
- **Lin & Bilmes（ACL 2011）**：Lin & Bilmes, *A Class of Submodular Functions for Document Summarization*, **ACL-HLT 2011**, pp. 510–520 ([ACL Anthology](https://aclanthology.org/P11-1052/)) — 借鉴查询覆盖与预算贪心范式，总体投影采用查询加权同源 facility-location 目标并计入来源闭包与实际缓存代价（工程适配不套用原 1 − 1/e 近似比保证；`EvidenceEncoding.project_population`）。
- **LoRA**：Hu et al., *LoRA: Low-Rank Adaptation of Large Language Models*, **ICLR 2022** ([OpenReview](https://openreview.net/forum?id=nZeVKeeFYf9)) — 冻结嵌入骨干并对查询编码器进行低秩适配（`training/shared_training.py`）。

### 评测基准、外部基线与训练数据集

- **MultiDoc2Dial**：Feng et al., **EMNLP 2021**, pp. 6162–6176 ([ACL Anthology](https://aclanthology.org/2021.emnlp-main.498/) · [仓库](https://github.com/doc2dial/multidoc2dial) · [HF 数据集](https://huggingface.co/datasets/IBM/multidoc2dial)) — 原始问题、文档与 F1/EM。
- **MT-RAG**：Katsis et al., **TACL 2025**, 13:784–808 ([ACL Anthology](https://aclanthology.org/2025.tacl-1.36/) · [仓库](https://github.com/IBM/mt-rag-benchmark)) — 原始任务与作者 MTRAGEval。
- **VersionQA**：Huwiler, Stockinger, & Fürst, 2025 ([arXiv:2510.08109](https://arxiv.org/abs/2510.08109) · [仓库](https://github.com/danielhuwiler/versionrag)) — 原始版本文档与作者二值正确性 rubric。
- **Loong**：Wang et al., **EMNLP 2024**, pp. 5627–5646 ([ACL Anthology](https://aclanthology.org/2024.emnlp-main.322/) · [仓库](https://github.com/MozerWang/Loong)) — 随题多文档任务、作者百分制评分 rubric 与 Perfect Rate。
- **LlamaIndex**：Liu, *LlamaIndex*, 2022 ([仓库](https://github.com/run-llama/llama_index)) — 外部对照基线，使用上游历史凝练与 compact/refine，生成共用固定 Qwen3-32B（`adapters/llamaindex_dialogue.py`）。
- **InSCIt**（辅助训练）：Wu et al., **TACL 2023**, 11:453–468 ([ACL Anthology](https://aclanthology.org/2023.tacl-1.27/) · [仓库](https://github.com/ellenmellon/INSCIT/tree/2319fb85932b9528a13417223ffc6fc629ae8087)) — 仅取训练分区用于集合效用与顺序配对实验（`training/prepare_inscit.py`）。
- **WikiSQL**（辅助训练）：Zhong, Xiong, & Socher, *Seq2SQL*, 2017 ([arXiv:1709.00103](https://arxiv.org/abs/1709.00103) · [仓库](https://github.com/salesforce/WikiSQL)) — 仅取训练分区构造范围头监督配对（[docs/study/results.md](study/results.md#辅助训练数据与评测边界)）。

---

## 2. 相关工作文献（未直接复用，供论文综述与定位对比）

### RAG 范式、多轮对话检索与自适应控制

- **RAG**：Lewis et al., *Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks*, **NeurIPS 2020** ([arXiv:2005.11401](https://arxiv.org/abs/2005.11401)) — 检索增强生成奠基范式。
- **ConvDR**：Yu et al., *Few-Shot Conversational Dense Retrieval*, **SIGIR 2021** ([DOI](https://doi.org/10.1145/3404835.3462856)) — 多轮对话稠密检索经典方法，对照本文共享查询编码与范围优先召回。
- **IRCoT**：Trivedi et al., *Interleaving Retrieval with Chain-of-Thought Reasoning for Knowledge-Intensive Multi-Step Questions*, **ACL 2023** ([ACL Anthology](https://aclanthology.org/2023.acl-long.557/)) — 交替检索与推理的多跳问答范式。
- **FLARE**：Jiang et al., *Active Retrieval Augmented Generation*, **EMNLP 2023** ([ACL Anthology](https://aclanthology.org/2023.emnlp-main.495/)) — 基于生成不确定性触发主动检索。
- **Self-RAG**：Asai et al., *Self-RAG: Learning to Retrieve, Generate, and Critique through Self-Reflection*, **ICLR 2024** ([arXiv:2310.11511](https://arxiv.org/abs/2310.11511)) — 通过反思 token 自适应决定检索与评估支持度（需微调生成器，与本文固定生成模型形成对照）。
- **Adaptive-RAG**：Jeong et al., *Adaptive-RAG: Learning to Adapt Retrieval-Augmented Large Language Models through Question Complexity*, **NAACL 2024** ([ACL Anthology](https://aclanthology.org/2024.naacl-long.389/)) — 按问题复杂度路由无检索/单步/多步检索策略。

### 长上下文利用、上下文压缩与证据组织

- **Lost in the Middle**：Liu et al., *Lost in the Middle: How Language Models Use Long Contexts*, **TACL 2024** ([ACL Anthology](https://aclanthology.org/2024.tacl-1.9/)) — 揭示长上下文位置敏感性与冗余干扰，支撑本文紧凑证据选择与证据顺序实验分析。
- **RECOMP**：Xu et al., *RECOMP: Improving Retrieval-Augmented LMs with Compression and Selective Augmentation*, **ICLR 2024** ([arXiv:2310.04408](https://arxiv.org/abs/2310.04408)) — 抽取/摘要压缩及空上下文选择性增强，与本文保留原文坐标的集合选择形成对照。
- **LongLLMLingua**：Jiang et al., *LongLLMLingua: Accelerating and Enhancing LLMs in Long Context Scenarios via Prompt Compression*, **ACL 2024** ([ACL Anthology](https://aclanthology.org/2024.acl-long.91/)) — 查询感知 token 级提示压缩，对比本文块级原文选择与精确 KV 前缀复用。
- **LongBench**：Bai et al., *LongBench: A Bilingual, Multitask Benchmark for Long Context Understanding*, **ACL 2024** ([ACL Anthology](https://aclanthology.org/2024.acl-long.172/)) — 长上下文多任务评测基准，与本文所测全文档依赖型 Loong 形成互补背景。

### LLM 推理引擎与分层 KV 缓存管理

- **vLLM / PagedAttention**：Kwon et al., *Efficient Memory Management for Large Language Model Serving with PagedAttention*, **SOSP 2023** ([DOI](https://doi.org/10.1145/3600006.3613165)) — 分页 KV 内存管理奠基系统。
- **Sarathi-Serve**：Agrawal et al., *Taming Throughput-Latency Tradeoff in LLM Inference with Sarathi-Serve*, **OSDI 2024** ([arXiv:2403.02310](https://arxiv.org/abs/2403.02310)) — 分块预填充（chunked prefill）与无停顿混批调度。
- **CachedAttention**：Gao et al., *Cost-Efficient Large Language Model Serving for Multi-turn Conversations with CachedAttention*, **USENIX ATC 2024** ([arXiv:2403.19708](https://arxiv.org/abs/2403.19708)) — 面向多轮对话的 GPU/Host/SSD 分层 KV 缓存与预取淘汰调度。
- **CacheGen**：Liu et al., *CacheGen: KV Cache Compression and Streaming for Fast Large Language Model Serving*, **SIGCOMM 2024** ([DOI](https://doi.org/10.1145/3651890.3672274)) — 面向长上下文加载带宽瓶颈的 KV 压缩与流式传输。
- **Mooncake**：Qin et al., *Mooncake: A KVCache-centric Disaggregated Architecture for LLM Serving*, **USENIX FAST 2025** ([arXiv:2407.00079](https://arxiv.org/abs/2407.00079)) — 以全局 KV 缓存池为中心的 Prefill/Decode 分离式推理架构。

### 非前缀 KV 复用与 RAG–KV 协同缓存

- **Prompt Cache**：Gim et al., *Prompt Cache: Modular Attention Reuse for Low-Latency Inference*, **MLSys 2024** ([arXiv:2311.04934](https://arxiv.org/abs/2311.04934)) — 基于结构化模板的模块化注意力状态跨请求复用。
- **RAGCache**：Jin et al., *RAGCache: Efficient Knowledge Caching for Retrieval-Augmented Generation*, **ACM TOCS**, Vol. 44, No. 1 ([arXiv:2404.12457](https://arxiv.org/abs/2404.12457)) — 面向 RAG 的知识树多级缓存、替换策略与检索-生成投机流水线。
- **CacheBlend**：Yao et al., *CacheBlend: Fast Large Language Model Serving for RAG with Cached Knowledge Fusion*, **EuroSys 2025** ([DOI](https://doi.org/10.1145/3689031.3696098) · [arXiv:2405.16444](https://arxiv.org/abs/2405.16444)) — 非前缀检索块 KV 拼接与选择性稀疏重算代表工作，对比本文带实时报价门控与上下文选材联动的双向协同。
- **EPIC**：Hu et al., *EPIC: Efficient Position-Independent Caching for Serving Large Language Models*, **ICML 2025**, PMLR 267:24391–24402 ([arXiv:2410.15332](https://arxiv.org/abs/2410.15332)) — 位置无关分块 KV 缓存与少量 token 重算机制。

### 表格与异构文档推理

- **Chain-of-Table**：Wang et al., *Chain-of-Table: Evolving Tables in the Reasoning Chain for Table Understanding*, **ICLR 2024** ([arXiv:2401.04398](https://arxiv.org/abs/2401.04398)) — 表格多步演化推理链方法。
- **TableRAG（NeurIPS 2024）**：Chen et al., *TableRAG: Million-Token Table Understanding with Language Models*, **NeurIPS 2024** ([arXiv:2410.04739](https://arxiv.org/abs/2410.04739)) — 面向百万级大表的 schema/cell 检索增强框架（与 EMNLP 2025 同名异构文档工作区分）。

---

## 3. 工程源码核对与第三方归属

> 本节仅用于仓库内部工程实现溯源与开源归属，所有外部源码链接均锁定至已验证 commit SHA。

| 上游源码 / 锚点 | 工程核对要点 | 项目实现 |
|---|---|---|
| [Qwen3-Embedding `44548aa` #L82](https://github.com/QwenLM/Qwen3-Embedding/blob/44548aa5f0a0aed1c76d64e19afe47727a325b8f/README.md#L82) | `last_token_pool` 与 L2 归一化 | `SharedQueryEncoder.encode_many`、`encode_documents_with_usage` |
| [RCEM §3，公式 4–7](https://arxiv.org/html/2606.01697v2#S3) | 会话→改写点 L2 对齐、普通查询自对齐与成对距离 SL（λ=1）；文档关闭适配器保持冻结空间 | `training/shared_training.py:task_loss` |
| [Jina-reranker-v3.5 `e8a93f3` #L81](https://huggingface.co/jinaai/jina-reranker-v3.5/blob/e8a93f33f0b22108f8c2364f8484ce3422552fbc/modeling.py#L81) · [`rerank` #L237](https://huggingface.co/jinaai/jina-reranker-v3.5/blob/e8a93f33f0b22108f8c2364f8484ce3422552fbc/modeling.py#L237) | 末层 marker 池化、原 projector 与跨批 query 加权 cosine | `adapters/jina_encoding.py`、`SharedEvidenceEncoder.encode`、`EvidenceEncoding.merge` |
| [OptiSet §3.2–3.3](https://arxiv.org/html/2601.05027v1#S3.SS2) | 相对无证据基线的生成器集合效用 ΔQ = Q(S) − Q(∅)，MSE + 集合列表 KL 校准 7 参数集合头 | `training/collect_shared_values.py`、`EvidenceHeads.gain_loss` |
| [TableRAG（EMNLP 2025）](https://aclanthology.org/2025.emnlp-main.710/) · [`data_persistent.py` #L90](https://github.com/yxh-y/TableRAG/blob/f8d798fc60fb44736d7285ca3ca1dbcb7b3e984d/offline_data_ingestion_and_query_interface/src/data_persistent.py#L90) · [`service.py` #L32](https://github.com/yxh-y/TableRAG/blob/f8d798fc60fb44736d7285ca3ca1dbcb7b3e984d/offline_data_ingestion_and_query_interface/src/service.py#L32) | 表结构抽取与执行思路，适配为保留原文坐标的 CPU 投影、筛选、连接与聚合 | `adapters/document_compute.py` |
| [VersionRAG `2a2cbe8`](https://arxiv.org/abs/2510.08109) · [`indexer_graph.py` #L45](https://github.com/danielhuwiler/versionrag/blob/2a2cbe8f285557f99dc7a79d6df7134e1e3eccff/src/indexing/versionrag_indexer_graph.py#L45) · [`retriever_db.py` #L180](https://github.com/danielhuwiler/versionrag/blob/2a2cbe8f285557f99dc7a79d6df7134e1e3eccff/src/retrieval/versionrag_retriever_db.py#L180) | 版本关联与检索前版本过滤思路（同 `bench/versionqa.py` 锁定 commit），适配为版本/章节/字段对应与目录解析 | `adapters/version_structure.py`、`adapters/shared_retrieval.py` |
| [Stop-RAG `fcbcb78`](https://github.com/chosolbee/Stop-RAG/blob/fcbcb781fd95d56b1fde72ada15357a6a6f03941/src/train/stop_rag_train.py) · [arXiv:2510.14337](https://arxiv.org/abs/2510.14337) | 多步回报 λ 加权；项目按当前策略贪心动作备份，同计划共享质量标签并保留各自实测成本 | `training/stop_value.py`、`controller/value_policy.py` |
| [InfoFlow KV `725b75c`](https://github.com/tx467/InfoFlow-KV/tree/725b75cc44870513c79fa7450944df2fd62f37e5/llm/models/qwen/kv_cache) · [arXiv:2603.05353](https://arxiv.org/abs/2603.05353) | 核对独立块、全局 RoPE、head 均值后 query L2 norm、末两层平均与稀疏重算；项目在 SGLang 中实现私有页、混批解码与共享 GPU/Host 页池融合 | `engine_patches/infoflow.py`、`engine_patches/infoflow_attention.py` |
| [SGLang `297e54f`](https://github.com/sgl-project/sglang/tree/297e54fd105e974cdf401e5b4d20253a22c16f90) | 补丁基线（SGLang 0.5.23）、精确前缀与 GPU/Host 分层缓存 | `engine_patches/contextkv_compute.py`、SGLang 集成补丁 |

模型和源码使用各自发布许可证；引擎与评测归属详见 [engine_patches/NOTICE.md](../engine_patches/NOTICE.md) 与 [docs/third_party/LOONG_NOTICE.md](third_party/LOONG_NOTICE.md)。
