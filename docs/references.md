# 当前方法的复用依据

| 相关工作 | ContextKV 采用的机制 | 项目实现 |
|---|---|---|
| [Qwen3 Embedding](https://arxiv.org/abs/2506.05176)、[RCEM](https://arxiv.org/abs/2606.01697) | 会话表示对齐到独立查询空间；同次编码支撑任务结构和 Need 检索 | `adapters/shared_encoding.py`、`training/shared_training.py` |
| [Jina-reranker-v3.5](https://huggingface.co/jinaai/jina-reranker-v3.5) | 列表证据共享编码；段落表示用于选材，token 表示用于字段及来源绑定 | `adapters/shared_encoding.py`、`training/shared_heads.py` |
| [OptiSet](https://arxiv.org/abs/2601.05027) | 生成器效用监督集合选择，学习互补、重复和干扰材料 | `training/collect_shared_values.py`、`EvidenceHeads` |
| [TableRAG](https://aclanthology.org/2025.emnlp-main.710/) | 结构化字段定位与确定性执行结合语义推理 | `adapters/document_compute.py` |
| [VersionRAG](https://arxiv.org/abs/2510.08109) | 版本、章节和字段对应关系，版本过滤与前后证据配对 | `adapters/version_structure.py` |
| [Stop-RAG](https://arxiv.org/abs/2510.14337) | 完整轨迹的多步继续/停止价值学习 | `training/stop_value.py`、`controller/value_policy.py` |
| [InfoFlow KV](https://arxiv.org/html/2603.05353v3) | 独立块 KV、全局 RoPE、查询注意力驱动的选择性重算 | `engine_patches/infoflow.py`、`engine_patches/infoflow_attention.py` |
| [SGLang](https://arxiv.org/abs/2312.07104)、[HiCache](https://docs.sglang.io/docs/advanced_features/hicache_design) | 原生执行、精确前缀和 GPU/Host 分层缓存 | `engine_patches/contextkv_compute.py`、SGLang 集成补丁 |

任务、字段、集合和停止头是本项目在共享表示上的训练实现。OptiSet 的效用思路落实为固定 Qwen3-32B 的真实答案结果；结构化计算保留原文地址。InfoFlow 的请求私有覆盖层、原生缓存融合和调度由本项目在 SGLang 内实现。

论文报告的收益不作为 ContextKV 的实测结果。当前代码的 GPU 实测由运行手册中的同版资产与 benchmark 入口产生。

## 评价依据

- [MultiDoc2Dial](https://github.com/IBM/multidoc2dial)：原始问题、文档和 F1/EM。
- [MT-RAG](https://github.com/IBM/mt-rag-benchmark)：原始任务及 MTRAGEval。
- [VersionQA](https://github.com/danielhuwiler/versionrag)：原始版本文档与作者二值正确性 rubric。
- [Loong](https://github.com/MozerWang/Loong)：作者参考评分 rubric 与 Perfect Rate。
- [LlamaIndex](https://github.com/run-llama/llama_index)：上游历史凝练与 compact/refine，生成使用相同固定 Qwen。

模型和源码使用各自发布许可证；SGLang 与 InfoFlow 的源码归属见 `engine_patches/NOTICE.md`，Loong rubric 归属见 `docs/third_party/LOONG_NOTICE.md`。
