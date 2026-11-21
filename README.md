# ContextKV

面向资源受限多轮文档问答，联合优化证据上下文、文档计算与 KV 复用。生成模型固定为 Qwen3-32B。

| 环节 | 当前实现 | 入口 |
|---|---|---|
| 查询与会话 | Qwen3-Embedding-0.6B、RCEM 表示对齐、共享任务头、范围优先混合召回 | `adapters/shared_encoding.py`、`adapters/shared_retrieval.py` |
| 证据选择 | Jina-reranker-v3.5 共享编码、软来源支持与语义优先选材 | `adapters/shared_encoding.py`、`adapters/shared_heads.py` |
| 文档计算 | 原文单元格/跨度、投影/筛选/连接/聚合、版本成对证据 | `adapters/document_compute.py`、`adapters/version_structure.py` |
| 执行选择 | 先复用结果；纯表绑定计算，其余按完整原文或必要读取执行 | `controller/cooperative.py` |
| KV 执行 | SGLang Native 精确前缀、真实容量与后续消费者需求 | `engine_patches/` |
| 评测 | 普通文档问答、ContextKV、LlamaIndex；各 benchmark 独立评分 | `bench/study.py` |

执行链：解析问题与范围 → 复用适用结果 → 必要证据与计算 → 编译来源与后续需求 → Native 执行 → 提交回答。纯表先绑定计算，其余固定总体全文可直接回答时跳过证据编码。

训练顺序、模型目录与运行命令见[运行手册](docs/study/final_runbook.md)。机制和复用边界见[架构](docs/architecture.md)、[方法依据](docs/references.md)，GPU 部署见[部署](docs/deployment.md)。

默认使用现有查询、字段与集合资产，训练保持暂停。InfoFlow 保留为显式实验路径。[已有实验结果](docs/study/results.md)对应各自代码和配置，本版清理尚未验证端到端质量或 P95 收益。

```bash
python3 -m bench.study evaluate --split test --limit-turns 20 --concurrency 4
```

题数上限分别作用于每个 benchmark、每种方法。最终保留答案、来源地址、官方质量指标、完整响应、Qwen 输入/输出 token 与少量缓存指标，以及整批 GPU/Host 占用积分。
