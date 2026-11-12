# ContextKV

面向资源受限多轮文档问答，联合优化证据上下文、文档计算与 KV 复用。生成模型固定为 Qwen3-32B。

| 环节 | 当前实现 | 入口 |
|---|---|---|
| 查询与会话 | Qwen3-Embedding-0.6B、RCEM 表示对齐、共享任务头、范围优先混合召回 | `adapters/shared_encoding.py`、`adapters/shared_retrieval.py` |
| 证据选择 | Jina-reranker-v3.5 共享编码、字段绑定与集合效用头 | `adapters/shared_encoding.py`、`training/shared_training.py` |
| 文档计算 | 原文单元格/跨度、投影/筛选/连接/聚合、版本成对证据 | `adapters/document_compute.py`、`adapters/version_structure.py` |
| 计算决策 | 多步继续/停止价值学习、完整总体结束条件 | `controller/value_policy.py`、`controller/cooperative.py` |
| KV 执行 | SGLang 内部精确前缀与 InfoFlow 块复用、GPU/Host 需求调度 | `engine_patches/` |
| 评测 | 普通文档问答、ContextKV、LlamaIndex；各 benchmark 独立评分 | `bench/study.py` |

执行链：共享查询编码 → 合法范围召回 → 共享证据编码与选材 → 必要结构计算 → 一次请求编译与 SGLang 执行 → 提交回答及可复用计算。

训练顺序、模型目录与运行命令见[运行手册](docs/study/final_runbook.md)。机制和复用边界见[架构](docs/architecture.md)、[方法依据](docs/references.md)，GPU 部署见[部署](docs/deployment.md)。

本轮重构已进入代码整合；辅助训练和真实 GPU 端到端评测尚未执行。[已有实测结果](docs/study/final_results.md)对应重构前版本。

```bash
python3 -m bench.study evaluate --split test --limit-turns 10 --concurrency 4
```

题数上限分别作用于每个 benchmark、每种方法。最终保留答案、来源地址、官方质量指标、完整响应、Qwen 输入/输出 token 与少量缓存指标，以及整批 GPU/Host 占用积分。
