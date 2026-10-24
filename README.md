# ContextKV

面向多轮文档问答的研究工程：在保留完整问答历史和来源身份的前提下，协同管理证据组织、APPEND/REBUILD 决策与原生 KV Cache 生命周期，权衡回答质量、完整响应时间和 GPU/host 资源。

**当前生成模型仅使用显式本地 Qwen3-8B/32B scope。Qwen2.5-7B 已退役，不再启动、下载、恢复或开展相关实验。** 历史证据和仍被复用的通用算法保留；它们不能充当新模型的成本表或质量结论。

系统已有可运行的 CLI/API、P1/P2/P3、多轮提交恢复、B0/B1 与共享矩阵入口。Qwen3 已完成有限真机协议及统一入口验收，尚未建立系统整体的质量非劣或性能优胜。当前状态及下一实现优先级见[模块盘点](docs/implementation_plan.md)，复用依据和研究边界见[研发策略](docs/research_strategy.md)。

## 本地开发与离线演示

```bash
uv sync --extra retrieval --extra calibration --locked
uv run contextkv --config configs/demo.yaml demo
uv run pytest -q
uv run ruff check controller adapters bench profiling tests scripts engine_patches
```

CPU mock 演示无需模型，结果用于协议正确性，不代表回答质量或 GPU 性能。`configs/append.yaml`、`periodic.yaml`、`redundancy.yaml`、`cost.yaml` 等保留为模型无关的模拟策略配置。

## Qwen3 运行入口

在用户容器项目目录 `/home/relay/songjiayun/ContextKV` 使用已有模型 `/media/ssd1/qwen3-8b` 或 `/media/ssd1/qwen3-32b`。对应共享配置为：

- [8B](configs/qwen3_8b_scope.example.yaml) / [32B](configs/qwen3_32b_scope.example.yaml)：显式 scope、共同采样、完整上下文和单次 REBUILD；不加载旧成本表。
- [固定引擎部署](docs/deployment.md)：安装独立 SGLang 环境、核对已有本地文件、生成 fingerprint、启动服务。
- [原生基线](docs/native_model_scope.md) / [owned 矩阵](docs/b6_owned_matrix.md)：与控制器使用同一模型、材料、输出预算和采样；只有显式声明任务才执行，没有旧固定13格生成器。

启动研究引擎后，API 使用同一配置：

```bash
python -m controller.cli --config configs/qwen3_8b_scope.example.yaml serve
```

`POST /sessions/{id}/turns` 接收 query、request_id 和来源 evidence；配置检索器后，`POST /sessions/{id}/query` 只需当前问题和 request_id，服务读取已提交完整历史并自动检索。两条入口共用提交、幂等、恢复和关闭机制。退役 ACK 不代表物理缓存立即释放。见[引擎协议](engine_patches/README.md)。

## 完整检索与多轮流程

数据准备与运行分开，不在服务启动时自动下载数据或模型。已有数据可直接复用；首次准备采用固定来源：

```bash
python scripts/fetch_mtrag.py
python -m adapters.prepare_data --cloud-corpus
```

HTTP 服务可直接复用现有索引：在 `serve` 后加 `--retriever-index data/cloud-bm25s --corpus-id mtrag`，或用 `--retriever-config` 选择已有混合检索配置。可选 `--evidence-corpus` 补入原文版本和明确条件依赖。检索器启动时加载一次；同一请求重放原答案与来源，不再次检索或生成。仓库附有无需 GPU 的真实 BM25 多轮示例，见[HTTP 完整流程](docs/http_retrieval.md)。

P2 使用给定五源和本方法真实生成历史；P3 增加真实检索、同样按已提交答案推进下一轮。例如，在研究引擎已启动、数据已准备后运行单一 P3 配置：

```bash
python -m controller.cli --config configs/qwen3_8b_scope.example.yaml \
  --state-dir runs/qwen3-p3-development \
  bench data/mtrag/RAG.jsonl --protocol P3 --domain cloud \
  --split-manifest configs/mtrag_split_manifest.json --split dev \
  --corpus data/mtrag-official/cloud.jsonl
```

也可提供已构建的 `--retriever-index`，或共享 `--retriever-config` 使用 BM25/BGE/RRF/reranker。索引和模型每次运行加载一次，完整响应计时包含检索及排队。检索失败、生成失败和未完成后续轮次保留在结果中，CLI 返回非零。见[数据与评测](docs/data_and_evaluation.md)、[混合检索](docs/hybrid_retrieval.md)。这些是运行说明，不是待执行的实验清单。

## 主要模块

| 目录 | 实际职责 |
|---|---|
| `controller/` | 版本化证据、完整历史、候选构造、成本/固定策略、SQLite CAS、attempt/outbox、API/CLI |
| `adapters/` | MTRAG、原始语料、BM25/BGE 检索与重排、多轮 query 视图、结构证据 |
| `engine_patches/` | 固定 SGLang 的 guard/receipt、GPU/host 观测、epoch 退役；复用原生分配、传输、淘汰 |
| `bench/` | 同源 B0/B1、有限矩阵、失败分母、质量绑定、冻结选择与匹配消融 |
| `profiling/` | 真实阶段观测、离线拟合、支持域、输出长度及成对决策分析 |
| `tests/` | CPU/HTTP/源码合同；真实运行证据另行保存，不互相替代 |

默认采用可用的单次回答基座。复杂来源计划、成本决策及host准入只有明确系统净收益才进入默认路径；不为功能齐全开启全部实验机制。研究记录见[验证索引](docs/verification.md)、[Qwen3统一入口验收](docs/owned_local_scope_validation.md)、[候选结构审计](docs/qwen3_candidate_opportunity.md)。

配置和资产在边界核对；不逐请求重哈希权重/语料，不建立影子缓存调度器。共享资源、模型身份、真实 token IDs 与提交一致性校验保留。原始方案见[技术方案](llm_qa_context_kv_technical_plan.md)，它不覆盖后续事实与模型退役决定。第三方来源与许可随各资产及[来源说明](docs/third_party)保留。
