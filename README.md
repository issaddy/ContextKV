# ContextKV

面向动态证据上下文的多轮文档与知识问答系统。在保留完整实际问答历史和来源身份的前提下，协调证据组织、APPEND/REBUILD 选择与原生 KV 生命周期，改善有限 GPU/Host 下的回答、性能和资源使用。

当前使用已有 Qwen3-8B/32B、固定 SGLang、单机单 GPU。Qwen2.5-7B 已退役。训练不属于运行前置。

工程已有 CLI/HTTP、自动检索、多会话、幂等提交和恢复；最新六会话三方法对照完成 120/120 轮，全部自然结束。成本策略实际选择 8 次追加、32 次重建。整体优于强原生基线的结论尚未建立，场景收益与剩余问题见[结果对比](docs/data_and_evaluation.md)。

## 从哪里开始

| 内容 | 入口 |
|---|---|
| 场景、目标与 A/B/C 边界 | [项目范围](docs/project_scope.md) |
| 已完成内容与剩余瓶颈 | [实现清单](docs/implementation_plan.md) |
| 架构、事务与故障恢复 | [架构](docs/architecture.md) |
| 方法复用与课题增量 | [贡献说明](docs/research_contributions.md) |
| 实际部署、配置和模型 | [部署](docs/deployment.md) |
| 实验比较、指标与复现 | [数据与评价](docs/data_and_evaluation.md) |

这六份文档是项目主入口。`docs/guides/` 保存接口用法，`docs/research/` 只保留与现有代码对应的方法说明。旧计划和逐轮实验纪要已从当前文档树移除，可从 Git 历史追溯；原始记录保留在 `runs/`。

## 本地运行

```bash
uv sync --extra retrieval --extra calibration --locked
uv run contextkv --config configs/demo.yaml demo
uv run pytest -q
uv run ruff check controller adapters bench profiling tests scripts engine_patches
```

CPU mock 演示用于协议和状态验证，无需模型。`configs/append.yaml`、`periodic.yaml`、`redundancy.yaml` 等是模拟策略配置。

## Qwen3 实际服务

容器项目位于 `/home/relay/songjiayun/ContextKV`，已有模型位于 `/media/ssd1/qwen3-8b` 和 `/media/ssd1/qwen3-32b`。按[部署说明](docs/deployment.md)启动固定研究引擎后：

```bash
python -m controller.cli --config configs/qwen3_8b_cost.yaml serve
```

[8B 成本配置](configs/qwen3_8b_cost.yaml)随仓库提供原测量成本文件，适用 H200、16384 GPU 槽、54272 Host 槽、page64、write_through、共同非思考采样。普通单次重建入口为 [8B](configs/qwen3_8b_scope.example.yaml) / [32B](configs/qwen3_32b_scope.example.yaml)。32B 不使用 8B 的成本表。

`POST /sessions/{id}/turns` 接收 query、request_id 和来源 evidence。配置检索器后，`POST /sessions/{id}/query` 只需问题和 request_id。两者共用完整历史、提交、重放、恢复及关闭机制；响应提供本轮可引用 ID 和实际可见来源。详见[HTTP 接口](docs/guides/http_retrieval.md)。

自动检索可复用 `--retriever-index data/cloud-bm25s --corpus-id mtrag`，或用 `--retriever-config` 选择已有 BM25/BGE/RRF/重排索引。可选 `--evidence-corpus` 提供版本、条件和原文关系。检索器启动时加载一次，正文与实际回答按原样保存。

首次准备数据与正常服务分开：

```bash
python scripts/fetch_mtrag.py
python -m adapters.prepare_data --cloud-corpus
```

P2 使用给定证据与实际生成历史，P3 加入真实检索。正式运行入口和方法匹配规则见[评价说明](docs/data_and_evaluation.md)、[原生基线](docs/guides/native_model_scope.md)、[显式矩阵](docs/guides/b6_owned_matrix.md)。已有数据、索引与标定直接复用。

## 仓库职责

| 目录 | 用途 |
|---|---|
| `controller/` | A 的证据/上下文、B 的成本选择、C 的事务/生命周期及 API/CLI |
| `adapters/` | 数据、原文、检索、重排与结构证据适配 |
| `engine_patches/` | 固定 SGLang 的观测、guard、receipt 和退役；复用原生调度/传输/淘汰 |
| `profiling/`、`bench/`、`tests/` | 标定、基线、评价与正确性验证 |
| `configs/`、`scripts/` | 可运行配置、原测量成本与部署工具 |
| `training/` | 保留的可选离线训练/导出工具，正常服务不导入 |
| `data/`、`runs/`、`vendor/` | Git 忽略的数据、原始运行记录和第三方检查树，不整体打包 |
| `.venv/`、`dist/`、构建缓存 | 本地依赖和可重新生成的产物 |

默认仅启用主线必要机制。额外来源选择器、表格增强、Host 准入和离线评分按用途显式选择；功能存在不代表应全部开启。原[技术方案](llm_qa_context_kv_technical_plan.md)保留为课题起点，当前实现以六份主文档为准。第三方代码、提示与资产的许可随[来源说明](docs/third_party/)保留。
