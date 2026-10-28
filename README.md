# ContextKV

**面向资源受限多轮文档问答的上下文管理与KV缓存协同优化研究。**

面向多资料、长上下文、证据随交互变化且GPU/Host资源受限的多轮问答，协调任务与证据组织（A）、完整成本决策（B）和原生KV生命周期（C），优先提高回答质量，再改善性能与资源效率。工程基于Qwen3与SGLang，提供CLI/HTTP、自动检索、多会话、幂等提交和恢复。研究边界和纠偏依据见[项目范围](docs/project_scope.md)。

## 文档导航

| 阅读目的 | 文档职责与入口 |
|---|---|
| 明确课题目标 | [项目范围](docs/project_scope.md)：场景、需求、模块职责和评价指标 |
| 查看交付与待办 | [实现状态](docs/implementation_plan.md)：本地/容器版本、验证、产物和未完成项 |
| 理解整体工程 | [架构](docs/architecture.md)：跨模块接口、状态、事务与恢复 |
| 理解课题工作 | [机制与研究依据](docs/research_contributions.md)：复用关系和适配增量 |
| 启动真实服务 | [部署](docs/deployment.md)：环境、进程、设备、命令与配置迁移 |
| 查看实验 | [数据与评价](docs/data_and_evaluation.md)：benchmark、比较协议和分版本结果 |
| 查找参考工作 | [参考文献清单](docs/references.md)：论文、方法和项目的简述与链接 |

机制细节与使用步骤分别维护在`docs/research/`和`docs/guides/`：

| 主题 | 设计与使用入口 |
|---|---|
| 多轮检索 | [任务与证据机制](docs/research/source_grounded_query_resolution.md)、[混合检索配置](docs/guides/hybrid_retrieval.md) |
| 数据与结构材料 | [Doc2Dial/MultiDoc2Dial](docs/guides/doc2dial.md)、[SDK证据包](docs/guides/api_evidence.md)、[表格关系](docs/research/table_relations.md) |
| 成本与决策 | [联合模型](docs/research/joint_optimization_model.md)、[等待估计](docs/research/admission_wait_model.md)、[阶段成本](docs/guides/stage_cost_model.md)、[成对验证](docs/guides/decision_pairs.md)、[冗余基线](docs/guides/redundancy_baseline.md) |
| 缓存与服务 | [Host准入](docs/research/host_admission_design.md)、[SGLang补丁](engine_patches/README.md)、[HTTP接口](docs/guides/http_retrieval.md) |
| 模型与对照 | [模型身份与采样](docs/guides/model_scope.md)、[原生基线](docs/guides/native_model_scope.md)、[服务器矩阵](docs/guides/b6_owned_matrix.md) |
| 离线工具 | [质量比较](docs/guides/quality_acceptance.md)、[训练与导出](docs/research/grounded_training.md) |

## 本地运行

```bash
uv sync --extra retrieval --extra calibration --extra evaluation --locked
uv run contextkv --config configs/demo.yaml demo
uv run pytest -q
uv run ruff check controller adapters bench profiling tests scripts engine_patches
```

CPU mock演示用于协议和状态验证。首次准备MT-RAG数据可执行`python scripts/fetch_mtrag.py`和`python -m adapters.prepare_data --cloud-corpus`。

## Qwen3服务入口

容器项目位于`/home/relay/songjiayun/ContextKV`，已有模型位于`/media/ssd1/qwen3-8b`和`/media/ssd1/qwen3-32b`。按[部署说明](docs/deployment.md#多轮检索服务)启动8B生成与32B解释服务后：

```bash
CUDA_VISIBLE_DEVICES=0,1 python -m controller.cli --config configs/qwen3_8b_cost.yaml serve \
  --retriever-config configs/retriever_conversational.example.yaml \
  --evidence-corpus data/mtrag-official/cloud.jsonl --corpus-id mtrag
```

配置检索器后，`POST /sessions/{id}/query`接收问题和request ID；`POST /sessions/{id}/turns`用于直接提供证据。请求示例及恢复操作见[HTTP指南](docs/guides/http_retrieval.md)。

## 仓库职责

| 目录 | 用途 |
|---|---|
| `controller/` | A/B/C核心、事务、API与CLI |
| `adapters/` | 数据、检索、重排及原文结构适配 |
| `engine_patches/` | 固定SGLang的观测、代际关联、完成回执与引用退役 |
| `profiling/`、`bench/`、`tests/` | 标定、基线、评价与正确性检查 |
| `configs/`、`scripts/` | 配置、成本数据及部署工具 |
| `training/` | 可选离线训练与导出工具 |
| `docs/` | 项目文档和[第三方许可说明](docs/third_party/) |
| `data/`、`runs/`、`vendor/` | Git忽略的数据、原始运行记录和第三方检查树 |
| `.venv/`、`dist/` | 本地依赖和构建产物 |
