# ContextKV

面向动态证据上下文的多轮文档与知识问答系统。在保留完整实际问答历史和来源身份的前提下，协调证据组织、APPEND/REBUILD 选择与原生 KV 生命周期，改善有限 GPU/Host 下的回答、性能和资源使用。

当前使用已有 Qwen3-8B/32B、固定 SGLang、单机部署，每个生成/解释实例各用一张 GPU。Qwen2.5-7B 已退役。训练不属于运行前置。

工程已有 CLI/HTTP、自动检索、多会话、幂等提交和恢复。新一轮主线是质量优先的统一A/B/C优化：补齐任务/条件理解、来源适用性与多资料覆盖，再降低完整执行成本。已有配置和测量作为比较基础，当前工作见[实现清单](docs/implementation_plan.md)，分版本结果见[结果对比](docs/data_and_evaluation.md)。

## 从哪里开始

| 内容 | 入口 |
|---|---|
| 场景、目标与 A/B/C 边界 | [项目范围](docs/project_scope.md) |
| 已完成内容与剩余瓶颈 | [实现清单](docs/implementation_plan.md) |
| 架构、事务与故障恢复 | [架构](docs/architecture.md) |
| 方法复用与课题增量 | [贡献说明](docs/research_contributions.md) |
| 实际部署、配置和模型 | [部署](docs/deployment.md) |
| 实验比较、指标与复现 | [数据与评价](docs/data_and_evaluation.md) |

当前主评价聚焦回答质量、实际交付延迟、时延达标服务能力和资源效率；缓存命中与动作比例用于解释机制。当前HTTP完整返回，内部引擎TTFT不作为对外首字体验。

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

容器项目位于 `/home/relay/songjiayun/ContextKV`，已有模型位于 `/media/ssd1/qwen3-8b` 和 `/media/ssd1/qwen3-32b`。按[部署说明](docs/deployment.md#多轮检索服务)启动8B生成和32B解释服务后：

```bash
export TOKENIZERS_PARALLELISM=false RAYON_NUM_THREADS=4
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
CUDA_VISIBLE_DEVICES=0,1 python -m controller.cli --config configs/qwen3_8b_cost.yaml serve \
  --retriever-config configs/retriever_conversational.example.yaml \
  --evidence-corpus data/mtrag-official/cloud.jsonl --corpus-id mtrag
```

[8B 成本配置](configs/qwen3_8b_cost.yaml)随仓库提供原测量成本文件，适用 H200、16384 GPU 槽、54272 Host 槽、page64、write_through、共同非思考采样。普通单次重建入口为 [8B](configs/qwen3_8b_scope.example.yaml) / [32B](configs/qwen3_32b_scope.example.yaml)。32B 不使用 8B 的成本表。

`POST /sessions/{id}/turns` 接收 query、request_id 和来源 evidence。配置检索器后，`POST /sessions/{id}/query` 只需问题和 request_id。两者共用完整历史、提交、重放、恢复及关闭机制；响应提供本轮可引用 ID 和实际可见来源。详见[HTTP 接口](docs/guides/http_retrieval.md)。

上面的入口用于需要任务承接和条件变化的多轮场景，保持默认稳定布局，并从原始语料恢复文档来源地址。已有正文、索引和向量继续复用；公共元数据只在启动时加载，在现有同一次重排中提供来源和标题，再进入生成上下文。旧last3配置保留用于复现已有比较；当前主线围绕任务、证据与成本的统一协同推进。

只接收外部 evidence 时可省略检索参数。自动检索也可复用 `--retriever-index data/cloud-bm25s --corpus-id mtrag`，或用 `--retriever-config` 选择已有 BM25/BGE/RRF/重排索引。可选 `--evidence-corpus` 提供版本、条件和原文关系。检索器启动时加载一次，正文与实际回答按原样保存。

多轮入口为 `--retriever-config configs/retriever_conversational.example.yaml`。示例通过独立32B `selector`更新主题、具体对象、任务与条件：从首轮建立任务与子需求，后续按字段继承，明确的字面任务补充直接组合；复合需求共享当前条件并联合检索。省略`selector`可使用已有历史回退。已实现的`source_window`可补入同源相邻原始段落，保留排名与独立引用；当前不默认扩大窗口，其适用范围见评价。所有轮次贯穿同一检索、A/R成本选择和提交链，解释和检索开销计入完整响应。默认`stable_key`布局可直接使用；已有`context.layout: contextpilot`可分开组织正文顺序与当前检索优先级。实测使用生成GPU0，32B解释与BGE检索共享GPU1，须计入两者驻留成本。机制与边界见[多轮意图解析](docs/research/source_grounded_query_resolution.md)和[真实评价](docs/data_and_evaluation.md)。

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
