# ContextKV

面向多轮文档问答的研究工程：在保留完整问答历史和来源身份的前提下，协同管理证据组织、APPEND/REBUILD 决策与原生 KV Cache 生命周期，权衡回答质量、完整响应时间和 GPU/host 资源。

**当前主线使用现有 Qwen3-8B/32B 基座，训练不是前置。Qwen2.5-7B 已退役，不再启动、下载、恢复或开展相关实验。** 历史证据和仍被复用的通用算法保留；它们不能充当新模型的成本表或质量结论。

课题的统一定义见[场景、目标与 A/B/C 职责](docs/project_scope.md)：有限 GPU/Host 下、动态证据上下文的多轮文档与知识问答，协调证据组织、实际成本选择和上下文版本的缓存生命周期。企业技术文档只是应用实例。训练与其他扩展不得替代核心系统的完整流程和联合验收。

系统已有可运行的 CLI/API、P1/P2/P3、多轮提交恢复、B0/B1 与共享矩阵入口。Qwen3 已完成有限真机协议及统一入口验收，尚未建立系统整体的质量非劣或性能优胜。当前状态及下一实现优先级只看[实现清单](docs/implementation_plan.md)，复用与增量见[贡献说明](docs/research_contributions.md)。

## 仓库导航：哪些是课题主线

源码仓库与本地研发工作区共用这个目录，但不是所有目录都是系统模块。

| 层次 | 目录或文件 | 阅读与使用方式 |
|---|---|---|
| 核心实现 | `controller/`、`adapters/`、`engine_patches/` | A/B/C 控制器、数据与检索适配、SGLang 集成；模块职责见下表 |
| 研究与验证支撑 | `profiling/`、`bench/`、`tests/` | 成本建模、基线与评价、正确性验证；不需要在在线服务中全部运行 |
| 运行与构建 | `configs/`、`scripts/`、`pyproject.toml`、`uv.lock`、`Makefile`、`MANIFEST.in`、`.github/` | 配置、部署、依赖、打包和持续检查 |
| 当前说明 | `docs/` 顶层 6 份文档 | 场景、实现清单、架构、贡献、部署、评价；只有实现清单维护当前工作顺序 |
| 使用与参考 | `docs/guides/`、`docs/research/` | 按需查阅接口和方法依据；研究参考不是待执行清单 |
| 历史材料 | `docs/archive/`、原技术方案 | 历次实验、旧计划与演进记录，不作为当前任务入口 |
| 暂缓的可选支线 | `training/` | 已有训练与导出工具，保留复用价值；不是 A/B/C 主流程或验收前置 |
| 本地数据与研发材料，未纳入 Git | `data/`、`runs/`、`vendor/` | 分别存放下载数据、运行结果/临时驱动/阶段产物、第三方源码检查树；不计作自研模块，不整体分发进 Python 包 |
| 本地环境与生成物，未纳入 Git | `.venv/`、`build/`、`dist/`、`*.egg-info/`、测试/字节码缓存 | 依赖环境、构建副本、安装包及缓存；可能由安装、测试或构建自动产生 |

了解课题只需先读：

1. [场景与边界](docs/project_scope.md)：研究什么、A/B/C 各自负责什么。
2. [实现状态](docs/implementation_plan.md)：已完成、缺口与当前推进顺序。
3. [架构与事务](docs/architecture.md)：执行链、状态和恢复边界。
4. [复用与贡献](docs/research_contributions.md)：已有方法、本项目增量与效果证据的边界。
5. [部署](docs/deployment.md)与[HTTP 使用](docs/guides/http_retrieval.md)：如何运行。
6. [数据与评价](docs/data_and_evaluation.md)及[验证索引](docs/archive/verification.md)：如何解释和追溯结果。

`runs/` 混有可复现原始记录和一次性脚本，不是正式运行入口；正式入口以本 README、`scripts/` 和 Python 模块命令为准。历史记录可能被文档或分析脚本引用，整理时按依赖迁移或删除，不批量删除整个目录。`vendor/` 的多份检查树也不表示集成了多套引擎，部署使用固定版本及单一指定 checkout。

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
- [原生基线](docs/guides/native_model_scope.md) / [owned 矩阵](docs/guides/b6_owned_matrix.md)：与控制器使用同一模型、材料、输出预算和采样；只有显式声明任务才执行，没有旧固定13格生成器。

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

HTTP 服务可直接复用现有索引：在 `serve` 后加 `--retriever-index data/cloud-bm25s --corpus-id mtrag`，或用 `--retriever-config` 选择已有混合检索配置。可选 `--evidence-corpus` 补入原文版本和明确条件依赖。检索器启动时加载一次；同一请求重放原答案与来源，不再次检索或生成。仓库附有无需 GPU 的真实 BM25 多轮示例，见[HTTP 完整流程](docs/guides/http_retrieval.md)。

P2 使用给定五源和本方法真实生成历史；P3 增加真实检索、同样按已提交答案推进下一轮。例如，在研究引擎已启动、数据已准备后运行单一 P3 配置：

```bash
python -m controller.cli --config configs/qwen3_8b_scope.example.yaml \
  --state-dir runs/qwen3-p3-development \
  bench data/mtrag/RAG.jsonl --protocol P3 --domain cloud \
  --split-manifest configs/mtrag_split_manifest.json --split dev \
  --corpus data/mtrag-official/cloud.jsonl
```

也可提供已构建的 `--retriever-index`，或共享 `--retriever-config` 使用 BM25/BGE/RRF/reranker。索引和模型每次运行加载一次，完整响应计时包含检索及排队。检索失败、生成失败和未完成后续轮次保留在结果中，CLI 返回非零。见[数据与评测](docs/data_and_evaluation.md)、[混合检索](docs/guides/hybrid_retrieval.md)。这些是运行说明，不是待执行的实验清单。

## 主要模块

| 目录 | 实际职责 |
|---|---|
| `controller/` | 版本化证据、完整历史、候选构造、成本/固定策略、SQLite CAS、attempt/outbox、API/CLI |
| `adapters/` | MTRAG、Doc2Dial离线原文/监督适配、BM25/BGE 检索与重排、多轮query视图、结构证据 |
| `engine_patches/` | 固定 SGLang 的 guard/receipt、GPU/host 观测、epoch 退役；复用原生分配、传输、淘汰 |
| `bench/` | 同源 B0/B1、有限矩阵、失败分母、质量绑定、冻结选择与匹配消融 |
| `profiling/` | 真实阶段观测、离线拟合、支持域、输出长度及成对决策分析 |
| `tests/` | CPU/HTTP/源码合同；真实运行证据另行保存，不互相替代 |

默认采用可用的单次回答基座。复杂来源计划、成本决策及host准入只有明确系统净收益才进入默认路径；不为功能齐全开启全部实验机制。研究记录见[验证索引](docs/archive/verification.md)、[Qwen3统一入口验收](docs/archive/owned_local_scope_validation.md)、[候选结构审计](docs/research/qwen3_candidate_opportunity.md)。

[Doc2Dial数据适配](docs/guides/doc2dial.md)提供另一条可复用的研究入口：完整原文可直接建索引，原始多轮角色和精确grounding单独保存；下一发言实例按严格历史前缀构造，目标与标签隔离。监督标签不进入在线检索；导出成功不代表数据已成为干净训练集。条件推理与对话策略的复用选择见[机制分析](docs/research/applicability_mechanism.md)。

可选的[有源多轮训练与导出](docs/research/grounded_training.md)工具保留，当前暂缓扩展。已完成一次小规模 8B 训练和 CPU 合并，尚无训练质量改善或新 checkpoint GPU 验证结论，不作为核心模块收益。

配置和资产在边界核对；不逐请求重哈希权重/语料，不建立影子缓存调度器。共享资源、模型身份、真实 token IDs 与提交一致性校验保留。原始方案见[技术方案](llm_qa_context_kv_technical_plan.md)，它不覆盖后续事实与模型退役决定。第三方来源与许可随各资产及[来源说明](docs/third_party)保留。
