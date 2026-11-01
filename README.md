# ContextKV

**面向资源受限、多资料、长上下文和证据动态变化的多轮问答。**

用户会连续追问、纠正条件、切换话题和回访材料；资料也会更新。ContextKV维护实际会话与适用证据，在共同材料上选择追加或重建上下文，并协调原生KV引用的生命周期。目标是提高任务完成质量，同时改善完整响应、服务能力和GPU/Host资源效率。方案选型以实测效果和成本为准，可直接复用任何有效的成熟方法，创新性不作为采用条件。

默认使用 **Qwen3-32B非思考模式**进行生成与可选任务解释，BM25S、BGE承担检索与重排；机制对照固定模型和采样条件，比较质量与完整成本。

## 架构

| 模块 | 职责 | 主要实现 |
|---|---|---|
| A：任务与证据 | 用当前问题与实际QA检索、选择原文；可选结构化任务解释 | `adapters/conversation.py`、`hybrid_retrieval.py`；`controller/evidence.py`、`candidates.py` |
| B：上下文选择 | 默认按当前原文与实际QA重建；可选原生报价与阶段成本选择 | `controller/planner.py`、`cost_model.py`、`admission.py` |
| C：KV生命周期 | 业务epoch关联原生会话，提交新结果后退役旧引用，支持恢复 | `controller/service.py`、`session_store.py`；`engine_patches/` |

SGLang负责物理KV匹配、分配、传输与淘汰。在线入口是`controller.cli`和HTTP服务；评测从外部调用同一装配、生成与提交接口。详细调用链见[架构](docs/architecture.md)。

## 运行

CPU演示用于检查协议和状态：

```bash
uv sync --extra retrieval --extra evaluation --locked
uv run python -m bench.run_sessions --config configs/demo.yaml demo
```

按[部署说明](docs/deployment.md)启动一个原生NGRAM生成服务后，运行多轮检索接口：

```bash
CUDA_VISIBLE_DEVICES=0 python -m controller.cli serve \
  --retriever-config configs/retriever_dialogue.example.yaml \
  --evidence-corpus data/mtrag-official/cloud.jsonl --corpus-id mtrag
```

CLI默认配置为`configs/qwen3_32b.yaml`，最终推荐检索为`dialogue + dense + ranked`（Dense5），使用已评测的BGE查询适配器；每轮重建当前证据与实际QA，原生Radix/HiCache继续复用共同前缀。`POST /sessions/{id}/query`接收问题，`/turns`接受显式证据；示例见[HTTP接口](docs/guides/http_retrieval.md)。

## 最终方案与实测结果

最终固定复验完成454份回答及同模型评分，全部评分的完整问题、实际历史、来源和裁判输出核查通过。按质量优先目标，选择 **Dense5＋REBUILD＋原生Radix/HiCache**：

| 任务 | RB_llm：混合→稠密 | 平均完整响应秒 | GPU / Host积分：稠密 |
|---|---:|---:|---:|
| MultiDoc2Dial既定206轮 | 0.815→0.838 | 4.57→4.60 | 90.07 / 27.95 GiB秒/轮 |
| Cloud既定21轮 | 0.743→0.733 | 6.32→6.60 | 187.07 / 59.34 GiB秒/轮 |

MultiDoc跨度完整覆盖66.5%→73.8%，与开发集方向一致；Cloud小样本略降，质量分差区间均跨零。Dense5是本次任务和预算下质量优先的工程选择，各语料的差异分别保留。前一阶段482份整体方案对照已验证，实际历史检索＋重建相对任务解释＋成本选择能改善完整响应与GPU/Host累计成本；具体配置及资源口径见[数据与评价](docs/data_and_evaluation.md)。

Loong v2完成122题完整原文阅读及评分，无截断；106题输入超过16K，平均参考评分64.36/100、完整响应3.81秒。原始323题中另201题超过当前完整预算。版本迁移、历史来源回访、并发与停顿回放的实测范围及结论也统一保存；自然历史来源恢复使用可选任务解释配置。

## 文档

| 内容 | 入口 |
|---|---|
| A/B/C调用链、状态和复用依据 | [架构](docs/architecture.md) |
| 环境、模型和服务启动 | [部署](docs/deployment.md) |
| 公开任务、比较协议与实测结果 | [数据与评价](docs/data_and_evaluation.md) |
| 相关论文与实现 | [参考工作](docs/references.md) |
| 使用细节 | [检索](docs/guides/hybrid_retrieval.md)、[成本标定](docs/guides/stage_cost_model.md)、[HTTP](docs/guides/http_retrieval.md) |
| 数据接入 | [MultiDoc2Dial](docs/guides/doc2dial.md)、[Loong](docs/guides/loong.md)、[SDK证据](docs/guides/api_evidence.md)、[表格关系](docs/research/table_relations.md) |

## 开发工具

| 目录 | 用途 |
|---|---|
| `bench/` | 公开数据评测、负载运行、结果分析、CPU演示与模拟后端 |
| `profiling/` | 成本采集、拟合和引擎探针 |
| `scripts/` | 数据下载和环境准备 |
| `tests/` | 模块行为与接口测试 |

这些工具使用核心接口，不属于A/B/C机制，也不进入真实服务请求链路。成本标定交付数据文件供B加载。


改动围绕当前需求、条件和来源适用范围、证据充分性及回答落实展开。持续出现的瓶颈需要比较相关论文与实现，再决定机制调整；依据实际回答和完整成本保留方案。测试覆盖模块行为、接口和状态一致性，不固定实现细节。废弃实现及其专属测试一起删除，历史实验通过Git和压缩原始记录追溯。

```bash
uv run pytest -q
uv run ruff check controller adapters bench profiling tests scripts engine_patches
```

`controller/`是运行核心，`adapters/`提供资料与检索适配，`engine_patches/`只保留原生桥接实现、补丁及版本清单。`data/`、`models/`、`runs/`、`vendor/`和构建产物不入Git；复制资产的许可保留在[第三方说明](docs/third_party/)。
