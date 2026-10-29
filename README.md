# ContextKV

**面向资源受限、多资料、长上下文和证据动态变化的多轮问答。**

用户会连续追问、纠正条件、切换话题和回访材料；资料也会更新。ContextKV维护当前任务与适用证据，在共同材料上选择追加或重建上下文，并协调原生KV引用的生命周期。目标是提高任务完成质量，同时改善完整响应、服务能力和GPU/Host资源效率。

默认使用 **Qwen3-32B非思考模式**进行实现和真实模型测试，生成与任务解释均使用32B；BM25S、BGE承担检索与重排。课题机制不以模型规模分工作为前提，但实际效果与成本会受模型影响，机制对照固定模型和采样条件。

## 架构

| 模块 | 职责 | 主要实现 |
|---|---|---|
| A：任务与证据 | 更新任务、对象、条件和子需求，检索与选材，在同一证据集合上构造APPEND/REBUILD | `adapters/conversation.py`、`hybrid_retrieval.py`；`controller/evidence.py`、`candidates.py` |
| B：成本决策 | 结合原生缓存报价、实测阶段成本、输出长度和容量等待选择执行方案 | `controller/planner.py`、`cost_model.py`、`admission.py` |
| C：KV生命周期 | 业务epoch关联原生会话，提交新结果后退役旧引用，支持恢复 | `controller/service.py`、`session_store.py`；`engine_patches/` |

SGLang负责物理KV匹配、分配、传输与淘汰。在线入口是`controller.cli`和HTTP服务；评测从外部调用同一装配、生成与提交接口。详细调用链见[架构](docs/architecture.md)。

## 运行

CPU演示用于检查协议和状态：

```bash
uv sync --extra retrieval --extra evaluation --locked
uv run python -m bench.run_sessions --config configs/demo.yaml demo
```

按[部署说明](docs/deployment.md)启动生成和任务解释服务后，运行多轮检索接口：

```bash
CUDA_VISIBLE_DEVICES=0,1 python -m controller.cli --config configs/qwen3_32b_cost.yaml serve   --retriever-config configs/retriever_conversational.example.yaml   --evidence-corpus data/mtrag-official/cloud.jsonl --corpus-id mtrag
```

默认检索为`conversational + single + ranked`，布局为`stable_key`。`POST /sessions/{id}/query`接收问题，`/turns`接受显式证据；示例见[HTTP接口](docs/guides/http_retrieval.md)。

## 文档

| 内容 | 入口 |
|---|---|
| A/B/C调用链、状态和复用依据 | [架构](docs/architecture.md) |
| 环境、模型和服务启动 | [部署](docs/deployment.md) |
| 公开任务、比较协议、当前结果与待办 | [数据与评价](docs/data_and_evaluation.md) |
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
