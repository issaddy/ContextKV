# ContextKV

**面向资源受限多轮文档问答的上下文管理与 KV 缓存协同优化研究。**

用户连续追问、撤销条件、切换版本和回访来源；每轮问答在固定模型窗口、累计计算、完整响应时间及 GPU/Host 容量内完成。项目把任务与证据、阅读计算、来源前缀保留连接成同一决策链。方法按实际效果采用，允许直接复用成熟方法。

## 当前主线

| 模块 | 决策对象 | 主要实现 |
|---|---|---|
| A：任务、证据与 reader | 当前条件和版本；必要证据集合；READ/REDUCE/ANSWER 训练目标 | `adapters/history_state.py`、`evidence_pool.py`、`support_model.py`；`controller/evidence_planner.py`；`training/` |
| B：有限阅读与收益决策 | 下一步读什么、如何归约、何时回答；完整成本约束下的计划收益 | `controller/reading.py`、`execution.py`、`claim_graph.py`、`plan_policy.py`、`plan_profiles.py`、`rolling_policy.py` |
| C：来源前缀保留 | 相对 native 的新增节省、独立 owner、共享页容量和保护期限 | `controller/prefix_view.py`、`prefix_retention.py`、`engine_patches/` |
| 评价 | 全部声明任务的官方指标、实际历史、完整成本与成对效果 | `bench/study_metrics.py`、`study_scoring.py`、`public_multihop.py` |

逻辑状态复用 `TaskState → EvidenceSet/ContextPlan → ReadResult/ReadingOutcome`。SGLang Radix/HiCache 负责物理 KV 匹配、传输与淘汰。固定执行和滚动执行共用实际生成与提交路径。

## 当前结果与本轮纠正

审查代码基线为 `702ebe9`。截至该提交，R1/R2、P1、B 收益模型和独立前缀 owner 均已有实际实现与实验；原始结果保存在 [封存报告](docs/study/final_report.md)。

| 已有实验 | 观察 |
|---|---|
| A 冻结历史四格 | 共同 13 状态上 P1R1 相对 P0R0 增加 5.38 个百分点；属于共同完成子集诊断 |
| A 自身历史 P3 | 当前 R0/R2 共同 8 状态上，P1R2 相对 P0R0 降低 6.875 个百分点 |
| B 开发选择 | B3 优于当前收益调度 S3；B3 公共批次 13 题消耗 2,242,604 在线 token |
| C 开发选择 | 选择 native；scope/TTL 模型覆盖两个开发窗口，产生 0 个正收益保留动作 |
| 公开测试 | 2Wiki 有 reader 改善信号；MultiDoc 与长文任务仍有明显质量和完成率问题 |

当前候选应以开发集重评结果选择。Dense5＋REBUILD＋native 是历史工程参照，早期实验见 [历史数据与评价](docs/data_and_evaluation.md)。

本轮修正了条件撤销监督、共享证据的多需求归属、前提删除负例、ANSWER 目标可见性、缺口关闭、有限分片归约、归约结果替代、合法回答的正文预算判定、缓存状态条件化成本，以及官方评分分母与条件化指标。完整根因、变更和下一阶段实现顺序见 [深度审查与实施路线](docs/study/review_20261103.md)。

## 运行与验证

```bash
uv sync --extra retrieval --extra evaluation --extra tokenizer --locked
uv run python -m bench.run_sessions --config configs/demo.yaml demo
uv run pytest -q
uv run ruff check controller adapters bench profiling tests scripts engine_patches
```

CPU 演示运行协议与状态。实际模型实验的模型、配置和批次入口见 [运行手册](docs/study/final_runbook.md) 与 [部署](docs/deployment.md)。训练目标、阶段成本和保留价值均绑定实际 reader、模板与原生缓存配置。

轻量机制测试可单独执行：

```bash
python -m unittest discover -s tests -p 'test_*review_semantics.py' -v
python -m unittest discover -s tests -p test_reader_visibility_review.py -v
```

HTTP 的 `/sessions/{id}/query` 接收问题，`/turns` 接收显式证据。接入见 [HTTP 指南](docs/guides/http_retrieval.md)。

## 文档入口

| 内容 | 入口 |
|---|---|
| 当前机制和服务调用链 | [架构](docs/architecture.md) |
| 当前资产与实验入口 | [课题索引](docs/study/README.md) |
| 本轮根因、代码纠正和下一阶段目标 | [深度审查](docs/study/review_20261103.md) |
| R1/R2 与 P1 的接续条件 | [Reader 与策略状态](docs/study/reader_policy_readiness.md) |
| 已运行批次和原始数值 | [封存报告](docs/study/final_report.md)、[运行手册](docs/study/final_runbook.md) |
| 参考方法 | [参考工作](docs/references.md) |

`controller/` 是运行核心，`adapters/` 提供资料与检索适配，`engine_patches/` 提供原生桥接；`bench/`、`profiling/`、`training/` 分别组织评价、成本采集与训练。`data/`、`models/`、`runs/` 和 `vendor/` 由实验环境保存。
