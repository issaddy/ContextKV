# 实测结果与本地资料

当前公开 benchmark 主比较为普通文档问答、ContextKV 与 LlamaIndex，按数据集分别评分。最新机制与普通基线的 GPU 队列结果待运行。

| Benchmark | 质量指标 | 当前主比较 |
|---|---|---|
| MultiDoc2Dial | F1、EM | 待运行 |
| MT-RAG Cloud | 作者 MTRAGEval 指标及聚合 | 待运行 |
| VersionQA | 作者二值正确性 | 待运行 |
| Loong Set1 | 作者百分制、Perfect Rate、领域与难度分组 | 待运行 |

已有机制实测可以用于确定重点：

| 环节 | 实测观察 | 结果资料 |
|---|---|---|
| 证据选择 | MultiDoc2Dial 固定候选局部对照，ranked 的 RB_llm 为 84.67，support 为 77.67 | `mechanisms.evidence_selection` |
| 阅读与计算 | 显式记录投影在 20 个统计任务上，计数和求和全部正确；CPU 扫描中位数 4.64 ms | `mechanisms.record_projection` |
| 缓存感知重组 | 四个受控任务预填充减少约 39%–40%，动作至回答缩短 0.21–1.31 秒 | `mechanisms.cache_regroup` |
| READ/ANSWER 前缀 | VersionQA 两轮 ANSWER 各命中 1,156 个正文 token | `mechanisms.read_answer_prefix` |
| 双层压力下的需求保留 | 6,976 token 从重算转为 GPU 命中；完整链 13.316 → 12.610 秒，下降 5.30% | `cache_cooperation.both-tiers` |

固定输入配对用于解释缓存工作与完整链变化，队列 P95 由本次 benchmark 实际运行计算。重点观察普通语义阅读的总开销、继续/停止决策，以及容量竞争中需求保留的净收益。

表中字段均位于[最终指标汇总](../../results/container-final-metrics-20261112.json)。各 benchmark 的结果保存在同一文件的 `queues` 中。

[实验定义](task_spec.md#六实验与验收) · [运行入口](final_runbook.md)
