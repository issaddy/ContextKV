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
| 证据选择 | MultiDoc2Dial 固定候选局部对照，ranked 的 RB_llm 为 84.67，support 为 77.67 | `runs/study-20261106-resume/ranked-first-turn-030-001/summary.json` |
| 阅读与计算 | 显式记录投影在 20 个统计任务上，计数和求和全部正确；CPU 扫描中位数 4.64 ms | `runs/final-20261110/record-projection-summary.json` |
| 缓存感知重组 | 四个受控任务预填充减少约 39%–40%，动作至回答缩短 0.21–1.31 秒 | `runs/study-20261109-joint/gpu-summary.json` |
| READ/ANSWER 前缀 | VersionQA 两轮 ANSWER 各命中 1,156 个正文 token | `runs/study-guide-3db7fb4/read-answer-prefix-evidence.json` |
| 双层压力下的需求保留 | 6,976 token 从重算转为 GPU 命中；完整链 13.316 → 12.610 秒，下降 5.30% | `runs/cache-cooperation-20261110/both-tiers.json` |

固定输入配对用于解释缓存工作与完整链变化，队列 P95 由本次 benchmark 实际运行计算。重点观察普通语义阅读的总开销、继续/停止决策，以及容量竞争中需求保留的净收益。

本地研究资料统一索引在 [results/README.md](../../results/README.md)，包含逐数据轨结果、答案及来源、评分、动作配对、成本与缓存回执。完整实测表已集中到该索引，原始数据路径保持可追溯。

[实验定义](task_spec.md#六实验与验收) · [运行入口](final_runbook.md)
