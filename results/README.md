# 实测指标

[重构前报告](../docs/study/final_results.md)按四个 benchmark 分别汇报普通问答、ContextKV、LlamaIndex 比较，以及实际多轮历史、辅助训练、物理资源和缓存配对。共享编码与 InfoFlow 重构后的运行状态见[运行手册](../docs/study/final_runbook.md)。

| 文件 | 内容 |
|---|---|
| [语义事实与动态决策](semantic-state-20261113.json) | 当前四项各 20 题结果、同题普通问答/LlamaIndex 参照、两题补跑及全部已付费用、辅助训练摘要；[当前报告](../docs/study/shared_smoke.md) |
| [动态流程迭代](dynamic-flow-20261113.json) | 此前四项各 20 题的重排试验、恢复任务聚焦输入后的 20 题复测、通用投影协议与关系计算检查 |
| [核心机制 20 题评测](core-mechanism-20-20261113.json) | 此前四项 benchmark 各 20 题的普通问答、ContextKV、LlamaIndex 对照，包含质量、完整响应、批次资源、辅助训练与缓存协同 |
| [此前源码审查小跑](method-review-20261113.json) | 此前各 2 题的指标及 Native／InfoFlow 非前缀配对 |
| [共享编码首次小跑](shared-encoding-smoke-20261113.json) | 审查前四项 benchmark 同题比较、真实历史 2 轮、实际批次资源及辅助训练摘要 |
| [本轮指标](evaluation-20261112.json) | `queue_results`：39组完成运行，含交付失败；`incomplete_attempts`：12组中断运行的已付费用；`auxiliary`：训练结果；`cache_pairs`：4个缓存配对；`group_continuations`：已有事实续跑 |
| [已有最终指标](container-final-metrics-20261112.json) | 65组此前的最终指标及证据选择、记录投影、阅读重组、READ/ANSWER 前缀的机制测量，保留各自协议 |

各 benchmark 分别比较；作者历史与方法自身历史分别报告。本轮完整队列分数包含未交付的零质量，已交付但未评分的项目保留缺失区间。中断运行不参与主比较。原始上下文、候选与评分日志在汇总后清理，模型保留在容器。
