# 实测指标

[重构前报告](../docs/study/final_results.md)按四个 benchmark 分别汇报普通问答、ContextKV、LlamaIndex 比较，以及实际多轮历史、辅助训练、物理资源和缓存配对。共享编码与 InfoFlow 重构后的运行状态见[运行手册](../docs/study/final_runbook.md)。

| 文件 | 内容 |
|---|---|
| [本轮指标](evaluation-20261112.json) | `queue_results`：39组完成运行，含交付失败；`incomplete_attempts`：12组中断运行的已付费用；`auxiliary`：训练结果；`cache_pairs`：4个缓存配对；`group_continuations`：已有事实续跑 |
| [已有最终指标](container-final-metrics-20261112.json) | 65组此前的最终指标及证据选择、记录投影、阅读重组、READ/ANSWER 前缀的机制测量，保留各自协议 |

各 benchmark 分别比较；作者历史与方法自身历史分别报告。本轮完整队列分数包含未交付的零质量，已交付但未评分的项目保留缺失区间。中断运行不参与主比较。原始上下文、候选与评分日志在汇总后清理，模型保留在容器。
