# 实测指标

[最终指标汇总](container-final-metrics-20261112.json) 保存 65 组队列的质量、完成率、P95、token 和资源结果，以及证据选择、记录投影、阅读重组与缓存保留的机制测量。

| 内容 | JSON 字段 |
|---|---|
| 各 benchmark、方法及并发的结果 | `queues` |
| 相同请求下 Native 与需求保留的结果 | `cache_cooperation` |
| 证据选择、确定性计算、READ/ANSWER 前缀复用 | `mechanisms` |

每组结果保留当时的任务分母、指标定义和方法名称。新评测由 `bench.capacity_workflow` 输出最终指标，使用 `profiling.final_study_index` 生成对比表。
