# ContextKV 课题索引

| 内容 | 入口 |
|---|---|
| 实施任务与研究边界 | [task_spec.md](task_spec.md) |
| 执行机制与状态 | [架构](../architecture.md) |
| 方法来源与复用 | [参考方法](../references.md) |
| 复现入口 | [final_runbook.md](final_runbook.md) |
| 实测结果与改进重点 | [final_results.md](final_results.md) |

## 实测与验收

当前交付保留原队列 E1–E4 与后续短队列的逐格质量、延迟、资源和出处，各批结果分别汇报。当前主比较为普通 RAG、ContextKV 完整方案和 LlamaIndex；新配置的完整队列质量与 P95 目标待实测。公开评测、实际历史系统重放与构造机制任务已分开配置，按数据集报告。旧四格结果保留原登记条件，退出当前执行入口。 [实测结果](final_results.md) · [实验定义](task_spec.md#六实验与验收) · [运行与验收](final_runbook.md)
