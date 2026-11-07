# ContextKV 课题索引

| 内容 | 入口 |
|---|---|
| 实施任务与研究边界 | [task_spec.md](task_spec.md) |
| 当前执行机制 | [架构](../architecture.md) |
| 当前运行入口 | [final_runbook.md](final_runbook.md) |


固定 Qwen3-32B，先收口统一问答主链，再冻结 A 与阅读协议。c1/c4 均比较固定／收益 B × Native／ContextPilot 编译，同时保留 LlamaIndex 外部参照和关闭跨轮事实复用的对照；各格使用声明的 350 题、77 会话及各自真实 P3 历史。

当前环境为 `test-0523`，运行与轻量记录位于 `runs/study-20261106-resume`。实际进度、资产与命令只在运行手册维护；完整质量和 P95 目标尚未验收。
