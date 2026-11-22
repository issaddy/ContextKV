# 实测结果索引

| 数据 | 内容 |
|---|---|
| [四项固定题单与扩大多轮会话](paired_qa_20261122.json) | 主干前后、普通 RAG、LlamaIndex 的质量、完整 P95 与分项费用 |
| [上下文需求、前缀布局与物理回收](kv_residency_20261122.json) | 固定重访请求、实际并发分配、GPU/Host 页位置与重访耗时 |
| [已提交请求的前缀回收优先级](kv_admission_20261122.json) | 报价、调度准入与回执的逐阶段对照，以及独立题单上的完整耗时 |
| [停止决策的驻留配对](stop_residency_20261122.json) | 固定计划的质量漂移、Native 回载单价、120 轮完整会话和首批即答消融 |
| [指定指标行的容量感知选材](range_projection_20261122.json) | 原文投影的配对结果、独立金融题单和 KV 开销 |
| [长文标题提及与容量反馈](source_mentions_20261122.json) | 4 GiB 和 8 GiB 下的长文选材、覆盖和开销 |
| [默认配置与原版](cleanup_20261118.json) | 两个多轮队列的逐题质量和完整耗时 |
| [协同机制](crosslayer_20261118.json) | 来源驻留对选材的影响、未来需求对 Host 回载的影响 |
| [公开参考历史](current.json) | 四个 benchmark 的质量、完整 P95 和资源费用 |
| [连续会话原配置](cooperation.json) · [协同配置](cooperation_optimization.json) | 同题 P3 对照和固定证据的缓存配对 |

[实验结果与数据划分](../docs/study/results.md) · [运行手册](../docs/study/final_runbook.md)
