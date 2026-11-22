# 实测结果索引

| 数据 | 内容 |
|---|---|
| [四项固定题单与扩大多轮会话](paired_qa_20261122.json) | 主干前后、普通 RAG、LlamaIndex 的质量、完整 P95 与分项费用 |
| [指定指标行的容量感知选材](range_projection_20261122.json) | 原文投影的配对结果、独立金融题单和 KV 开销 |
| [长文标题提及与容量反馈](source_mentions_20261122.json) | 4 GiB 和 8 GiB 下的长文选材、覆盖和开销 |
| [默认配置与原版](cleanup_20261118.json) | 两个多轮队列的逐题质量和完整耗时 |
| [协同机制](crosslayer_20261118.json) | 来源驻留对选材的影响、未来需求对 Host 回载的影响 |
| [公开参考历史](current.json) | 四个 benchmark 的质量、完整 P95 和资源费用 |
| [连续会话原配置](cooperation.json) · [协同配置](cooperation_optimization.json) | 同题 P3 对照和固定证据的缓存配对 |

[实验结果与数据划分](../docs/study/results.md) · [运行手册](../docs/study/final_runbook.md)
