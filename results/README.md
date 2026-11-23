# 实测结果索引

| 数据 | 内容 |
|---|---|
| [全量题单对照](full_qa_20261123.json) | 四套最终配置的质量、完整 P95 与费用，三套同期四方对照及 VersionQA 分类得分 |
| [MultiDoc2Dial 历史 P3 基线](full_multidoc_p3_20261116.json) | 普通 RAG、LlamaIndex 与早期 ContextKV 的 4,094 轮实际历史对照 |
| [四项固定题单与扩大多轮会话](paired_qa_20261122.json) | 主干前后、普通 RAG、LlamaIndex 的质量、完整 P95 与分项费用 |
| [上下文需求、前缀布局与物理回收](kv_residency_20261122.json) | 固定重访请求、实际并发分配、GPU/Host 页位置与重访耗时 |
| [已提交请求的前缀回收优先级](kv_admission_20261122.json) | 报价、调度准入与回执的逐阶段对照，以及独立题单上的完整耗时 |
| [Native 与 InfoFlow 混合解码](mixed_decode_20261122.json) | 等长输出单步时间、自然答案质量、真实路由及计入预热的 P95 |
| [从冷池开始的非前缀重访](kv_cold_start_20261123.json) | 首访及重访的实际路线、独立块准备费用、真实会话中超出 Native 前缀的来源需求 |
| [停止决策的驻留配对](stop_residency_20261122.json) | 训练首批与线上证据的对齐、固定计划的动作质量与费用、Native 回载和 120 轮停止消融 |
| [指定指标行的容量感知选材](range_projection_20261122.json) | 原文投影的配对结果、独立金融题单和 KV 开销 |
| [长文标题提及与容量反馈](source_mentions_20261122.json) | 4 GiB 和 8 GiB 下的长文选材、覆盖和开销 |
| [协同机制](crosslayer_20261118.json) | 来源驻留对选材的影响、未来需求对 Host 回载的影响 |
| [早期连续会话原配置](cooperation.json) · [协同配置](cooperation_optimization.json) | 同题 P3 和固定证据缓存配对。普通 RAG 基线见原配置的 `before`，上一版 ContextKV 见 `current` |

[论文实验与结题报告](../docs/study/final_report.md) · [实验结果与数据划分](../docs/study/results.md) · [运行手册](../docs/study/final_runbook.md)
