# ContextKV 课题索引

| 内容 | 入口 |
|---|---|
| 实施任务与研究边界 | [task_spec.md](task_spec.md) |
| 执行机制与状态 | [架构](../architecture.md) |
| 方法来源与复用 | [参考方法](../references.md) |
| 复现入口 | [final_runbook.md](final_runbook.md) |

## 实测证据

轻量记录保存在运行目录 `runs/study-guide-3db7fb4`。每条记录的范围与对应产物如下。

| 观察 | 结果与范围 | 记录 |
|---|---|---|
| 实际端到端 | `304866a` 的 VersionQA Native/需求保留各完成四轮，末轮均正确引用两版原文。一个多跳用例经三次 READ、一次 REDUCE，引用三份来源并提交。 | `e2e-304866a.E3dwid/result/results.jsonl`、`e2e-read-304866a.W7ni09/multihop.json` |
| 缓存状态影响下一步阅读 | 同一统计题和候选输入下，GPU 命中与 Host 回载改变首读来源，三种状态均提交正确结果及来源。 | `cache-tier-decision-evidence.json` |
| 阅读需求影响缓存 | 同输入压力对照观测到 Host 回载转为 GPU 命中，以及 Host 压力下减少重新预填充。真实 TTL 回放另外覆盖无正文重访窗口。 | `demand-native-evidence.json`、`c-ttl-window/paired-evidence.json` |
| 完整问答中的需求保留 | 固定三次 READ 的输入，两组都引用三份原文并答出 15 kg。短租将 896 token 的 Host 回载转为 GPU 命中，重访后释放。完整响应为 Native 12.256 秒、需求保留 12.242 秒。 | `qa-pressure-4ada1d4.jsnU48/pair.json` |
| READ 与 ANSWER 共用正文前缀 | 已提交的 VersionQA 轨迹中，最终回答实际命中先前阅读所见的正文页。 | `read-answer-prefix-evidence.json` |
| 目录与窗口成本 | 实际 MT-RAG 目录、Qwen tokenizer 和两轮问答分别记录目录分词、初始化及完整响应变化。E1／E6 两轮配对均采用原文回答。 | `task-catalogue-evidence.json`、`mtrag-short-evidence.json`、`mtrag-e1-e6-short-evidence.json` |
| 收益决策 | 当前收益资产使用 24 个去重训练配对。开发集 13 个配对覆盖 7 个可评估决策状态，选中动作相对直接回答的平均目标差为 0。两道补采多跳题的直接回答已经满分，追加 READ 没有提高质量。 | `distinct-material-and-fixed-retention-evidence.json` |
| 必做路径耗时预留 | Qwen3-32B 单并发下，两条受控统计会话产生 8 条实测余程。第三条独立会话正确计数并提交，耗时 3.35 秒，READ、筛选、CPU 归约和 ANSWER 的实际余程均在预留内。 | `completion-timing-f5c5bb1/evidence.json` |

## 完整队列验收

优化前的正式队列保留逐轨覆盖、官方评分和实际付费记录，质量与 P95 门槛尚未验收。队列覆盖及补评分见 `formal-coverage-evidence.json` 和 `formal-recovery-evidence.json`。正式结论使用优化版冻结配置下的 E1–E6、并发 1 与 4、每格 350 题及各自实际 P3 历史，按 [预登记验收口径](final_runbook.md) 判定。
