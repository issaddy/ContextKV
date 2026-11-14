# 当前实测指标

[current.json](current.json) 独立保存四个 benchmark 的 ContextKV 与普通文档问答基线结果：同题清单、质量、完整响应 P95、GPU/Host 占用、输入输出 token、缓存计数及评分费用。

每项结果记录实际实现提交，`records_path` 指向容器中对应的逐题答案和来源。MT-RAG 的一项评分缺失保留完整分母区间。

结果说明见[评测报告](../docs/study/results.md)，执行命令见[运行手册](../docs/study/final_runbook.md)。
