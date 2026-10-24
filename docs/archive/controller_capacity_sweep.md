# 研究引擎容量与标定验证 helpers

> 归档记录：保留当时的设计、结果与复用依据；其中“下一步”和执行计划不代表当前任务。当前工作仅以[实现清单](../implementation_plan.md)为准。

固定模型的 `bench.controller_capacity_sweep` CLI已删除，不再启动7B容量实验。正式执行统一使用显式Qwen3 [owned矩阵](../guides/b6_owned_matrix.md)。原始有限容量结果仍为只读历史材料，不能改标用于新模型。

保留三个通用函数：`artifact_plan`检查现有artifact的声明容量与来源；`check_research_checkout`核固定patched源码；`validate_runtime`核实际pool/cache/写策略/host大小和可选成本绑定。声明匹配后仍需真实runtime检查，不能改hash或让不匹配模型被标calibrated。host大小由调用者传入，默认16仅用于已有历史验证函数的读记录合同，不启动服务。

这里没有旧方法列表、自动容量网格、强制1024生成或模型路径兜底。运行与失败分母、native/controller共同资源、冻结重复和质量均复用当前声明式矩阵入口。
