# ContextKV 技术方案入口

**面向资源受限多轮文档问答的上下文管理与 KV 缓存协同优化研究。**

目标是在多资料、长上下文、多轮问答中提高任务理解、证据使用与回答质量，并在此基础上改善完整响应性能和 GPU/Host 资源效率。

技术方案已随工程实现归并到以下文档，原有初版排期、重复伪代码、旧模型配置和实验准入清单由当前接口与实现替代。初版全文保存在 Git 历史中。

| 内容 | 当前入口 |
|---|---|
| 场景、需求与评价目标 | [项目范围](docs/project_scope.md) |
| A/B/C 架构、接口及事务流程 | [架构](docs/architecture.md) |
| 成本目标及算法依据 | [联合问题建模](docs/research/joint_optimization_model.md) |
| 任务状态与证据组织 | [多轮理解](docs/research/source_grounded_query_resolution.md) |
| 原生缓存、报价与生命周期接线 | [SGLang 集成](engine_patches/README.md) |
| 方法复用与课题工作 | [机制与贡献](docs/research_contributions.md)、[相关工作](docs/research/related_work.md) |
| 数据、实验协议、指标与已有结果 | [数据与评价](docs/data_and_evaluation.md) |
| 启动与使用 | [README](README.md)、[部署](docs/deployment.md) |
| 交付状态与后续事项 | [实现状态](docs/implementation_plan.md) |

主流程为：问题与实际历史 → 任务和证据组织 → APPEND/REBUILD 候选 → 原生缓存报价及完整成本选择 → 生成 → 状态提交 → 旧引用退役与成本反馈。

A 管理回答所需的上下文，B 选择当前执行方案，C 协调上下文版本与缓存引用。SGLang 承担物理缓存、调度、传输和淘汰。机制可随实际效果组合、优化或替换，质量、性能和资源结果按目标场景共同评价。
