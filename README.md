# ContextKV

面向资源受限多轮文档问答的上下文管理与 KV 缓存协同优化研究。

实验固定 Qwen3-32B 权重与解码参数。

业务入口 `DocumentQA` 贯通资料登记、HTTP 问答与 `bench.study`。
`configs/study.yaml` 使用 `selection.mode: ranked` 按实际检索相关性与依赖闭包选材。检索、查询适配与重排资产由 retriever 配置确定；`selection.mode: support` 用于支持模型对照。

系统按实际问答历史、条件与版本选择原文，复用适用的原子事实，并编译原文、结论及混合表示。
SGLang 原生报价参与选择下一步阅读、归约或回答，剩余证据需求决定哪些正文前缀应获得短租。
执行后用新增事实更新证据缺口和保留需求。数值需求共享提取，CPU 执行确定性运算，最终结论保留来源链。

| 主线 | 当前入口 |
|---|---|
| A：实际历史、证据选择与事实复用 | `adapters.study_retrieval`、`controller.fact_reuse` |
| B：显式阅读动作与完整成本 | `controller.action_policy`、`controller.rolling_policy`、`training.plan_gain_rollout`、`training.plan_gain_completion` |
| C：请求编译与缓存协同 | `controller.request_compiler`、ContextPilot index/reorder、`controller.demand`、`controller.prefix_retention` |
| 评价与总成本 | `bench.study_scoring`、`bench.reassess_saved`、`profiling.study_cost_ledger` |

默认在并发 1 和 4 比较普通文档问答 `RAG_native`、完整方案 `context_cache_demand` 与 `LlamaIndex_native`。公开评测保留作者题目、参考答案和历史，按数据集分别报告；系统连续重放使用各方法自身实际历史，构造机制任务使用独立配置。各方法共用本次声明的任务、调度规则与资源配额。目标是同预算提高最终质量，或同质量降低完整响应 P95，所有辅助计算与 GPU/Host 实际占用计入成本。

缓存贡献按需比较 `context_native`、`context_cache_order`、`context_cache_demand`：上下文处理和计算能力相同，只改变缓存组织与需求保留。`Rule_contextpilot_joint_demand` 与 `context_no_fact_reuse` 分别对照完整方案的动作收益决策与跨轮事实复用；当前入口只生成本次选择的方法，旧四格结果留作历史记录。[实验定义](docs/study/task_spec.md#六实验与验收) · [当前结果与归因](docs/study/final_results.md)

`numeric_need_isolation`、`whole_plan`、支持模型及 TTL/value 保留为显式对照入口，默认主线不加载它们。

```bash
python3 -m bench.run_sessions --config configs/demo.yaml demo
```

CPU 测试保留核心阅读、数值归约、来源链、目标可见性、动作决策和前缀接口冒烟。
CI 仅按需手动触发，提交代码和创建 PR 不自动运行测试。
实际模型运行由本次配置明确声明模型、端点、reader 身份、缓存设置、cohort 和输出目录。

| 文档 | 内容 |
|---|---|
| [课题索引](docs/study/README.md) | 当前机制与入口 |
| [任务要求](docs/study/task_spec.md) | 研究边界 |
| [运行手册](docs/study/final_runbook.md) | 当前复现入口 |
| [架构](docs/architecture.md) | 服务与状态链 |
| [数据与评价](docs/data_and_evaluation.md) | 官方任务、完整分母和成本口径 |
| [部署](docs/deployment.md) | 原生引擎与 HTTP 接入 |

`controller/` 为运行核心，`adapters/` 提供资料与检索适配，`engine_patches/` 对接原生缓存；
`bench/`、`profiling/`、`training/` 分别组织评价、成本采集与训练。
