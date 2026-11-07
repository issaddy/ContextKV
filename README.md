# ContextKV

面向资源受限多轮文档问答的上下文管理与 KV 缓存协同优化研究。

实验固定 Qwen3-32B 权重与解码参数。

业务入口 `DocumentQA` 贯通资料登记、HTTP 问答与 `bench.study`。
`configs/study.yaml` 使用 `selection.mode: ranked` 按发布版重排与依赖闭包选材。`selection.mode: support` 用于支持模型对照。

系统按实际问答历史、条件与版本选择原文，复用适用的原子事实，并编译原文、结论及混合表示。
SGLang 原生报价参与选择下一步阅读、归约或回答，剩余证据需求决定哪些正文前缀应获得短租。
执行后用新增事实更新证据缺口和保留需求。数值需求共享提取，CPU 执行确定性运算，最终结论保留来源链。

| 主线 | 当前入口 |
|---|---|
| A：实际历史、证据选择与事实复用 | `adapters.study_retrieval`、`controller.fact_reuse` |
| B：显式阅读动作与完整成本 | `controller.action_policy`、`controller.rolling_policy`、`training.plan_gain_rollout`、`training.plan_gain_completion` |
| C：请求编译与缓存协同 | `controller.request_compiler`、ContextPilot index/reorder、`controller.demand`、`controller.prefix_retention` |
| 评价与总成本 | `bench.study_scoring`、`bench.reassess_saved`、`profiling.study_cost_ledger` |

并发 1 和 4 各运行 E1–E6。E1–E4 交叉比较固定／收益决策与 Native／ContextPilot 编译，E5 增加有限表示选择，E6 再增加剩余需求驱动的保留。另有 LlamaIndex 外部参照和跨轮事实复用消融。各格共用 350 题、77 会话与固定到达计划，使用自身实际 P3 历史。验收以 E6 对 E1 的完整队列质量、完整响应 P95 和全部资源成本为准。

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
