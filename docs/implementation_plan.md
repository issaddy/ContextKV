# 实现状态与交付

长期研发目标保持暂停，未完成项保留。容器及现有安装包对应交付版本`ad84c60`，其运行基线为`edad283`；本地源码随后完成工程清理与上下游修正，尚未重新部署到容器。

## 模块状态

| 模块 | 已完成 | 剩余重点 |
|---|---|---|
| A | 多轮任务/条件状态、因果历史、检索选材、版本依赖、原文共享和共同引用范围；接入CLI/HTTP及原生对照 | 最新v15的真实回答质量、条件继承及纠正效果 |
| B | 原生GPU/Host报价、Qwen3-8B/H200成本装配、完整成本选择、输出长度与控制开销反馈 | 解释及生成成本与缓存收益的整体权衡 |
| C | 业务epoch与原生引用关联、提交后退役、共享引用、outbox、响应丢失恢复与重启隔离 | 并发竞争和Host回访中的服务及资源收益 |
| 工程 | 自动检索、多会话、事务、幂等、关闭、部署、发行包和可追溯评测 | 将后续已验证改进整合到下一次交付 |

机制与适配见[研究依据](research_contributions.md)，跨模块流程见[架构](architecture.md)。

## 当前配置与验证

默认使用Qwen3-8B生成、独立Qwen3-32B解释服务及原生NGRAM；检索为`conversational-intent-v15`、`single`融合、`ranked`选材，布局为`stable_key`。原文是默认阅读表示。完整启动参数和可选配置统一见[部署](deployment.md)与[检索指南](guides/hybrid_retrieval.md)。

本地工程清理已移除旧稳定轮数门槛、重复候选检查和缺报价切换策略分支，统一模板/引擎接口；维护错误、有效回答及统计异常分别处理。drain并发与取消关闭、离线计数处理和BM25版本记录已完成上下游修正。旧配置迁移见[配置接口](deployment.md#清理版本的配置接口)。

本地运行代码验证为**1738 passed、16 skipped、8 subtests**，最后的API接线回归为**19 passed**；ruff通过。文档整理只检查职责、链接、来源及配置说明，未启动模型实验。

上一阶段容器验证为**124 passed、11 deselected**，两端wheel/sdist及独立安装演示通过，300份源码完成当时的部署校验。历史源码和安装包清单保持原版本。

## 实测结果入口

| 内容 | 记录位置 |
|---|---|
| Cloud完整系统对照 | [固定公开P3比较](data_and_evaluation.md#固定公开p3主比较2026-10-27)、`runs/public-main/` |
| 跨文档系统对照 | [MultiDoc2Dial比较](data_and_evaluation.md#multidoc2dial跨文档完整系统比较)、`runs/multidoc-system/` |
| 联合选材、原文结构、状态输入与控制诊断 | [机制与分版本结果](data_and_evaluation.md)、对应`runs/`原件 |

## 交付产物

| 内容 | 位置 |
|---|---|
| 当前源码与文档 | 仓库根目录、[README导航](../README.md#文档导航) |
| 上次交付的wheel、sdist与源码包 | `dist/contextkv-0.1.0-py3-none-any.whl`、`dist/contextkv-0.1.0.tar.gz`、`dist/contextkv-source.tar.gz` |
| 本地/容器交付清单 | `dist/delivery.json`、`dist/remote-delivery.json`，按主机记录实际文件与本地后续修订 |
| 原始实验与审查记录 | `runs/`，按各运行源码版本保留 |
| 容器项目 | `/home/relay/songjiayun/ContextKV` |

## 保留的未完成项

1. 验证最新A协同对真实多轮任务、条件继承和纠正的收益。
2. 降低解释成本、冗余输出及旧结论沿用的代价，改善系统净收益。
3. 在有意义的长上下文、并发工作集与Host回访负载下验证B/C。
