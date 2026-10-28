# 实现状态与交付

## 当前版本与默认机制

本地分支为 `main`，会话解释协议为 `conversational-intent-v18`。GPU独立目录`runs/current-query/baseline`已验证`1c4b14d`运行源码及对应原生补丁；此前`80d421f`的验证保留在`runs/current-mechanism/project`。常规安装包仍对应`ad84c60`，交付文件身份以`dist/delivery.json`、`dist/remote-delivery.json`为准。

默认使用 Qwen3-8B 生成、独立 Qwen3-32B 非思考解释及原生 NGRAM；检索采用 `single` 融合、`ranked` 选材、原文阅读和 `stable_key` 布局。配置入口见[部署](deployment.md)与[检索指南](guides/hybrid_retrieval.md)。

机制详见[架构](architecture.md)、[多轮状态与证据](research/source_grounded_query_resolution.md)和[成本装配](guides/stage_cost_model.md)。实验数值、否决原型、验证结果和原始记录入口统一维护在[数据与评价](data_and_evaluation.md)。

## 保留的未完成项

各阶段按[目标场景与纠偏原则](project_scope.md#实现过程中的纠偏)选择工作。最近98轮基线输入均不足8K tokens、最多9轮用户交互、并发1；已有材料集合切换，但尚无来源版本变化或Host恢复的实测。这些诊断不能代替完整目标场景验证。

1. 围绕目标场景扩大通用机制调研，改进当前需求、持续条件和来源适用范围在回答中的落实。原话与派生任务的单查询拼接已完成公开闭环比较，未采用；新增大模型阅读规划原型已在模型调用前撤下。模型与机制的边界见[项目范围](project_scope.md#模型边界与机制归因)，实测见[评价记录](data_and_evaluation.md#当前原话与派生任务的单查询拼接未采用)。
2. 减少串行辅助处理和冗余输出。当前解释阶段与last3无解释的196份回答消融已完成：完整响应降低约30%，质量在两个开发队列方向不同、区间均跨0，来源复核同时发现对象丢失和旧条件残留。暂不统一更改默认；优先改善条件作用域与必要调用，见[必要性消融](data_and_evaluation.md#现有任务解释阶段的必要性消融)。
3. 推进长上下文、真实并发与Host回访验证。Loong第一组323题已接入，122题完整输入完成P1基线，其中106题超过16K；201题预算失败保留（145题输入本身超限，56题受4096输出预留限制）。完整材料在场仍有关系方向和对象分类错误，参考式均分56.79/100；不把长输入可运行当成质量改善。下一步处理需求/证据覆盖及预算策略，多轮、版本和并发仍需单独验证，见[长资料基线](data_and_evaluation.md#loong完整长资料p1基线)。
4. 将验证后的当前源码部署并重新交付。
