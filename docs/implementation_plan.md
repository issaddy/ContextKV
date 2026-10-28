# 实现状态与交付

## 当前版本与默认机制

本地分支为 `main`，会话解释协议为 `conversational-intent-v18`。GPU独立目录`runs/current-query/baseline`已验证`1c4b14d`运行源码及对应原生补丁；此前`80d421f`的验证保留在`runs/current-mechanism/project`。常规安装包仍对应`ad84c60`，交付文件身份以`dist/delivery.json`、`dist/remote-delivery.json`为准。

新增可选`dialogue`查询适配器，使用已训练BGE查询侧LoRA和原始文档索引，免去在线解释与重排；入口为`configs/retriever_dialogue.example.yaml`。360份完整回答对照及196次整理后GPU检索复核已完成，默认未切换。

默认使用 Qwen3-8B 生成、独立 Qwen3-32B 非思考解释及原生 NGRAM；检索采用 `single` 融合、`ranked` 选材、原文阅读和 `stable_key` 布局。配置入口见[部署](deployment.md)与[检索指南](guides/hybrid_retrieval.md)。

机制详见[架构](architecture.md)、[多轮状态与证据](research/source_grounded_query_resolution.md)和[成本装配](guides/stage_cost_model.md)。实验数值、否决原型、验证结果和原始记录入口统一维护在[数据与评价](data_and_evaluation.md)。

## 保留的未完成项

各阶段按[目标场景与纠偏原则](project_scope.md#实现过程中的纠偏)选择工作。最近会话查询适配器对照完成360份回答，7份输入达到8K、最大9299 tokens、最多9轮用户交互、并发1；已有材料集合切换，但尚无来源版本变化或Host恢复的实测。这些诊断不能代替完整目标场景验证。

1. 围绕目标场景扩大通用机制调研，改进当前需求、持续条件和来源适用范围在回答中的落实。原话与派生任务的单查询拼接已完成公开闭环比较，未采用；新增大模型阅读规划原型已在模型调用前撤下。模型与机制的边界见[项目范围](project_scope.md#模型边界与机制归因)，实测见[评价记录](data_and_evaluation.md#当前原话与派生任务的单查询拼接未采用)。
2. 减少辅助调用，并优先保住当前用户需求。会话查询训练对照中，Cloud评分与基线接近，新12个MultiDoc2Dial测试会话提高，完整响应分别降低约36%和40%，已保留为可选组件。仍有SSA规则混入VA、用户明确提及VA后沿着模型旧FAFSA回答继续检索等错误；需要处理用户原话、助手历史及来源条件的不同作用。见[最新对照](data_and_evaluation.md#查询侧会话检索训练与完整系统对照)，不能把全部加速或训练收益归因于A/B/C。
3. 推进长上下文、真实并发与Host回访验证。Loong第一组323题已接入，122题完整输入完成P1基线，其中106题超过16K；201题预算失败保留（145题输入本身超限，56题受4096输出预留限制）。完整材料在场仍有关系方向和对象分类错误，参考式均分56.79/100；不把长输入可运行当成质量改善。下一步处理需求/证据覆盖及预算策略，多轮、版本和并发仍需单独验证，见[长资料基线](data_and_evaluation.md#loong完整长资料p1基线)。
4. 将验证后的当前源码部署并重新交付。
