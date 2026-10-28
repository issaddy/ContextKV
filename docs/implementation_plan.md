# 实现状态与交付

## 当前版本与默认机制

本地分支为 `codex/contextkv-engineering`，会话解释协议为 `conversational-intent-v18`。GPU独立目录`runs/current-mechanism/project`已验证`80d421f`源码及对应原生补丁；常规安装包仍对应`ad84c60`，交付文件身份以`dist/delivery.json`、`dist/remote-delivery.json`为准。

默认使用 Qwen3-8B 生成、独立 Qwen3-32B 非思考解释及原生 NGRAM；检索采用 `single` 融合、`ranked` 选材、原文阅读和 `stable_key` 布局。配置入口见[部署](deployment.md)与[检索指南](guides/hybrid_retrieval.md)。

机制详见[架构](architecture.md)、[多轮状态与证据](research/source_grounded_query_resolution.md)和[成本装配](guides/stage_cost_model.md)。实验数值、否决原型、验证结果和原始记录入口统一维护在[数据与评价](data_and_evaluation.md)。

## 保留的未完成项

1. 改进当前需求到检索的传递、条件作用范围及原文使用。当前公开验证已完成，但任务残留、范围混用和遗漏仍未解决。
2. 减少串行辅助处理和冗余输出，争取质量与完整成本共同改善；当前模型分工和统一协议的实测结论见评价页。
3. 在长上下文、真实并发与 Host 回访负载下识别 B/C 的独立收益。
4. 将验证后的当前源码部署并重新交付。
