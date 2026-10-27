# 可选离线训练与导出工具

`training/`保留可复用的离线准备、SFT和LoRA合并接口，正常ContextKV服务不导入它，也不需要训练权重。当前研发主线使用已有Qwen3基座完善A/B/C；早期训练路线及小样记录保留在Git和`runs/grounded-training-reuse/`、`runs/grounded-supervision-design/`，不再作为当前步骤或验收要求。

## 保留的接口

| 入口 | 作用 |
|---|---|
| `python -m training.sft prepare --help` | 将已审核的messages/target转换为真实模型前缀及仅目标部分的监督tokens |
| `python -m training.sft train --help` | 在独立环境中用Transformers/PEFT运行显式指定步数的离线LoRA训练 |
| `python -m training.export --help` | 使用PEFT合并接口导出独立dense checkpoint和模型scope，供现有服务读取 |

可选依赖通过`UV_PROJECT_ENVIRONMENT=.venv-training uv sync --extra training --locked`安装，与推理环境分开。准备输入的每条记录包含`id`、`split`、`messages`、`target`和`provenance`；来源审核与开发/测试划分由数据准备者负责，工具不会把原作者grounding自动当成正确性认证。

训练前缀直接复用`HuggingFaceTemplate.render`，历史/来源全部mask，仅目标及EOS参与loss，collator保留已有labels；不通过完整chat减prefix来猜目标边界。导出的权重拥有独立身份，按[模型scope接口](../guides/model_scope.md)与[部署入口](../deployment.md)读取，原基座KV和成本表不能直接混用于新权重。当前原生guard不提供动态LoRA切换。

[RAFT](https://arxiv.org/abs/2403.10131)和[PA-RAG](https://aclanthology.org/2025.naacl-long.459/)提供证据抗干扰、完整性和来源偏好的研究参考。本项目保留相关工具能力，没有把训练损失、成功导出或工具单测当成回答质量收益；目前不开展训练实验。
