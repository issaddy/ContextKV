# 当前机制的复用依据

本文只维护当前主线使用的基础能力与直接相关的方法。课题定位、实施次序和结果解释见 [当前审查](study/review_20261104.md)；运行记录见 [数据与评价](data_and_evaluation.md)。历史文献选型和旧实验数值通过 Git 对应版本追溯。

| 能力与来源 | 当前复用 | 下一步使用原则 |
|---|---|---|
| [SGLang / RadixAttention](https://arxiv.org/abs/2312.07104)、[HiCache](https://docs.sglang.io/docs/advanced_features/hicache) | 原生推理、因果前缀复用、GPU/Host KV；项目补丁提供真实报价和来源 owner | 先稳定真实来源 token，再优化保留；以同请求轨迹的原生对照测净收益 |
| [SGLang Model Gateway](https://github.com/sgl-project/sglang/tree/main/sgl-model-gateway) | 云端多副本部署的上游集成方向；当前补丁测量范围为单 worker | 复用其路由与服务接口；集群适配不增加研究主模块 |
| [BGE-M3](https://huggingface.co/BAAI/bge-m3)、[BGE-reranker-v2-m3](https://huggingface.co/BAAI/bge-reranker-v2-m3) | 查询/文档表示、交叉编码器；查询 LoRA、任务头、支持及集合效用属于项目适配 | 按候选上限与选择遗憾确定训练目标，按最终质量和完整在线成本选资产 |
| [REPLUG](https://arxiv.org/abs/2301.12652) | 冻结生成模型指导检索的相关方法依据 | 当前偏好监督使用固定 Qwen 的实际候选回答；不把该适配描述为完整复现 REPLUG 集成生成 |
| [CacheBlend](https://arxiv.org/abs/2405.16444) | 非前缀 KV 融合的相关工作，当前未实现 | 本阶段先完成精确原生前缀路线；如后续引入，单独评价重算成本与输出影响 |
| [MT-RAG](https://github.com/IBM/mt-rag-benchmark)、[MultiDoc2Dial](https://github.com/IBM/multidoc2dial) | 公开多轮文档任务和作者评价；项目 P3 使用自己的实际历史 | 固定作者历史与自身生成历史分轨报告 |
| [Loong](https://github.com/MozerWang/Loong)、[2WikiMultihopQA](https://github.com/Alab-NII/2wikimultihop)、[VersionQA](https://github.com/danielhuwiler/versionrag) | 长资料、多跳和版本能力补充 | 公开原任务与项目构造机制任务分开；单轮题不自动形成跨轮缓存实验 |

确定性 count/sum、量纲转换、来源依赖闭包、有限 DAG 和现有成本采样直接复用当前实现。新增方法应明确改变哪个决策、所需在线信息、训练标签、完整费用及相应消融。已有第三方代码说明与许可保留在 `docs/third_party/` 和各模块 NOTICE 中。
