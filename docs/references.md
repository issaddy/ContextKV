# 当前机制的复用依据

本文只维护当前主线使用的基础能力与直接相关的方法。课题要求见 [任务书](study/task_spec.md)，历史问题和结果解释见 [审查基线](study/review_20261104.md)；运行记录见 [数据与评价](data_and_evaluation.md)。历史文献选型和旧实验数值通过 Git 对应版本追溯。

| 能力与来源 | 当前复用 | 下一步使用原则 |
|---|---|---|
| [SGLang / RadixAttention](https://arxiv.org/abs/2312.07104)、[HiCache](https://docs.sglang.io/docs/advanced_features/hicache) | 原生推理、因果前缀复用、GPU/Host KV；项目补丁提供真实报价和来源 owner | 先稳定真实来源 token，再优化保留；以同请求轨迹的原生对照测净收益 |
| [SGLang Model Gateway](https://github.com/sgl-project/sglang/tree/main/sgl-model-gateway) | 云端多副本部署的上游集成方向；当前补丁测量范围为单 worker | 复用其路由与服务接口；集群适配不增加研究主模块 |
| [BGE-M3](https://huggingface.co/BAAI/bge-m3)、[BGE-reranker-v2-m3](https://huggingface.co/BAAI/bge-reranker-v2-m3) | 查询/文档表示、交叉编码器；查询 LoRA、任务头、支持及集合效用属于项目适配 | 按候选上限与选择遗憾确定训练目标，按最终质量和完整在线成本选资产 |
| [Dozat & Manning：Deep Biaffine Attention for Neural Dependency Parsing（2017）](https://nlp.stanford.edu/pubs/dozat2017deep.pdf) | 任务学生模型复用 span 节点与双线性边分类结构表达关联 | Need 条件、主体、前驱和范围是项目任务图适配；不声称直接复现论文或迁移其性能 |
| [Bertsekas：Rollout / MPC](https://www.mit.edu/~dimitrib/Newton%27sMethodforRLMPC.pdf) | B 使用固定有界后续评价首动作，线上执行一步后依据新观测重规划；剩余资源并入状态 | 复用有限前瞻和滚动决策结构；本项目使用近似回归、部分观测与有限后续，不把经典精确值函数结论直接当作质量保证 |
| [Khuller–Moss–Naor：Budgeted Maximum Coverage](https://doi.org/10.1016/S0020-0190(99)00031-9) | C 以读取需求为收益对象、共享页为容量对象，比较绝对边际价值与容量密度两种构造 | GPU/Host 双预算、同 owner 多租期和带代价目标是本项目适配；保留真实去重目标，不声称复现论文的单预算近似界 |
| [REPLUG](https://arxiv.org/abs/2301.12652) | 冻结生成模型指导检索的相关方法依据 | 当前偏好监督使用固定 Qwen 的实际候选回答；不把该适配描述为完整复现 REPLUG 集成生成 |
| [CacheBlend](https://arxiv.org/abs/2405.16444) | 非前缀 KV 融合的相关工作，当前未实现 | 本阶段先完成精确原生前缀路线；如后续引入，单独评价重算成本与输出影响 |
| [MT-RAG](https://github.com/IBM/mt-rag-benchmark)、[MultiDoc2Dial](https://github.com/IBM/multidoc2dial) | 公开多轮文档任务和作者评价；项目 P3 使用自己的实际历史 | 固定作者历史与自身生成历史分轨报告 |
| [Loong](https://github.com/MozerWang/Loong)、[2WikiMultihopQA](https://github.com/Alab-NII/2wikimultihop)、[VersionQA](https://github.com/danielhuwiler/versionrag) | 长资料、多跳和版本能力补充 | 公开原任务与项目构造机制任务分开；单轮题不自动形成跨轮缓存实验 |

确定性 count/sum、量纲转换、来源依赖闭包、有限 DAG 和现有成本采样直接复用当前实现。新增方法应明确改变哪个决策、所需在线信息、训练标签、完整费用及相应消融。已有第三方代码说明与许可保留在 `docs/third_party/` 和各模块 NOTICE 中。


当前复用核对：BGE-M3 作者说明推荐混合召回后接 reranker；reranker 输出 query–passage 相关性分数，本项目的支持关系与集合效用是额外监督任务，不能把相关性直接当成事实支持。HiCache 官方设计以共享 token 前缀为复用条件，并区分 GPU/Host/存储层；当前实现保持固定 vendor 版本与原生 owner 接口，不随最新文档更换引擎。来源见 [BGE-M3 模型卡](https://huggingface.co/BAAI/bge-m3)、[reranker 模型卡](https://huggingface.co/BAAI/bge-reranker-v2-m3)、[HiCache 设计](https://docs.sglang.io/docs/advanced_features/hicache_design)。
