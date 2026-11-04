# 当前机制的复用依据

本文只维护当前主线使用的基础能力与直接相关的方法。课题要求见 [任务书](study/task_spec.md)，机制见 [架构](architecture.md)；运行与评分口径见 [数据与评价](data_and_evaluation.md)。历史文献选型和旧实验数值通过 Git 对应版本追溯。

| 能力与来源 | 当前复用 | 下一步使用原则 |
|---|---|---|
| [SGLang / RadixAttention](https://arxiv.org/abs/2312.07104)、[HiCache](https://docs.sglang.io/docs/advanced_features/hicache) | 原生推理、因果前缀复用、GPU/Host KV；项目补丁提供真实报价和来源 owner | 先稳定真实来源 token，再优化保留；以同请求轨迹的原生对照测净收益 |
| [SGLang Model Gateway](https://github.com/sgl-project/sglang/tree/main/sgl-model-gateway) | 云端多副本部署的上游集成方向；当前补丁测量范围为单 worker | 复用其路由与服务接口；集群适配不增加研究主模块 |
| [BGE-M3](https://huggingface.co/BAAI/bge-m3)、[BGE-reranker-v2-m3](https://huggingface.co/BAAI/bge-reranker-v2-m3)、[LoRA](https://arxiv.org/abs/2106.09685) | 编码器和交叉编码器直接复用；`task_model.py` 的查询 LoRA 接检索损失，任务头接 detached 特征上的任务损失；`support_model.py` 接支持和集合偏好 | 未复现 BGE-M3 的多功能自知识蒸馏。学习资产仅在配置显式选择时启用；当前主线仍可使用任务/支持教师和 coverage 效用 |
| [Dozat & Manning：Deep Biaffine Attention for Neural Dependency Parsing（2017）](https://nlp.stanford.edu/pubs/dozat2017deep.pdf) | `task_model.py` 借鉴图边评分，使用投影点积的双线性关联头 | Need 条件、主体、前驱和范围是项目任务图适配；没有直接复现 biaffine 依存句法模型 |
| [Bertsekas：Rollout / MPC](https://www.mit.edu/~dimitrib/Newton%27sMethodforRLMPC.pdf) | B 使用固定有界后续评价首动作，线上执行一步后依据新观测重规划；剩余资源并入状态 | 复用有限前瞻和滚动决策结构；本项目使用近似回归、部分观测与有限后续，不把经典精确值函数结论直接当作质量保证 |
| [Khuller–Moss–Naor：Budgeted Maximum Coverage](https://doi.org/10.1016/S0020-0190(99)00031-9) | C 以读取需求为收益对象、共享页为容量对象，比较绝对边际价值与容量密度两种构造 | GPU/Host 双预算、同 owner 多租期和带代价目标是本项目适配；保留真实去重目标，不声称复现论文的单预算近似界 |
| [REPLUG](https://arxiv.org/abs/2301.12652) | 冻结生成模型指导检索的相关方法依据 | 当前偏好监督使用固定 Qwen 的实际候选回答；不把该适配描述为完整复现 REPLUG 集成生成 |
| [CacheBlend](https://arxiv.org/abs/2405.16444) | 非前缀 KV 融合的相关工作，当前未实现 | 本阶段先完成精确原生前缀路线；如后续引入，单独评价重算成本与输出影响 |
| [MT-RAG](https://github.com/IBM/mt-rag-benchmark)、[MultiDoc2Dial](https://github.com/IBM/multidoc2dial) | 公开多轮文档任务和作者评价；项目 P3 使用自己的实际历史 | 固定作者历史与自身生成历史分轨报告 |
| [Loong](https://github.com/MozerWang/Loong)、[2WikiMultihopQA](https://github.com/Alab-NII/2wikimultihop)、[VersionQA](https://github.com/danielhuwiler/versionrag) | 长资料、多跳和版本能力补充 | 公开原任务与项目构造机制任务分开；单轮题不自动形成跨轮缓存实验 |

确定性 count/sum、量纲转换、来源依赖闭包、有限 DAG 和现有成本采样直接复用当前实现。新增方法应明确改变哪个决策、所需在线信息、训练标签、完整费用及相应消融。已有第三方代码说明与许可保留在 `docs/third_party/` 和各模块 NOTICE 中。

## 实际落点与方法取舍

这些方法组成资源预算下的近似方案，当前没有逐环节最优或全局最优的实测依据。

| 环节 | 代码中的实际方法 | 相关替代与采用条件 |
|---|---|---|
| 召回 | `hybrid_retrieval.py` 已有 dense、BM25+dense RRF、hybrid_rerank；主配置 `study_multidoc_retriever.yaml` 选择 dense | BGE 作者建议混合召回后重排；专名、数值和版本召回不足时可直接启用已有 hybrid 路径。避免在 A 支持评分之前无依据再加一遍交叉编码器 |
| 任务/证据 | `support_model.query_view` 传 Need scope、条件及完整版本意图；`evidence_planner.py` 做依赖闭包束搜索，P1 在 `train_set_utility` 学 pairwise log-sigmoid 偏好，线上 `learned` 消费同一图输入 | 查询适配与任务头分开训练；先看候选最优值与选择遗憾，不能把训练入口存在等同于学生优于教师或 P1 优于 coverage |
| 阅读/数值 | `reading.py`、`reduction.py`、`claim_graph.py` 共享提取、按 Need 选择、CPU 固定运算、有限 DAG，`candidates.py` 编译完整结论 | [Program of Thoughts](https://arxiv.org/abs/2211.12588) 将计算交给程序执行；本项目采用固定运算符，没有复现模型生成任意程序。当前 count/sum 无需额外程序生成模型 |
| 多跳/停止 | `action_proposals.py`、`plan_gain_protocol.py` 对真实未满足的 infer 缺口做有界续接，`plan_gain_completion.py` 拟合实际最终质量差，`action_policy.py` 执行一步后重规划 | [IRCoT](https://aclanthology.org/2023.acl-long.557/) 的推理与检索交替是相关依据；当前用有来源的桥接实体与 OpenNeed，不复现自由文本 CoT 循环。更长前瞻只在现有边界确实截断必要链路时考虑 |
| 前缀/期限 | `prefix_retention.py` 比较绝对边际价值和容量密度两种贪心构造；`retention_trace.py fit` 的单 owner 配对经验模型被线上 observer 直接使用 | 当前既非精确覆盖求解，也未学习多 owner 联合干预的全部挤出效应；集合净收益由完整 Native/value 对照测量。CacheBlend 适用于非前缀复用瓶颈，但需改变 KV 执行协议，不能直接套入现有收益资产 |
| 阶段错误归因 | `reader_data.py` 直接核验实际 READ/REDUCE/ANSWER 输出，区分可见输入支持、完整原文核验及上游错误 | 不再先生成更好答案再冒充实际运行质量；任务/支持教师的监督生成仍保留自己的输入可见性规则 |

## 保留的可选对照

- [ContextPilot](https://arxiv.org/abs/2511.03475)：`adapters/contextpilot.py` 保留会话内重叠顺序对齐与引用注释，供 `contextpilot` 策略使用；没有复现跨会话索引与调度。
- [EfficientAgent 的 Host 准入适配](third_party/EFFICIENTAGENT_NOTICE.md)：原生补丁 `HostAdmissionPolicy` 默认关闭；显式启用时配置进入 cache scope，线上与配对重放均传真实业务会话身份。它是 Host 写入对照，未替代 C 的 TTL 分配。
- [CRAFT 行文本函数](third_party/CRAFT_NOTICE.md)与[表格关系适配](research/table_relations.md)：仅在表格注释选项启用时使用，未引入上游完整检索系统。

当前复用核对：BGE-M3 作者说明推荐混合召回后接 reranker；reranker 输出 query–passage 相关性分数，本项目的支持关系与集合效用是额外监督任务，不能把相关性直接当成事实支持。HiCache 官方设计以共享 token 前缀为复用条件，并区分 GPU/Host/存储层；当前实现保持固定 vendor 版本与原生 owner 接口，不随最新文档更换引擎。来源见 [BGE-M3 模型卡](https://huggingface.co/BAAI/bge-m3)、[reranker 模型卡](https://huggingface.co/BAAI/bge-reranker-v2-m3)、[HiCache 设计](https://docs.sglang.io/docs/advanced_features/hicache_design)。
