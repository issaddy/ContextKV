# 方法复用与候选评估

本文维护当前主线的复用依据及候选方法评估，调研更新于 2026-11-04。课题要求见 [任务书](study/task_spec.md)，机制见 [架构](architecture.md)；运行与评分口径见 [数据与评价](data_and_evaluation.md)。方法状态以当前代码为准，论文报告的收益不作为本项目实测结果。历史文献选型和旧实验数值通过 Git 对应版本追溯。

| 能力与来源 | 当前复用 | 下一步使用原则 |
|---|---|---|
| [SGLang / RadixAttention](https://arxiv.org/abs/2312.07104)、[HiCache](https://docs.sglang.io/docs/advanced_features/hicache) | 原生推理、因果前缀复用、GPU/Host KV；项目补丁提供真实报价和来源 owner | 先稳定真实来源 token，再优化保留；以同请求轨迹的原生对照测净收益 |
| [SGLang Model Gateway](https://github.com/sgl-project/sglang/tree/main/sgl-model-gateway) | 云端多副本部署的上游集成方向；当前补丁测量范围为单 worker | 复用其路由与服务接口；集群适配不增加研究主模块 |
| [BGE-M3](https://huggingface.co/BAAI/bge-m3)、[BGE-reranker-v2-m3](https://huggingface.co/BAAI/bge-reranker-v2-m3)、[LoRA](https://arxiv.org/abs/2106.09685) | 编码器和交叉编码器直接复用；`task_model.py` 的查询 LoRA 接检索损失，任务头接 detached 特征上的任务损失；`support_model.py` 接支持和集合偏好 | 未复现 BGE-M3 的多功能自知识蒸馏。学习资产仅在配置显式选择时启用；当前主线仍可使用任务/支持教师和 coverage 效用 |
| [Dozat & Manning：Deep Biaffine Attention for Neural Dependency Parsing（2017）](https://nlp.stanford.edu/pubs/dozat2017deep.pdf) | `task_model.py` 借鉴图边评分，使用投影点积的双线性关联头 | Need 条件、主体、前驱和范围是项目任务图适配；没有直接复现 biaffine 依存句法模型 |
| [Bertsekas：Rollout / MPC](https://www.mit.edu/~dimitrib/Newton%27sMethodforRLMPC.pdf) | B 使用固定有界后续评价首动作，线上执行一步后依据新观测重规划；剩余资源并入状态 | 复用有限前瞻和滚动决策结构；本项目使用近似回归、部分观测与有限后续，不把经典精确值函数结论直接当作质量保证 |
| [Khuller–Moss–Naor：Budgeted Maximum Coverage](https://doi.org/10.1016/S0020-0190(99)00031-9) | C 以读取需求为收益对象、共享页为容量对象，比较绝对边际价值与容量密度两种构造 | 单一逻辑保护容量、同 owner 多租期和分层驻留代价是本项目适配；保留真实去重目标，不声称复现论文的近似界 |
| [REPLUG](https://arxiv.org/abs/2301.12652) | 冻结生成模型指导检索的相关方法依据 | 当前偏好监督使用固定 Qwen 的实际候选回答；不把该适配描述为完整复现 REPLUG 集成生成 |
| [CacheBlend](https://arxiv.org/abs/2405.16444) | 非前缀 KV 融合的相关工作，当前未实现 | 本阶段先完成精确原生前缀路线；如后续引入，单独评价重算成本与输出影响 |
| [MT-RAG](https://github.com/IBM/mt-rag-benchmark)、[MultiDoc2Dial](https://github.com/IBM/multidoc2dial) | 公开多轮文档任务和作者评价；项目 P3 使用自己的实际历史 | 固定作者历史与自身生成历史分轨报告 |
| [Loong](https://github.com/MozerWang/Loong)、[2WikiMultihopQA](https://github.com/Alab-NII/2wikimultihop)、[VersionQA](https://github.com/danielhuwiler/versionrag) | 长资料、多跳和版本能力补充 | 公开原任务与项目构造机制任务分开；单轮题不自动形成跨轮缓存实验 |

确定性 count/sum、量纲转换、来源依赖闭包、有限 DAG 和现有成本采样直接复用当前实现。新增方法应明确改变哪个决策、所需在线信息、训练标签、完整费用及相应消融。已有第三方代码说明与许可保留在 `docs/third_party/` 和各模块 NOTICE 中。

## 评估口径

优先级是依据当前课题约束作出的工程判断，没有逐环节最优或全局最优的实测依据。评估同时考虑：是否保留需求、条件、版本与来源；是否需要改变固定 Qwen；能否复用现有接口和资产；离线建索引/训练成本；在线解释、检索、阅读、控制与驻留总成本。压缩率、召回率和 TTFT 单项改善不能替代完整队列质量及完整响应 P95。

下表的“已有”指代码与调用已接通，“局部借鉴”指采用同类机制但未复现整篇方法，“候选”指尚未接入。候选不自动成为实施清单，只有对应瓶颈出现才比较；文档更新不改变冻结配置。

## 从目标场景反推主方法

主目标是在固定 Qwen、资料和资源条件下，提高完整队列的最终质量；或保持质量并降低完整响应 P95。各环节按对这一目标的贡献选择，单独最大化召回、压缩率或缓存命中率不能确定联合最优。以下是主方法选型，具体模型优劣与参数取值仍由当前任务数据决定。

| 场景与预期效果 | 必要机制 | 明确的主方法 |
|---|---|---|
| 多轮追问、纠正和版本切换仍回答当前问题 | 保留需求、主体、条件、时间、版本及其真实历史出处，增量更新适用关系 | 显式 TaskState/Need 图＋原文引用＋版本目录过滤。查询改写服务这些语义；教师与 BGE 学生使用相同状态契约 |
| 专名/编号与同义表述都能召回，片段具有解释上下文 | 精确匹配与语义匹配互补；检索粒度与实际阅读粒度分别组织 | BM25＋BGE 融合召回，按 Need 发查询；目录/粗粒度命中映射到稳定原文窗口并补齐依赖。混合融合是主候选，RRF 不具有普遍优于单路排序的定理 |
| 多条材料共同支持答案，预算用在互补证据上 | 对完整证据集合估计最终用途，保留多跳前驱及单/双/三段支持 | 支持关系模型＋集合效用＋依赖闭包预算搜索。现有有限 beam 是可执行近似；MMR 只在评分名额被重复材料占用时辅助多样化 |
| count/sum/all 等问题覆盖全部指定记录 | 明确声明范围与完整覆盖；区分事实提取、按需求选择和运算 | 范围内扫描＋共同事实提取＋各 Need 运算视图＋CPU 确定性聚合。top-k 仅适用于候选定位，无法单独保证全量统计完整性 |
| 已读信息形成完整结论并进入真实 ANSWER | 先确定最终容量，保留足够回答当前问题的事实及来源证明，归约有限结束 | ANSWER 预算反向分配＋短 ID/共享字段＋Claim DAG＋有限 REDUCE。优先压缩结构重复；不以任意删原文 token 替代完整结论 |
| 必要桥接继续，低收益读取停止 | 比较继续动作对最终回答的预期增益和新增完整成本，状态包含剩余预算 | 计算价值＋固定有界后续监督＋一步重规划。优先现有轻量回归；充分性是信息与诊断，不能直接等同于停止价值 |
| 多轮重复计算尽量避免，必要重读尽量便宜 | 区分已知结果复用、来源 KV 复用和重新计算 | 同版本、适用范围、来源及覆盖语义下，先复用已提取的原子事实，按本轮条件重算聚合，再对必要 READ 使用精确前缀 KV。`fact_reuse.py` 已从 SessionStore 的已提交 READ 投影合同并接入任务解释和执行 seed；实际质量、读取节省与新增解释成本由关闭复用的对照评价 |
| GPU/Host 受限、重访间隔不同且存在并发 | 对完整读取需求估计保留相对 Native 的新增节省，计入其他请求受到的影响 | 条件重访估计＋相对 Native 成本差＋共享页/需求去重＋逻辑容量内的期限分配，Native 管理物理层。PGDSF 是便宜对照；固定 TTL 与频率排序不能替代净收益目标 |
| 改动改善完整队列，而非少量成功题或某个局部指标 | 分开定位候选不足、提取错误、最终窗口不足和答案使用错误 | 官方全分母质量＋实际 P3＋逐阶段充分性/来源核验＋全部费用及完整响应。复用现有账本和审核入口 |

B 的决策目标可写成 `E[ΔQ(a | 当前状态, 剩余预算)] − λ·E[新增完整成本(a)]`，其中 ΔQ 和费用都相对当前直接 ANSWER，且包含固定有界后续。已付费用只影响剩余资源。所有合法继续动作的预期净收益都不大于零时选择 ANSWER；必须完成的范围覆盖由任务语义决定。经典 rollout 为此提供结构依据，当前预测器与有限后续只近似这个目标。

C 对新增保护项及期限 `(p, τ)` 的目标是 `ΔV(p, τ | 已保护集合) = 预期目标读取节省 − 新增控制费 − 新增驻留费 − 外部请求代价`。目标读取节省相对同一原生状态、不含右侧另列费用；共享需求与共享页分别只计新增部分。重访概率只是其中一项，即便很快重访，Native 已能命中也可能无需额外保护。当前两种贪心近似求解这个集合目标，没有完整联合最优保证。

可由机制直接确定的性质包括：给定正确完整事实和量纲，CPU 可按声明精度执行确定性运算；相同且适用的已算结果可避免重复计算；有限 DAG 会结束；因果前缀相同才可直接复用对应原生 KV。具体窗口大小、融合权重、beam 宽度、回归模型和 TTL 的最佳值不由这些性质决定。跨轮原子事实复用现已接入已有 contract/Claim DAG，范围完成按来源区间核查；其匹配成本与减少的 READ 一并计入实际响应。

## A：任务、历史、来源与证据选择

| 方法与原始来源 | 可复用机制和当前状态 | 成本、适用条件与判断 |
|---|---|---|
| [CONQRR](https://aclanthology.org/2022.emnlp-main.679/)／[Rewrite-Retrieve-Read](https://aclanthology.org/2023.emnlp-main.322/) | 将历史依赖问题改写为检索查询，分别用检索奖励或 reader 反馈训练改写器。当前有 Need 子查询、原问题查询与查询 LoRA，未实现两者的 RL rewriter | 固定 Qwen 可用；完整方法增加改写生成、训练和奖励采集。**条件候选**：任务解释正确而查询措辞仍造成漏召回时，先比较受现有条件图约束的查询视图，保留原问题流；候选已含答案时不优先训练 rewriter |
| [BGE-M3](https://huggingface.co/BAAI/bge-m3) 混合表示／RRF | `hybrid_retrieval.py` 已有 BM25+dense RRF；主配置选择 dense。未实现 M3 learned sparse 和 multi-vector 三路融合，不能把 BM25 等同于 learned sparse | **优先复用已有 hybrid**，尤其专名、数值、编号和版本精确匹配不足时；无需新生成或训练。新增 M3 表示需索引、存储和评分，当前 dense query LoRA 也没有适配其他输出头 |
| [ColBERTv2](https://aclanthology.org/2022.naacl-main.272/) | token 级 late interaction，结合残差压缩和去噪监督。当前单向量召回与交叉编码器支持评分之间没有该层 | **条件候选**：单向量丢失局部实体/关系、支持模型又无法覆盖更大池时。预训练模型可免训练，但要维护多向量表示并增加匹配成本；先比较有限候选排序，不直接替换多语种 BGE 全索引 |
| [MMR](https://doi.org/10.1145/290941.291025) | 在相关性与已选材料相似度之间取舍。当前有完全相同正文合并、Need 配额及支持覆盖，没有语义 MMR | **优先局部候选**：重复模板占满每 Need 的支持评分名额时，在评分前做多样化。复用向量，仅增加 CPU 选择；保持版本、条件及依赖身份，不把互补材料当冗余，也不以多样性替代支持性 |
| [预算最大覆盖](https://doi.org/10.1016/S0020-0190(99)00031-9) | 当前 `SupportCoverageUtility` 对各 Need 取集合最大支持，`evidence_planner.py` 用依赖闭包束搜索；可比较边际覆盖/新增真实 token 的轻量构造 | **已有相近目标，贪心是条件候选**。适合搜索控制成本突出时；复用已评分支持，不增加模型调用。pair/triple 互补、来源依赖与编译成本改变经典假设，不能直接套近似保证 |
| [REPLUG](https://arxiv.org/abs/2301.12652)／[RAG-DDR](https://arxiv.org/abs/2410.13509) | reader 指导检索/模块偏好。当前 `train_set_utility` 用真实答案的 pairwise log-sigmoid 偏好，线上 `learned` 消费同一图输入；属于局部借鉴 | **保留实际最终效用监督**。REPLUG 的多文档分布集成和似然监督、DDR 的模块训练整法均未复现；完整引入有分支生成或联合训练费用。只优化辅助模型，Qwen 保持固定 |
| [Late Chunking](https://arxiv.org/abs/2409.04701) | 整段上下文先过 embedding Transformer，随后按 chunk 做 mean pooling，保留跨片段语境；当前未实现 | **条件候选**：片段缺少上文指代或章节关系时。可免额外训练，但需长文编码和重建索引；当前 BGE CLS 向量及 query LoRA 不能直接与新 pooling 混用。只改检索表示，执行仍映射原文固定窗口 |
| [LongRAG（Jiang 等）](https://arxiv.org/abs/2406.15319) | 较长资料单元检索后交给 reader。当前 canonical 窗口不是论文的长检索单元 | **优先局部候选：粗检索→固定阅读窗口**，用于跨段条款、表头或相邻关系漏召回。需粗索引与窗口映射；直接扩大 READ 会增加预填充、输出和最终窗口压力，不作为默认替换 |
| [RAPTOR](https://arxiv.org/abs/2401.18059) | 递归聚类摘要并检索多个抽象层；当前没有摘要树 | **较后候选**：跨章节/全局问题持续漏召回时，可用摘要导航到原文叶子。离线建树和版本更新要支付 embedding、聚类、摘要成本；摘要节点不替代事实来源，不跨版本混树 |
| [LongMemEval](https://arxiv.org/abs/2410.10813) 的记忆设计 | 研究记忆粒度、索引键、时间查询及结构化阅读。当前有原 QA 引用、任务状态与版本目录，没有复现其记忆系统 | **局部参考**：从整轮历史检索改为有关联的片段及多个索引键，适合任务回访召回不足时。额外抽取/索引计费；历史摘要只导航，条件更新仍引用实际消息。其个人聊天任务不直接替换本项目文档问答队列 |

版本与有效期继续由显式目录及条件关系确定。`support_model.query_view` 已传 Need scope、条件和完整版本意图；最新文档、最相似文档和任务要求的适用版本是不同概念。查询改写或摘要不能替代该语义层。

## B：共享阅读、结论、动作与停止

| 方法与原始来源 | 可复用机制和当前状态 | 成本、适用条件与判断 |
|---|---|---|
| [多查询优化](https://drum.lib.umd.edu/items/dc467374-f73b-42c6-8298-8a4a745191e1) | 多个查询共用中间结果。`reading.py` 已按来源/条件/前驱/范围分组共享数值事实，再按各 Need 构造运算视图；属于机制借鉴，未复现数据库优化器 | **保留主线**。共享应减少重复提取，但按 Need 选择仍有费用；评价全部调用和输出 token。共享键不能仅按同一文档，否则不同群体与版本会混合 |
| [PAL](https://proceedings.mlr.press/v202/gao23f.html)／[PoT](https://arxiv.org/abs/2211.12588) | 将语义理解与计算分开。`reduction.py` 已做去重、量纲转换和 count/sum/group；不是模型生成任意程序 | **保留主线**。CPU 运算便宜，主要费用在提取和选择；现有类型化算子已覆盖目标，不新增程序生成服务 |
| [数据溯源：Why and Where](https://www.research.ed.ac.uk/en/publications/why-and-where-a-characterization-of-data-provenance/) | 区分来源位置与推导依据。当前 SourceSpan＋Claim DAG 表达原文坐标和父事实，有限归约保留证明路径；未复现其数据库 provenance 演算 | **保留主线**。短 ID、共享条件/版本表可以压缩表示，事实语义及必要来源仍应进入实际 ANSWER。先分别计量结论文字和来源元数据，定位窗口真正负担 |
| [RECOMP](https://arxiv.org/abs/2310.04408) | 抽取/生成式压缩服务下游回答，可不给无益材料输出。当前 READ/REDUCE 已进行任务条件事实提取和有限归约 | **局部借鉴已有**。若正确根结论的自然语言仍占满最终窗口，可借用终端质量监督改进压缩；新增独立压缩器需训练或推断。来源元数据占主导时，通用摘要器并不能解决问题 |
| [LLMLingua-2](https://arxiv.org/abs/2403.12968)／[LongLLMLingua](https://arxiv.org/abs/2310.06839) | 前者用蒸馏后的 token 分类器压缩，后者强调问题相关信息密度与位置。当前未实现 | **较后候选**：仅当 ANSWER 辅助摘录成为主要负担时局部比较。额外压缩模型有推断和驻留费用；对 READ 原文按问题删 token 会改变前缀与完整阅读语义，不能据压缩率直接推断净收益 |
| [IRCoT](https://aclanthology.org/2023.acl-long.557/) | 推理与检索交替。当前 OpenNeed 保存桥接实体、缺失关系和父链，优先使用就绪来源，必要时接一次 GAP/终端 READ | **保留有界适配**。固定 Qwen 可执行，无需 reader 训练；更长循环增加检索和生成。仅当有界后续实际截断必要链路时扩展，不按“问题复杂”无条件加调用 |
| [Adaptive-RAG](https://arxiv.org/abs/2403.14403)／[FLARE](https://aclanthology.org/2023.emnlp-main.495/) | 前者按复杂度选择检索深度；后者预测下一句，用低置信 token 触发检索与重生成。当前动作收益基于已观察状态，OpenNeed 负责缺口，无这两种路由器 | **借鉴监督与触发粒度，暂不叠加模型**。初始路由只有能省掉大量无益首轮准备才有独立价值；逐句 FLARE 需概率信息和额外解码，与当前一次最终 ANSWER 的形态不同 |
| [Self-RAG](https://selfrag.github.io/) | 训练 reader 输出 reflection tokens，控制检索、支持和效用。当前外围支持/审核字段不等同于 Self-RAG | **整法不采用**：要求 reader 训练，与固定 Qwen 条件冲突。可参考相关性、支持性和效用的区分，复用当前审核而非再增加 critic 或反思解码 |
| [计算价值](https://www2.eecs.berkeley.edu/Pubs/TechRpts/1988/5755.html)／[Rollout](https://web.mit.edu/dimitrib/www/dp_rollout_book.html) | 将继续计算视为有成本的动作；当前离线首动作＋固定后续得到 ΔQ，线上一次动作后重观察 | **保留主线**。线上不实际展开所有分支，离线配对和评分费用完整记录。预测增益与成本属于近似量，现有步骤执行预算继续约束实际支出，不继承精确 rollout 改进保证 |
| 岭回归／[XGBoost](https://arxiv.org/abs/1603.02754) | 当前 B 使用岭回归质量和绝对成本头；公共状态的线性项会在动作差中抵消，非线性或显式动作×状态交互可表达条件化排序，尚未接入 | **条件候选**：当前协议的配对标签显示错误排序时，先检验少量明确交互，再决定是否比较小树模型。保持同一前动作特征与会话分区，以实际动作选择结果判断；本轮只统一执行与监督协议 |

## C：精确前缀、保留对象、容量与期限

| 方法与原始来源 | 可复用机制和当前状态 | 成本、适用条件与判断 |
|---|---|---|
| [RadixAttention](https://arxiv.org/abs/2312.07104)／[HiCache](https://docs.sglang.io/docs/advanced_features/hicache_design) | 已有原生因果前缀和 GPU/Host 迁移；项目通过独立软 owner 保留来源 | **保留主干**。先区分原生已有命中、额外保护得到的命中、Host 回载和重算，再决定保留；最新上游能力不能直接视为固定 vendor 已有 |
| [RAGCache／PGDSF](https://arxiv.org/abs/2404.12457) | 前缀知识树及成本、频率、大小驱动的分层替换。当前有路径、成本、访问与容量信息，未实现 PGDSF | **优先局部候选**：经验模型覆盖不足且存在重访时，用 PGDSF 风格排序作为无训练 owner 对照，仍复用原生缓存。它不直接预测相对 Native 的新增收益，须单独报告实际净效果；不另建一套淘汰器 |
| [ContextPilot](https://arxiv.org/abs/2511.03475)、[作者实现](https://github.com/EfficientContext/ContextPilot) | 上下文重叠索引、顺序对齐与去重注释。`SourceOrderIndex` 已接作者 CPU index/reorder，`RequestCompiler` 在真实 READ/ANSWER 编译中消费；没有采用作者 prompt pipeline 或 hook | **已接线、待净收益实测**：原排列和作者对齐排列在选择前取得 Native 报价。无需训练但要 CPU 索引与同步；新顺序生成新的因果路径，不能直接拼接旧 KV。独立窗口顺序已稳定时机会可能小 |
| [Adaptive TTL](https://arxiv.org/abs/1704.04448)／[hazard–TTL](https://aferragu.github.io/assets/pdf/sigmetrics16.pdf) | 用访问过程或重访风险估计期限。当前是粗特征、配对净成本与 TTL15/60/180，没有生存模型 | **条件候选**：粗桶被重访间隔差异明显误导时，用分组经验生存曲线替换重访概率部分，保留三个期限和 Native 净成本目标。无须神经网络；按共同观察窗处理右删失，同窗三个 TTL 不能重复计作三个访问样本 |
| 预算覆盖、局部交换与精确小规模求解 | 当前 `allocate_retention` 在一个逻辑保护容量下比较价值排序和容量密度两次贪心；同一需求收益取 max、共享页按并集、每 owner 只选一期限；Native 管理物理 GPU/Host 容量 | **已有可执行近似方案**。只有发现明显容量碎片或组合遗漏时，再比较一次局部交换；小规模枚举可作离线选择上限。当前目标含非单调项与多 owner 外部性，不具有标准最大覆盖保证，完整集合效果由实际配对评价 |
| [CacheBlend](https://arxiv.org/abs/2405.16444)／[Cache-Craft](https://arxiv.org/abs/2502.15734) | 选择性重算以复用非前缀块，后者还管理上下文化 chunk 缓存与重算。当前均未实现 | **独立后续路线**：非前缀限制挡住大量有效重访时才值得考虑。无需微调 Qwen，但会改变 KV 执行过程，需要模型内部统计、重算与搬运计量；位置修正不能消除不同前文的因果差异 |
| [LMCache](https://arxiv.org/abs/2510.09665)、[官方实现](https://github.com/LMCache/LMCache) | KV 存取与传输系统，可集成不同后端；当前未接入，已有 HiCache 双层能力 | **条件候选**：出现跨进程/节点共享或现有后端不覆盖的需要时。普通 KV 存取可保持精确语义，CacheBlend 是另外的可选能力；当前叠加两个管理层会增加连接器和搬运成本 |
| [H2O](https://arxiv.org/abs/2306.14048)／[SnapKV](https://arxiv.org/abs/2404.14469)／[KIVI](https://arxiv.org/abs/2402.02750) | 按重要性保留 token KV，或对 KV 低位量化。当前未实现 | **本阶段不采用**。权重可冻结，但会改变完整 KV 或数值精度，已超出现有固定推理条件；需要独立 attention/kernel 与质量比较，不能并入 Native/TTL 期限对照后仍声称仅改变保留 |

## 评价、充分性与错误归因

| 方法与原始来源 | 可复用内容 | 本项目判断 |
|---|---|---|
| [Sufficient Context](https://arxiv.org/abs/2411.06037) | 区分上下文缺少回答信息和模型没有用好充分信息 | **优先局部候选**：将“候选充分→已读事实充分→最终可见结论充分”分开标注，复用 `evidence_diagnosis`、真实 traces 与 `reader_data`。需要离线核验；充分性可作诊断/特征，不能代替实际最终 ΔQ，也不自动成为硬停止条件 |
| [RAGChecker](https://arxiv.org/abs/2408.08067) | claim 级蕴含诊断，区分检索和生成问题 | **借鉴归因维度**。当前已有 Claim DAG 与阶段审核，可复用实际事实减少重复提取；尚未集成其整套评分。额外 judge 成本属于离线费用，不替代登记的官方分数 |
| [ALCE](https://aclanthology.org/2023.emnlp-main.398/) | 分开评价答案正确性及引用支持质量 | **借鉴来源核验**。引用存在不等于引用支持结论；当前来源链给核验材料，自动语义核验仍有成本。优先核对实际 ANSWER 的结论与引用，不另造多套验收分数 |
| [ARES](https://aclanthology.org/2024.naacl-long.20/) | 合成数据训练轻量 judge，并用人工标注做 prediction-powered inference | **较后候选**：只有反复评分成本突出且具备独立人工标注时考虑。当前未实现 PPI；少量自动评分或普通 bootstrap 不能冒称其校正方法，正式作者指标继续保留 |
| [Lost in the Middle](https://aclanthology.org/2024.tacl-1.9/) | 证据位置会影响长上下文利用，是布局的评价依据 | **用于解释现有布局对照**。固定来源前缀有缓存价值，但提取质量仍要测；最终结论的位置可与原文布局分别组织。无需因此增加一组长度/位置扫描实验 |

## 收敛后的采用顺序

1. **直接复用现有能力**：显式条件/版本、BM25+dense RRF、支持覆盖/偏好、共享提取与 CPU 算子、有限 Claim DAG、有界收益、原生前缀保留。是否切换配置由对应错误决定，不同时启用所有选项。
2. **优先局部候选**：候选冗余→评分前 MMR；跨段关系漏召回→粗检索映射固定窗口；错误环节不清→上下文充分性分层；C 经验模型缺覆盖→无训练成本/频率 owner 对照。这些分别接现有池、目录、审核和分配入口。
3. **条件成熟再扩展**：查询改写训练、late interaction、Late Chunking、RAPTOR、轻量非线性回归、经验生存模型。每项都需可定位的独立瓶颈，不按方法新旧排序。
4. **另设执行路线**：CacheBlend/Cache-Craft、KV 剪枝/量化、跨节点缓存。它们改变的条件较多，本阶段不混入冻结 A 与 Native/TTL 主比较。

只比较命中当前瓶颈的候选与现有方案；候选资料充分性、最终答案、完整响应和全部已付成本共同决定采用。方法清单不自动扩大正式队列、训练对象或参数网格。

## 保留的可选对照

ContextPilot 的作者索引已是主四格的编译因素；TTL/value 仍为独立可选保留策略。以下不改变该因素定义。

- [ContextPilot](https://arxiv.org/abs/2511.03475)：`adapters/contextpilot.py` 复用作者低层索引及来源排列；旧会话内重叠对照入口保留，不等同于当前作者索引。
- [EfficientAgent 的 Host 准入适配](third_party/EFFICIENTAGENT_NOTICE.md)：原生补丁 `HostAdmissionPolicy` 默认关闭；显式启用时配置进入 cache scope，线上与配对重放均传真实业务会话身份。它是 Host 写入对照，未替代 C 的 TTL 分配。
- [CRAFT 行文本函数](third_party/CRAFT_NOTICE.md)与[表格关系适配](research/table_relations.md)：仅在表格注释选项启用时使用，未引入上游完整检索系统。

当前复用核对：BGE-M3 作者说明推荐混合召回后接 reranker；reranker 输出 query–passage 相关性分数，本项目的支持关系与集合效用是额外监督任务，不能把相关性直接当成事实支持。HiCache 官方设计以共享 token 前缀为复用条件，并区分 GPU/Host/存储层；当前集成复用容器 CUDA 依赖和明确适用的 SGLang 源码补丁；实际包版本、源码与补丁 hash 进入记录，不以旧包版本阻止启动。来源见 [BGE-M3 模型卡](https://huggingface.co/BAAI/bge-m3)、[reranker 模型卡](https://huggingface.co/BAAI/bge-reranker-v2-m3)、[HiCache 设计](https://docs.sglang.io/docs/advanced_features/hicache_design)。

## 成熟外部参照的工程复用

[LlamaIndex CondenseQuestionChatEngine](https://github.com/run-llama/llama_index/blob/v0.14.25/llama-index-core/llama_index/core/chat_engine/condense_question.py) 和 [CompactAndRefine](https://github.com/run-llama/llama_index/blob/v0.14.25/llama-index-core/llama_index/core/response_synthesizers/compact_and_refine.py) 由 `adapters/llamaindex_dialogue.py` 直接调用。适配层使用上游 Memory、问题凝练和 compact/refine，所有 LLM 调用交给项目回调；实际框架版本记录。`execution.dialogue_mode: llamaindex` 已接入统一 Controller/SessionStore；正式入口声明 `LlamaIndex_native`，为真实历史 condense＋hybrid/rerank 检索＋compact/refine＋Native 编译/保留。各数据集物化 `hybrid_rerank`、`query_mode: current` 配置，关闭项目 query adapter/selector，使用发布版 BGE reranker；不调用项目任务/支持规划器。所有框架 LLM 调用仍走实际固定 Qwen 引擎与账本。该代码接线不等于已完成外部比较，质量、完整响应与全部费用留待当前环境实际执行。

最终评价在 c1/c4 均以固定/收益 B × Native/ContextPilot 编译构成主四格，统一使用 Native retention；`S3_contextpilot_no_fact_reuse` 与联合格配对检验跨轮事实复用。外部参照、事实复用和编译四格分别回答不同问题，不将历史 E0 或可选 TTL/value 替代其中任何一项。
