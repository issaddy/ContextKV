# 2026-10-24 独立研究审查：模块 B 的近邻、复用与下一步证据

本文保留审查时的源码事实、实验状态和阶段建议。文中“收窄”“暂不扩展”等表述不作为当前研究范围上限；现行路线以[贡献与复用](research_contributions.md)及[工程目标](implementation_plan.md)为准：A、B、C 都可以复用、优化和扩展，按目标场景中的质量、完整成本和资源效果决定采用方式。对已有方法准确归属，约束的是成果表述，不能替代继续改进。

本次审查读取了 [research_screening.md](research_screening.md)、[related_work.md](related_work.md)、[research_contributions.md](research_contributions.md)，再核查作者论文、论文直接链接的仓库及固定源码。检索截止 2026-10-24，围绕 cache-aware compaction、append/rebuild、re-prefill、transfer-aware planner，并沿新论文的参考文献补查。它是针对贡献边界的定向审查，不是穷尽性综述，也不能证明“首次”。文献审查阶段仅新增本文，未修改核心实现、运行上游系统或采集新 GPU 数据。

**判断：现有 B 值得继续验证，但应按受约束的系统方法贡献来表述。** 前缀稳定、压缩与缓存的成本权衡、转移感知计划、Prefill/Decode 分项建模、上下文版本与缓存生命周期协同，均已有直接近邻。新增检索尤其发现了现有三份文档未纳入的 **AKV、SmoothAgent、Context Language Models 的 Suffix Cache Reuse**。本项目可以检验的剩余主张是：在完整原始 QA、同一 required 证据和原生精确前缀语义下，利用真实 GPU/host 驻留选择两个完整文本动作，能否在明确负载区域超过调优简单策略。这个区别本身不等于已证明有效或足以构成强新颖性。

## 1. 最影响主张的论文核查

下表“论文事实”只描述原文；“本课题判断”是本审查的推断。论文中的收益均是作者在其模型、数据和系统上的报告，未经本项目复现。

| 工作与核查入口 | 论文事实 | 本课题判断与复用边界 |
|---|---|---|
| [CacheSmith，v1](https://arxiv.org/html/2610.18925v1)，§5–6 | WarmUp/Probe/Patch 选择性重算；离线计算时间、运行时有效带宽和队列剩余预算共同决定计划，漂移时重计划；目标是质量门控的 **TTFT** SLO。 | **B 的直接近邻。** 不可声称首个转移感知、标定驱动或轻量在线 KV 计划器。可复用成本边界、控制开销计入和强静态对照的评价思想；本项目动作是完整 APPEND/REBUILD，目标还包括生成完成时间。此次未确认作者关联的公开实现，不称代码复用。 |
| [SmoothAgent，v1](https://arxiv.org/html/2607.00151v1)，§3–5；[作者仓库](https://github.com/PanZaifeng/SmoothAgent) | 提前执行可分段的上下文变换和新 KV 准备，在 commit 点切换；后台请求与前台请求由感知时延的调度器共存。性能模型含标定 GEMM 插值、随 KV 长度增长的 Decode 注意力，以及 `q·prefix + q(q+1)/2` 的 Prefill 注意力工作量。 | **现有综述的重要遗漏。** “上下文长会增加 Decode 成本”“使用注意力工作量解析式”“变换准备与提交分离”不是本项目独创。其决策重点是如何提前执行给定变换；本项目重点是本轮两动作是否值得切换。应引用其建模依据，暂不增加后台队列、抢占或预执行第三分支。 |
| [CAPC，v1](https://arxiv.org/html/2607.15516v1)，§3、§7、Appendix A | 建立 API 缓存计费与压缩交叉模型；AdaptiveCacheBoundary 按历史变化率划分区域。主要缓存层次证据来自 Sonnet API，作者讨论了内容相关质量下降和跨提供商泛化限制。 | **直接重叠成本权衡的动机。** API 计费阈值、TTL 和压缩率不能当作 SGLang GPU/host 时延参数。复用“先检查质量，再比较成本区域”的实验结构即可；无需复制压缩器、句子变化追踪或再加一套内容哈希。 |
| [ContextPilot，v4](https://arxiv.org/html/2511.03475v4)，§4–5；[作者仓库](https://github.com/EfficientContext/ContextPilot) | 上下文索引、对齐、去重和顺序注释促进跨请求复用，包含超出单会话局部排序的组织与调度能力。 | **A 与 B4 的主要来源。** 当前 B4 是受同一证据约束的方法适配，不能标为完整官方 ContextPilot。B 必须在这个独立布局对照和强简单策略之上仍有增量，不能把布局收益算作成本决策收益。 |
| [TokenPilot，v1](https://arxiv.org/html/2606.17016v1)，§2–3；[论文链接的仓库](https://github.com/zjunlp/LightMem2)现重定向 LightRSI | 全局稳定前缀/入口缩减，局部按残余效用及批量轮次清理；优化公式使用 cache-hit/cache-miss token 的折扣成本。 | **A/C 设计原则及 B 简单对照重叠。** 稳定前缀和保守批量门控应归为复用。其原始内容缩减及额外状态估计不等于本项目完整 QA 约束；固定周期只是局部思想对照，不能代表其全部系统。 |
| [Cache Engineering for Agents / AKV，v1，10 月 20 日](https://arxiv.org/html/2610.28577v1)，§3–4；[作者仓库](https://github.com/thu-nics/AKV) | 通过 `drop` 与 `repos` 改变活动 KV；相同文本可能因操作历史不同形成不同状态。扩展 radix 匹配、drop-aware 淘汰和 segment-batch attention 恢复支持这些语义；作者报告了实际服务吞吐和尾延迟结果。 | **最新且很接近 A/C 的系统近邻，也削弱 B 的宽泛动机主张。** 不能泛称首次把上下文语义和 KV 生命周期协同。它不按完整当前文本重算全部存活后缀，不能直接嵌入本项目的精确前缀路径。应复用“共享状态隔离、恢复后状态一致、真实物理回收”的验证问题。 |
| [Context Language Models / SCR，v1](https://arxiv.org/html/2609.37725v1)，§4.3、Appendix B；[作者实现说明](https://github.com/facebookresearch/context-language-models/blob/483198229ab7b90c0eed4651306ab3b3505aaf50/suffix_cache_reuse/README.md) | 模型编辑上下文；SCR 复用编辑后仍存活的后缀 KV，明确保留包含旧前缀信息的状态，重定位 key，并把移动项放进 session-private slot。 | **另一项近期强路线，不能遗漏后声称重建必然全后缀重算。** “精确原生前缀路线”才有该边界。它改变状态语义，不是当前可无缝替换的优化。可借鉴按来源拆解实际复用和 session 隔离测试，不移植 suffix splice。 |

还筛查了 [Budgeted Cache Repair](https://arxiv.org/html/2610.02233v1) 和 [CacheScout](https://arxiv.org/html/2608.14624v1)。前者研究跨上下文近似 KV 的修复，Appendix C 明确区分质量配置与计时配置，并指出部分 TTFT 优势没有体现为端到端收益；它提供的是评价警示，不能用其“整次调用分配无增益”否定本项目精确文本动作。后者学习多 agent 的执行转移来决定保护与预取，关注固定 agent 前缀的未来复用，不是本轮两个文本布局的选择。两者本轮不扩为新实现任务。

## 2. 作者源码可复用到哪一层

以下 commit 是 2026-10-24 实际读取的仓库快照；“有源码”与“已在本项目执行”分开记录。

| 仓库快照 | 实际核查到的入口 | 本轮结论 |
|---|---|---|
| ContextPilot `b4c80327f6cfae2286839413731fcb6f53578b2f`，Apache-2.0 | [`live_index.py`](https://github.com/EfficientContext/ContextPilot/blob/b4c80327f6cfae2286839413731fcb6f53578b2f/contextpilot/server/live_index.py) 的 `reorder` 与 `deduplicate` 均更新 `seen_docs` 和轮次 | 现有只读候选适配有实际必要性；不能为了“直接复用”让两个未执行候选污染索引。继续保留短实现与固定来源，无需引入整个层次索引。 |
| LightRSI `9f0f19308a30445438779efbf3108a7d965a55c7`，MIT | [`lifecycle-planner.ts`](https://github.com/zjunlp/LightRSI/blob/9f0f19308a30445438779efbf3108a7d965a55c7/components/packages/features/eviction/src/lifecycle-planner.ts) 返回版本相关计划，比较 estimator 的 baseVersion 与 registry.version | 计划/执行分离、批量触发和版本检查是已有工程模式；无需移植其运行时、模型估计器及全部旁路分支。 |
| SmoothAgent `7c2b7743e3c3d3e699eca78e244fac02234a5489`，Apache-2.0 | [`README`](https://github.com/PanZaifeng/SmoothAgent/blob/7c2b7743e3c3d3e699eca78e244fac02234a5489/README.md)、[`manifest`](https://github.com/PanZaifeng/SmoothAgent/blob/7c2b7743e3c3d3e699eca78e244fac02234a5489/third_party/manifest.toml)、[`SGLang patch`](https://github.com/PanZaifeng/SmoothAgent/blob/7c2b7743e3c3d3e699eca78e244fac02234a5489/patches/sglang.patch)；有固定上游 revision、标定参数及调度改动 | 可引用可分解成本模型和固定补丁复现方法。其后端固定版本不同，后台执行改变资源竞争，不能把整块 patch 当作低风险小优化合入。本文没有运行该补丁。 |
| AKV `c567a429a058f1abce3b2879e69978b09e0fda4c`，Apache-2.0 | [`system README`](https://github.com/thu-nics/AKV/blob/c567a429a058f1abce3b2879e69978b09e0fda4c/akv/system/README.md) 写明 Drop/Repos 只支持 page size 1；Context 请求当前排除 HiCache 和 external/session cache | 与本项目 page size 64、HiCache/session 实测路径存在明确接口与配置差异。先采用论文的机制验证问题，不做移植或声称可直接作为同配置替换。 |
| CLM `483198229ab7b90c0eed4651306ab3b3505aaf50` | [`SCR README`](https://github.com/facebookresearch/context-language-models/blob/483198229ab7b90c0eed4651306ab3b3505aaf50/suffix_cache_reuse/README.md) 同时列出 SGLang/vLLM 实现；根 [`LICENSE`](https://github.com/facebookresearch/context-language-models/blob/483198229ab7b90c0eed4651306ab3b3505aaf50/LICENSE) 为 CC BY-NC 4.0 | 记录为有作者实现的相关路线；本轮不复制代码，不能按 Apache/MIT 依赖对待。它的历史状态复用也不满足当前原生精确前缀合同。 |

CacheSmith/CAPC 在本次论文与定向仓库检索中未找到可确认的作者代码入口；这不等于断言代码不存在。未确认之前只复用公开方法与实验设计，不把同名仓库归给作者。

## 3. 应保留与应收窄的贡献表述

建议将 B 表述为以下**待验证假设**：

> 对保持完整原始 QA 和同一份必需证据的多轮文档问答，在原生精确前缀与 host 恢复机制之上，真实驻留报价及完整响应成本能在部分容量、并发和输出长度区域，改进 APPEND/REBUILD 的选择，并超过开发集同预算选出的简单策略。

这个表述保留了动作、约束、观测和比较对象，也允许结果为负。不要追加“首次”“普遍更快”“达到最优”或“质量不变”。两候选都保留材料，不代表顺序、距离和引用变化不会影响答案；精确前缀指缓存计算语义，不代表 APPEND/REBUILD 的输出分布相同。

具体需要收窄三处：

1. **成本模型不是独立算法贡献。** 现有可选正系数基函数中的 `U·P + U²/2` 与 SmoothAgent 的注意力工作量同属标准因果注意力计数建模。可选拟合的价值是本机数据适配、清楚的支持域和可复现验证。当前 `first_token` 是含恢复/排队的联合墙时，不能直接拿上游 kernel/batch 系数代替它，也不能把离线 MAE 改善写成在线策略收益。
2. **生命周期协同需要具体事务边界和效果。** TokenPilot、SmoothAgent 和 AKV 已有不同层次的状态/生命周期处理；本项目可以验证的增量是业务答案持久化成功与旧 epoch 软引用退役的绑定，及其对另一会话实际恢复/重算的影响。CAS、outbox、哈希和版本检查本身都是工程手段。
3. **对照名称与能力对应。** B4 写“ContextPilot 思想适配”，周期/冷热策略标明其局部机制；不要用它们代表完整上游系统。约束内的主表应以原生 B0/B1 和强 B6 为核心。AKV/SCR 等异路线无需成为本轮完成前置条件，但论文若主张普遍优于上下文管理或缓存工程，就必须扩大端到端、同质量预算比较，不能用约束差异回避该广泛主张。

完整 QA 还意味着重建只能消除可移除的材料冗余，不能保证无限轮次下上下文有界。这应作为适用范围说明，而不是新增压缩模型来隐性改变技术方案。

## 4. 依据当前证据，只优先补三组工作

本地已存在真实 H200 集成和标定数据；这里依据 [verification.md](verification.md) 与 [阶段/成对诊断](data_and_evaluation.md#标定误差按阶段与特征单元解释) 的当前记录，不把工程成功等同于论文有效。

| 优先级 | 当前具体不足 | 最小下一步及停止扩展边界 |
|---|---|---|
| P0：B 是否真的改善选择 | 原标定 first-token holdout 支持为 36/84，supported-only MAE 约 55.56 ms；Decode TPOT 误差较小，混合平均会掩盖此问题。现有 9 对来自 3 个早期场景，7 次滞回、2 次回退，不能证明选择器的增量。 | 固定已有默认模型、参数与比较协议，按会话年龄、短/长上下文、冷/GPU/host 分层选取独立 dev 场景，保留两臂独立初态。报告支持/回退/真正由成本触发切换的数量、成本差误差及 regret，再决定是否扩大标定。暂不加入长期预测或新调度器。 |
| P0：强基线与回答质量 | 512 上限的 Cloud P2 有 21/41 截断；1024 敏感性为 0/41，但缓存初态不同，不能据此比较时延。现有 P3 BM25 检索 recall@5 约 0.2094，仅覆盖 39/41 有 qrels 轮次；正式回答非劣仍未验证。 | 所有方法统一输出上限、材料、缓存准备和实际资源预算；完成现有 judge/盲审闭环与同预算开发集选优后冻结 test，B6 包含原生 host 写策略的合理候选。P2 隔离决策，P3 检查低召回下的完整效果；检索误差与策略误差分开，先不更换检索栈或数据集。 |
| P1：C 的实际机制与总体效果 | 新的 GPU/host 受控退役探针已经观察到同压力下旧/新路径保留变化，退役瞬间 pool 变化为 0；这解决部分机制证据，但尚未证明一般工作负载吞吐提升。 | 复用已有 fixture 与固定动作轨迹，再做少量有限容量、多会话负载对照。按同一 frozen Ours 配置仅改 C 开关，连接退役、共享/独有路径、真实 Prefill/transfer 与完成时间。保持低压力负例，不先开发全图追踪或新的淘汰算法。 |

输出长度是第一组实验中需要单列的解释变量。现有冷态 pair 的 APPEND/REBUILD 产生 422/363 tokens：首 token 差仅约 1.14 ms，Decode 差却约 273.91 ms，两边 TPOT 近似相同；共享长度先验因此遗漏了主要差值。这是**项目已观测的局限**，不是上述论文证明的结果。当前 3 个场景不能用于拟合候选条件先验；只有独立 dev 数据确认稳定关系后才值得尝试，并须与“继续使用共享先验”的简单版本公平比较。真实当前输出长度始终只能离线诊断，不能进入当轮在线决策。

可选 positive estimator 已改善若干离线分割上的点预测，但其经验 OOF 误差门槛较大；继续保留 lookup 默认是合理的。下一轮验收应看它是否减少实际错误动作及损失，而不是再增加一种拟合器来追逐这批数据的 MAE。

## 5. 文档状态需要一次性对齐

现有材料有若干时间上已过期的句子，后续统一整理时应一起改，避免形成互相冲突的完成度描述：

- `related_work.md` 开头和 `data_and_evaluation.md` 开头仍写尚无 GPU 验证；`verification.md` 已记录 H200 实测。应改成“已有限定配置的集成/诊断验证，正式质量与优于强基线的结论仍未建立”。
- `research_screening.md` 与 `research_contributions.md` 部分条目仍把成对入口写为尚待真实执行；现已有 9 对诊断，但覆盖范围不足，应该更新为这一准确状态。
- C 的旧独有分支归因已出现受控 fixture 证据；“完全没有路径归因”不再准确，“尚无真实多会话总体性能结论”仍准确。
- 三份综述应补入 SmoothAgent、AKV 和 SCR，且把本次审查的定向检索日期保留。SmoothAgent 的论文页含 PVLDB 模板信息，本次只按已核查的 arXiv v1 引用，不据占位卷期推断正式接收状态。

本轮研究没有给出必须新建的模块。最有价值的复用是现有代码中的真实原生执行、短小的 ContextPilot 适配、成熟离线拟合与已具备的审计/实验工具；接下来应让这些工具产出足以证伪 B 的强对照证据，再决定是否保留、简化或扩展机制。

## 附记：原数据 v2 工件复算

随后按独立任务，使用修正总输入长度支持域后的代码复算原有 1500 rows，生成 `runs/remote-h200-cost/cost-dev-v2.json`、`cost-dev-positive-v2.json` 和 `offline-model-comparison-v2.json`；旧工件与原始数据保持不变，输出先验仍为 433/512，没有加入新增长度采样或成对答案。

27 个 fold/model 组合的点预测全部不变。唯一覆盖变化为 lookup 的 `leave_length_2048`：150 条观测中原先 90 条错误地被视为有支持，修正后为 0，因为该 fold 的训练输入长度从 4096 开始。原 holdout 首 token 仍为 36/84，positive 系数、OOF P95/P99（151.17/269.53 ms）和 OOF lookup 覆盖率 19/30 均不变。新 positive 工件显式记录 `fit_sources=["hardware"]`。10000 与 11217 总输入均被两模型标为不支持，2048/8192 的已测例仍支持；无 site-packages 的加载/推理也通过。核查结果写入 `runs/remote-h200-cost/v2-recomputation-check.json`。这次变化纠正了校准声明，没有改善预测数值或证明方法收益。
