# 硕士课题贡献、复用依据与验收证据

本课题的主线是：**在相同证据与回答质量约束下，让上下文追加/重建决策与真实 KV 驻留、恢复成本和版本生命周期协同，改善有限缓存、多会话文档问答的完整响应时间。** 研究对象是一次完整问答及其对其他会话缓存的影响，不把减少输入 token、增加理论前缀或释放 session 引用单独等同于系统收益。

三个贡献分工为：A 提供可比较、质量受约束的上下文候选；B 是主要方法，利用真实状态和标定成本选择候选；C 将选中版本安全提交并落实为缓存生命周期变化。原生 RAG、BM25、RadixCache、HiCache、session 软引用、SQLite 事务和一般插值都属于复用的基础能力，不作为本项目独立创新。下述为**待实验支持的贡献主张**，代码实现和测试通过不代替性能或质量结论。

论文宜表述为**一个主要方法贡献 B，加上两个支撑它的适配/系统贡献 A、C**，而不是三个首创算法。对已检索工作的比较只能支持具体差异，不能证明不存在其他相同方案；本项目不使用“首次统一缓存与压缩”“首次按成本管理上下文”这类宽泛首创主张。

最新[独立研究审查](research_review_20261024.md)进一步纳入 SmoothAgent、AKV 与 SCR：注意力工作量解析式、Prefill/Decode 成本分解、上下文变换的准备/提交和缓存状态生命周期均有直接近邻。B 的增量需要落到完整 QA / 同一 required 证据 / 原生精确前缀约束下的实测选择效果；C 的增量需要落到答案持久化后退役的具体事务边界及其实际缓存影响。当前不将通用公式、正系数拟合、提示词修订或 CAS/outbox 单列为研究创新。

**当前实证约束：** 新的[90 对分层诊断](stratified_decision_validation.md)没有支持 B 优于固定重建或固定阈值。主要遗漏是候选间输出长度差，同时无条件经验误差量过于保守。已据此补查[长度预测方法与独立验证设计](output_length_research.md)，尚未训练或引入新预测器；不能继续仅凭 Prefill 点预测改善宣称主要贡献成立。

## 为什么选择这组基础工作

| 基础工作 | 已有能力与选型依据 | 本项目实际复用及边界 |
|---|---|---|
| SGLang | 同一引擎已提供精确前缀缓存、host tier、原生 admission 和 session 软引用，能保持底层计算语义不变研究上层决策。[原生 session 说明](https://docs.sglang.io/docs/advanced_features/session_radix_cache) | **直接复用引擎代码**；固定 `3f36fc0…`，新增只读报价、epoch 协议和测量接缝。选 SGLang 是为了在一个可核查运行时里比较布局、恢复与退役，不是断言它优于所有引擎 |
| ContextPilot | 直接研究上下文重叠、排序、去重和注释，问题与 A 最接近。其 RAG 实验涵盖多会话/多轮和 MTRAG；离线批量与在线冷启动实验设置有区别，不能混用结论。[论文 §7](https://arxiv.org/html/2511.03475v4#S7) | **方法原理适配**：复用固定源码的 overlap/position 度量、命中前缀对齐和引用注释思想；当前未接入完整层次索引、跨会话聚类或批调度 |
| TokenPilot / LightRSI | 将前缀稳定与保守批量清理结合，说明“删得更短”与“保留复用”需要共同考虑。论文使用 PinchBench/Claw-Eval、GPT-5.4-mini、API 缓存计数和货币成本，属于与本地 GPU 不同的测量体系。[论文 §4.1](https://arxiv.org/html/2606.17016v1#S4.SS1) | **设计思想复用**：稳定内容与动态内容分离、批量/门槛触发、计划后提交；不引入其模型估计器、整套 TypeScript 服务或训练压缩器 |
| EfficientAgent | 对并发 agent 的可复用工作集、host 容量和写入/恢复收益做系统分析，说明 host 有副本并不保证下次能复用。主实验是 OpenHands、SWE-bench、vLLM/LMCache 和多 GPU；本项目不继承其部署结果。[论文 §5.1](https://arxiv.org/html/2609.33762v1#S5.SS1) | **分析与实验设计复用**：关注真实恢复量、工作集压力、host 写入代价；当前没有迁移其 connector 或写入准入策略，不把这一扩展计入已完成贡献 |
| MTRAG + BM25S | MTRAG 含非独立追问、不可回答问题与多领域完整会话，适合同时检验上下文行为和回答质量。[官方论文](https://arxiv.org/html/2501.03468v1) | **直接使用固定数据与检索库**，在线白名单与 P1/P2/P3 协议为本项目适配；检索变化与缓存策略收益分开评价 |

这组选择围绕一条因果链展开：**材料怎样排列 → 实际可用前缀是什么 → 恢复/计算/生成分别要花多少 → 旧版本何时退出保护 → 下轮及其他会话是否受益。** 每项基础工作对应一个明确环节，而不是把多个系统叠加后笼统比较总吞吐。ReCAP 的持久注意力图不在当前范围；其 warm/idle 启发只形成明确阈值的策略对照，标为适配版。

论文结果也不意味着任意质量指标都不降：例如 ContextPilot 的 QASPER 部分模型结果、TokenPilot 的部分任务质量结果并非全面优于未优化方法。因此本项目需要自行做逐会话质量比较、条件/版本审计和非劣检验，不能只引用上游“保持质量”的概括性描述。详细固定版本、许可证与 API 副作用核查见[相关工作记录](related_work.md)。

## 候选方法筛选：主动核查后的纳入与取舍

筛选标准为：能否解决当前可观察瓶颈、是否保持完整证据及因果前缀语义、有无可核查实现入口、迁移是否改变底层计算/质量条件、能否在硕士课题规模内形成公平对照。文献中报告过收益只能说明值得核查，不足以决定直接集成。

| 候选 | 官方证据与成熟能力 | 可复用入口、纳入内容 | 当前取舍与触发后续扩展的证据 |
|---|---|---|---|
| RAGCache | [arXiv v2](https://arxiv.org/abs/2404.12457v2)、[作者 TOCS 版本](https://chaojin0310.github.io/files/papers/TOCS25-RAGCache.pdf)使用保留文档顺序的知识树、GPU/host 两层、prefix-aware GDSF 与检索/Prefill 推测流水线；其比较覆盖 TTFT、长输出 TBT 和容量消融 | 可核查入口是论文 §5.1 的前缀感知代价/大小/频率策略、§5.3 的检索流水线；纳入“相同文档不等于相同因果 KV”和分项成本/容量消融的依据。当前没有把 PGDSF 说成本项目已经实现的策略 | 它属于精确前缀路线，不能与下面的部分重算方法混为一类。本次在论文/作者主页未确认可维护的官方源码包，不安装同名 PyPI 包。先复用 SGLang 已有两层缓存；若实际数据表明淘汰策略或检索等待成为瓶颈，再单独移植策略或流水线，而非同时替换缓存树和检索执行器 |
| CacheBlend | [原论文 §4/§7](https://arxiv.org/html/2405.16444v1)通过选择性重算融合非前缀 chunk KV，并在多跳问答等任务检验质量与 TTFT；[LMCache 官方当前入口](https://docs.lmcache.ai/kv_cache_optimizations/cacheblend.html)提供 `lmcache server --engine-type blend` / `BlendModule` | 存在真实可运行入口，值得作为另一条研究路线；它不是“只按语义相似度直接拿来用”的简单近似缓存。可借鉴质量—重算比例联合评价；当前未调用该引擎，也未移植其融合/位置恢复逻辑 | 会改变本项目完整前缀恢复后的计算路径，需要 LMCache/引擎集成和选择性重算支持。先不塞入 SGLang 主实验。若排序后仍大量存在非前缀块重叠且 Prefill 主导，可建立独立实验分支；届时要锁定兼容版本、同质量阈值和同资源预算，不能把其论文 TTFT 结果当作本项目端到端优势 |
| Cache-Craft | [作者所属 Adobe 官方页面](https://research.adobe.com/publication/cache-craft-managing-chunk-caches-for-efficient-retrieval-augmented-generation/)、[论文 2502.15734 §3–§5](https://arxiv.org/html/2502.15734v1)核实为 chunk-cache 可复用性、局部修复与层次管理方法；论文实现修改 vLLM/Triton attention，实验包括 LLaMA-3 8B/70B、A100、Sys-X 及公开任务 | 可核查方法入口是 cache variant 选择、重算提前终止和逐层预载；可用于分析“非前缀复用需修复”的代价。论文/作者页没有提供本次可确认的官方源码入口，当前没有代码复用，不把同名 CacheCraft 语义缓存项目当成论文实现 | 涉及部分 Prefill、动态 attention mask、位置处理和多份 chunk KV，超出当前精确前缀/单 GPU/两层主线。只有非前缀重复是主要损耗且有可复现代码/足够质量预算时才值得扩展；当前不重写 kernel，也不声称已复现 |

相较之下，ContextPilot 的纯上下文排序/注释可以在保持原生 attention 的前提下接入；SGLang 的 GPU/host 和 session 能力可以直接复用；TokenPilot 的批量切换与 EfficientAgent 的压力分析可以在控制器/测量层验证增量。这个选择优先形成一条可部署、可消融的完整链路，并不说明 CacheBlend 或 Cache-Craft 效果较差。若实测表明精确前缀路线不能利用主要重叠，应承认适用范围，而不是通过挑选样本回避其他路线。

来源辨认也属于工程前置工作：ContextPilot 的作者、题名及官方仓库由 [MLSys 2026 proceedings](https://proceedings.mlsys.org/paper_files/paper/2026/hash/b0131b6ee02a00b03fc3320176fec8f5-Abstract-Conference.html)与 `EfficientContext/ContextPilot` 交叉确认；不将同名项目或 fork 的接口/许可证默认归给论文。上述筛选没有复制未确认许可的实现，也没有把三个系统的性能数字拼成项目收益预期。

### 补充检索发现的直接相近工作

2026-10-23 进一步检索“cache-aware compression / context compaction / append-rebuild cost”后，发现两项需要纳入新颖性边界的作者论文：

- [Cache-Aware Prompt Compression（CAPC），2607.15516v1](https://arxiv.org/html/2607.15516v1)已经建立缓存与压缩的成本交叉模型，并提供按历史变化率选择稳定/动态区域的 AdaptiveCacheBoundary。其直接目标是 API 计费和命中率，核心验证围绕 Sonnet API；文中 Appendix A 是可核查的算法入口。本项目因此不把“压缩可能破坏缓存”或“利用成本选策略”作为新发现。区别限于本地 GPU/host 的实际驻留与原生恢复、完整回答时延，以及共享证据约束的两种文本动作。未复用其压缩器、归一化或 API 专属阈值，也没有将 API 命中层次推断当作 SGLang 硬件事实。
- [Don’t Break the Cache，2601.06007v1](https://arxiv.org/html/2601.06007v1)已经比较长程 agent 的缓存边界，并讨论动态内容置尾。其 Appendix C 给出策略实现方式。因此“Question 置尾”属于有依据的工程修正，不是独立贡献；本项目 B0 仍保持原生前缀缓存，不照搬通过随机 UUID 刻意破坏命中的无缓存对照作为主要弱基线。

这两项工作目前只纳入方法比较与实验设计，没有复制未经确认的代码包。若后续扩大文献检索发现与本项目相同的动作/观测组合，应继续收窄主张，转向可复现工程和适用边界贡献。

继续核查的 [CacheSmith（2026-10）](https://arxiv.org/html/2610.18925v1) 已有转移感知、离线标定与队列预算驱动的自适应 KV 重用计划器，其动作是 WarmUp/Probe/Patch 与局部重算。因此 B 也不能宣称首次用实际转移成本进行在线计划，区别应限定为相同 required/full QA 下的完整文本 APPEND/REBUILD 决策。RR-Evict、生产淘汰分析、SpecCache 和 GroundedCache 的原理、边界与五项可落地机制，见[较新工作筛选](research_screening.md)；其中已实施资料锚点作用域纠偏，没有把答案缓存或部分重算混作精确前缀优化。

### 与已有工作的最小差异及其证伪方式

| 对照工作 | 不能重复宣称的已有内容 | 本项目需要证明的最小差异 |
|---|---|---|
| ContextPilot | 上下文重排、精确去重、位置/排名注释、与缓存关联的索引 | 重建后当前可见性、版本与条件闭包成为所有候选共用的约束；在其思想适配基线之上，成本选择仍有额外价值。当前线性会话索引不能代表完整 ContextPilot 的跨会话层次索引 |
| TokenPilot / CAPC | 前缀稳定、归档恢复、生命周期门控、缓存感知的成本权衡 | 在不裁剪完整 QA/必需正文的前提下，用本地驻留/恢复测量决定何时整轮重建，而不是复述稳定前缀和批量清理原则 |
| RAGCache / EfficientAgent | 前缀依赖、恢复对重算成本、工作集/容量分析、缓存保留或写入准入 | 决策变量是两个实际提示布局及其后续 Decode，不是固定输入上的替换/写入策略。若只靠缓存容量或原生写策略解释收益，B 的独立贡献不成立 |
| CacheBlend / Cache-Craft | 非前缀 chunk KV 复用与选择性重算 | 保持原生精确前缀计算的独立路线及其代价边界；不声称此路线普遍更快，或完整证据就保证模型质量不变 |
| 原生 SGLang / LightRSI | session generation、软引用、版本检查、CAS 式提交 | 业务语义版本成功提交与缓存 epoch 退役的可恢复绑定，以及它对其他会话工作集的可测影响；事务/幂等本身不是新算法 |

## 贡献 A：以当前可见性和版本依赖约束的上下文组织适配

**可检验主张。** 将已有排序、去重和注释方法适配到会发生重建、版本变化、历史回访的多轮文档问答，保证两个候选共享本轮必需材料，只有仍在当前实际输入中的完整正文才能被引用替代。在质量约束成立时，稳定组织可能减少重复 Prefill；收益大小及适用重叠率由实验确定。

**已有能力 → 具体复用。** ContextPilot 已有上下文距离、前缀对齐、去重与位置提示。本项目 [`adapters/contextpilot.py`](../adapters/contextpilot.py)按[固定标量距离源码](https://github.com/EfficientContext/ContextPilot/blob/b4c80327f6cfae2286839413731fcb6f53578b2f/contextpilot/context_index/compute_distance_cpu.py)实现 `context_distance`，按[固定前缀对齐与注释路径](https://github.com/EfficientContext/ContextPilot/blob/b4c80327f6cfae2286839413731fcb6f53578b2f/contextpilot/server/live_index.py)实现只读 `order_evidence` 和 `reuse_annotation`，并依据论文 §5.3 用 `retrieval_order_annotation` 在本轮证据正文之后保留原检索优先序，使缓存对齐后的正文顺序不掩盖相关性顺序。这是独立重实现的方法适配，不是直接调用整个上游服务或完整复现官方 hierarchical clustering。上游 `reorder`/`deduplicate` 会更新已见状态，不能在两个未执行候选上直接调用。

**必要改动。** [`evidence.py`](../controller/evidence.py)、[`candidates.py`](../controller/candidates.py)、[`templates.py`](../controller/templates.py)将“历史见过”改为“当前执行账本中仍有正文”，维护稳定证据身份、引用与正文 span；版本冲突使 APPEND 不可行，REBUILD 保留完整原始 QA 并恢复本轮所需证据。两个布局都把证据放在变化的问题之前，并用真实 tokenizer 验证前缀，而不是用字符长度估计。新增 [`corpus.py`](../adapters/corpus.py)适配真实文档/片段/版本及显式条件依赖；保守自然回访只使用已提交历史与当前问题，并记录选择原因。

`CorpusIndex.retrieved_keys` 保留原始 top-k 优先序，`resolve` 再追加条件闭包；候选的 `retrieval_priority` 与完整 required 集合分别传递。B4 因而能保留原检索语义而不为新增条件/回访材料虚构排名，新增映射只进入当前消息尾部。

**假设、基线与消融。**

- 假设 A1：在真实重叠和回访存在时，当前可见性检查消除“正文已移除却仅给引用”的失败，版本/条件错误不增加。
- 假设 A2：共享 required 材料下，稳定排序或 ContextPilot 风格布局增加有效前缀，并使真实 Prefill 的减少覆盖其组织开销；低重叠时允许没有净收益。
- 布局实验固定 B、C，只比较正常全量重建、稳定布局追加和独立 ContextPilot 风格布局；不能把两个都走固定追加分支的策略名称当成独立算法。
- 回访策略单独消融 `explicit` / `conservative`，所有布局使用同一版本与条件规则。省略条件或破坏可见性的错误实现只能作为故障演示，不能充当弱化的正式基线。

**工程产物与验收。** A 交付可独立调用的语料适配器、纯候选生成器、稳定引用和证据审计；[`test_candidates.py`](../tests/test_candidates.py)、[`test_corpus.py`](../tests/test_corpus.py)、[`test_tokenizer_optional.py`](../tests/test_tokenizer_optional.py)验证不丢正文/QA、回访/版本/条件与真实 token 一致性。实验还需报告重复正文量、构建耗时、理论 LCP、实际命中/Prefill、版本/条件/引用错误及完整回答质量。该贡献是场景适配与方法改进，不宣称创造了前缀缓存或通用语义理解方法。

## 贡献 B：真实驻留与完整回答成本驱动的追加/重建计划器

**可检验主张。** 在同一份 required 材料和同一引擎执行语义下，利用当前 GPU ready / host 可恢复前缀与实际恢复行为、输出长度先验和测量成本，选择 APPEND 或 REBUILD。与调优后的固定阈值、固定周期和冷热门控相比，在部分容量/负载区域降低完整响应成本，同时明确无效区域和预测失败原因。这是论文的主要方法贡献；“采用插值”或“增加一个 if 判断”本身不是贡献。

**已有能力 → 具体复用。** SGLang 原生匹配、恢复、调度和 kernel 保持不变；研究补丁在 scheduler 所有者上提供同批只读报价。TokenPilot 的批量稳定思想成为切换门槛/滞回的依据，EfficientAgent 的恢复成本与工作集分析成为压力和传输测量的依据。两者没有现成的本项目 APPEND/REBUILD 计划器代码：[`cost_model.py`](../controller/cost_model.py)与[`planner.py`](../controller/planner.py)是本项目实现的增量。

**必要改动。** 计划器先检查证据、历史、版本和预算约束；再依据选中输入的真实 token 与报价预测成本：

`J(candidate) = controller_cost + measured_first_token_phase + expected_later_decode`

当前真实探针的 `first_token` 是服务端收到请求至首 token 的联合墙时，包含其中发生的排队、恢复和 Prefill；不能再重复加同一段排队/H2D，也不能将它称为独立 kernel 时间。Decode 排除已计入联合阶段的首 token。输出长度只取开发集和本会话已经完成的回答。未覆盖的测量范围、未知 host 状态或不确定恢复触发显式固定策略回退；预计节省必须超过误差/门槛并满足稳定轮次。容量风险和尾部成本目前属于诊断输出，未据此实现全局最优资源调度。

标定由 [`profiling/sglang_probe.py`](../profiling/sglang_probe.py)和[`profiling/fit.py`](../profiling/fit.py)完成；支持冷/GPU/host 状态准备、独立测量单元留出、误差/覆盖率报告和冻结模型。服务集成见 [`service.py`](../controller/service.py)，固定策略跳过不需要的报价。CPU 解析公式只作诊断，不得冒充目标设备标定。

**假设、基线与消融。**

- 假设 B1：相同逻辑前缀在 GPU、host 或不可用状态下的最优动作可能不同，真实报价比单纯输入长度/命中比例提供额外信息。
- 假设 B2：长 APPEND 即使 Prefill 省得多，也可能因后续 Decode 更贵而失去优势；完整成本选择比只看 Prefill 更可靠。
- 假设 B3：误差门槛和滞回能减少收益不足的频繁重建，额外控制开销被端到端计入后仍有可解释收益。
- 对照必须包含原生 B0/B1、同生命周期下的固定 token 阈值/周期 B3、独立 ContextPilot 布局 B4、warm/idle B5，以及仅从开发集选出的最强简单策略 B6；同调参预算、同资源上限、同材料、同输出预算。
- 分别关闭 host 成本、Decode 成本、控制成本、滞回，并比较 frozen / 在线残差版本。关闭分项的消融不能通过残差学习把该成本重新混回模型。

**工程产物与验收。** 交付标定入口/冻结文件、预测和回退原因、实际压力/传输信息、完整实验矩阵。先确认预测覆盖率与成对决策误差，再评价完整回答时间、会话 makespan、吞吐、失败/截断和控制成本；不能只报告选择准确率。若一直回退，结果只能证明回退策略可运行；若没有超过调优 B6，就不能声称成本计划器有效。H200 上的标定必须与测量特征和运行时查询口径一致。

## 贡献 C：语义提交与上下文版本缓存生命周期的系统协同

**可检验主张。** 将业务会话和上下文版本分离：新版本的完整回答成功持久化后，才撤销旧版本的缓存软引用。该协议在失败/重启中保持可恢复的语义历史，并在多会话容量压力下改善过时分支的淘汰机会。是否转化为少传输、少重算和更短完成时间，需要物理事件证明。

**已有能力 → 具体复用。** 直接复用 SGLang [固定 `session_ref_tracker.py`](https://github.com/sgl-project/sglang/blob/3f36fc08331f64cd63c4fbbdf39cc402f38bb38b/python/sglang/srt/mem_cache/unified_cache/session_ref_tracker.py)的 generation 与软引用机制、原生关闭/淘汰路径。LightRSI [生命周期计划器](https://github.com/zjunlp/LightRSI/blob/9f0f19308a30445438779efbf3108a7d965a55c7/components/packages/features/eviction/src/lifecycle-planner.ts)的版本与计划/提交分离是设计参考。SQLite CAS、outbox、幂等操作等为成熟工程手段，不分别标为新算法。

**必要改动。** [`session_store.py`](../controller/session_store.py)、[`epoch.py`](../controller/epoch.py)、[`service.py`](../controller/service.py)把完整 QA 档案、实际执行账本和 epoch 绑定：候选不写已见状态；只有选中版本 prepare；结果可恢复后 CAS 提交；同一事务写旧 epoch 退役 outbox；异步 ACK/重试不会回滚新答案。研究补丁补齐报价、guard、终态查询及可观测边界，普通原生 session ID 不假装具备完整业务事务语义。

**假设、基线与消融。**

- 正确性假设 C1：生成失败、响应丢失、提交竞争、重复退役或引擎重启不会提交半答案，也不会在新版本成功前提前废弃旧当前版本。
- 性能假设 C2：在有限 GPU/host 容量下，撤销旧独有分支的保护可让当前工作集更易存活；共享前缀仍可复用。容量充足或没有竞争时，允许没有性能收益。
- 固定 A、B 后比较 `epoch_retirement=true/false`。无 C 路径仍保留必要故障保护，仅让文本重建沿用原生 session，不可通过删除恢复逻辑制造更快的错误对照。
- 用原生 B1 检验是否仅来自 SGLang 自带能力；用故障注入验证语义边界；用压力扫描验证物理效果。host 写入准入是独立可选扩展，当前不计入 C。

**工程产物与验收。** 交付 CLI/API、持久账本、恢复命令、固定源码补丁及 CPU 合同/真实引擎冒烟入口，见 [`engine_patches/README.md`](../engine_patches/README.md)。验收链必须逐项有证据：**账本提交 → 退役 ACK → 引用状态变化 → 实际淘汰/唯一 slot 占用变化 → 后续请求恢复/重算变化**。ACK 只说明引用被撤销，不说明立即释放了 KV；预留显存、逻辑 prompt token 和多个引用相加也不能代替唯一物理占用。

## 统一实验与工程完成标准

[`bench/experiments.py`](../bench/experiments.py)负责等预算开发集选择和冻结测试，[`bench/report.py`](../bench/report.py)、[`bench/judge.py`](../bench/judge.py)、[`bench/blind_review.py`](../bench/blind_review.py)负责运行汇总、外部 judge 和人工盲审。P1 固定参考历史用于受控诊断；P2 使用各方法自身回答与固定证据；P3 使用真实检索与自身回答。不能把 P1 的结果等同于完整闭环能力，也不能用测试集答案、未来轮次或质量标签驱动在线策略。

H200 是验证设备，不自动满足“有限缓存”的实验条件：需要记录实际 KV pool、host pool、page size、并发与模型配置，在所有方法间保持预算相同。预先冻结容量和负载网格，包含低重叠、低压力和高压力区域；报告全部区域，不能只挑有利点。质量阈值、调参预算、随机种子和统计单位在测试前确定，按会话做配对区间；完整原文审计条件/版本/回访差异，LLM judge 之外保留人工盲审。

工程完成要求是每个公开入口可执行、模块合同一致、关键失败路径可恢复、证据和测量可追溯；研究完成还要求 A/B/C 的假设得到有边界的实证支持。若某项收益不成立，应缩小贡献主张或删除无效机制。硕士论文可以贡献一个经过验证的适用边界与工程方法，不能以更多代码或更多策略名称代替实验论证。

## 独立审查：对照技术方案仍需补齐的研究证据

下表区分已有可执行入口和尚缺的验收证据。完成某个入口不自动消除后面的缺口；优先级针对论文结论的可信度，不代表需要额外搭建大系统。

| 优先级 | 当前有的能力 | 尚缺证据或必要补充 | 结论约束 |
|---|---|---|---|
| P0 | 硬件探针、冻结标定及[离线成对入口](decision_pairs.md)；新增 90 对分层诊断与独立离线重评分 | 已观察成本切换，但同条件 regret 尚差于固定策略。需要独立数据检验候选长度及成对误差，并完成同预算闭环/质量对照；不在已看 90 对上回调后宣称验证 | B 的性能贡献尚未成立；不得以点预测 MAE、额外拟合器或未经质量检查的更短输出替代增量证据 |
| P0 | `bench.experiments` 等预算选择和冻结配置 | 正式硬件矩阵需加入原生 B0/B1、调优 B3、B4/B5、实际 host 写策略组合，并记录相同资源/清缓存协议；现有 smoke 的 B6 只是候选方法中的赢家，不自动等于最强缓存组合 | 没有强 B6 或基线缓存预算不等，就不能支持 B 的增量；先用少量开发集关键点缩小网格即可 |
| P0 | 官方格式、完整证据审计、judge/盲审/配对区间工具 | 用真实生成填充同任务质量评分，预先冻结非劣界限；重点核验条件、版本、回访、不可回答和输出截断。`quality_eligible` 尚需来自开发集评分，不是工具自动证明质量合格 | 完整 required 覆盖只证明材料没有缺失，不能代替回答质量；模拟回答不能用于非劣论证 |
| P1 | 退役事件、唯一池占用采样、传输 ACK、无 C 开关及已完成的 GPU/host 受控路径探针 | 限定 fixture 已区分旧/新独有路径与共享祖先，并观察到相同淘汰压力下的保留变化；还需固定动作轨迹的真实多会话完成时间对照 | 退役瞬间 pool 变化为 0；受控路径效果不能外推为一般吞吐提升，也不能只由 ACK 声称物理释放 |
| P1 | 开关式 host/Decode/控制成本/滞回/退役消融 | 为正式矩阵冻结同一个 Ours 基础配置，再逐项只改一个开关；补自然回访 explicit/conservative 的共享质量对照。B4 的排序和排名注释暂作为整体适配，不单独归因每一部分 | smoke 中独立命名配置不必然与已选 Ours 只差一个因素；若继承参数不同，不能称单因素消融 |
| P1 | 端到端、服务端与容量统计 | 将自然容量及受控容量、低重叠与高重叠、无/有 host、短/长输出的负例纳入结果；检测 C 收益是否被更大缓存或不同动作轨迹混淆 | 若主效应仅见于特定区域，就按区域报告；H200 大显存上无收益也应作为结论保留 |

对于 C，完整闭环的“同算法开/关退役”给出的是总效果；若要证明“某旧分支被回收使另一会话少重算”的机制，还需要固定选中 prompt/动作轨迹的受控实验。对于 B，低平均时间误差并不保证成本差值方向正确，应分别报告预测差、实际差、选错时的损失和切换频率。当前不把复杂长期预测、全图记忆、host 准入迁移或新的 attention kernel 设为完成上述核心验收的前置条件。
