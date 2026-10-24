# 硕士课题贡献、复用依据与验收证据

本课题的主线是：**复用成熟开源方法，围绕证据组织、候选完整成本预测和缓存生命周期形成可检验的联合改进，提升有限缓存、多会话文档问答的质量与完整响应效率。** 当前 APPEND/REBUILD 实验在相同 required 证据和完整 QA 下比较；证据重评分、检索改进及其他缓存路线也可作为明确的新实验分支推进。研究对象是一次完整问答及其对其他会话缓存的影响，最终同时检查质量、时延、资源和工程开销。

三个贡献方向分别是：**A 改进证据选择、组织与来源绑定，B 改进候选完整成本与切换决策，C 改进语义提交和有限缓存资源的协同。** 三者都围绕可用答案、完整响应时间和资源占用形成独立假设，也共同接受系统实验。原生 RAG、BM25、RadixCache、HiCache、session 软引用、SQLite 事务和一般插值是复用起点；增量在于面向本场景的机制优化及可复核效果，而非给成熟组件重新命名。下述为**待实验支持的贡献主张**，代码实现和测试通过不代替性能或质量结论。

**A、B、C 均继续发展，不预先规定只能由 B 形成方法贡献。** A 已进入检索到真实回答的闭环，B 已从共享长度先验推进到候选预测与成对收益校准，C 已进入实际 host 恢复可触发的有限容量负载。每条路线从复用、必要改动、可证伪假设和强对照建立增量，最后按实验结果组织论文。相近方法和负面结果用于定位下一次改进，不自动成为收紧研究目标的理由；宽泛的“首次”表述仍需要证据。

复用范围以目标场景价值决定：可直接采用作者实现、使用其模型权重、适配局部机制，或组合多个方法后扩展。候选机制按答案可用性、完整响应成本、资源占用和维护复杂度持续筛选；贡献可以来自单模块改进，也可以来自跨模块协同及有解释力的系统实验。研究过程中继续扩大有价值的方法覆盖，成果表述随真实证据更新。

最近一轮的增量证据和实现进一步明确了三个切入点：

- **A：按用途分配查询与来源。** 已直接复用作者ConvSearch-R1权重/协议，与来源约束selector、用户主题视图做[全部41轮八方法对照](frozen_query_retrieval_validation.md)。长改写的候选池有可利用召回，但直接作为重排query退化，因此新增可选召回/重排查询分离；[四臂164条实测](frozen_rerank_validation.md)将长改写的R@5从.2701恢复至.55，但简单用户主题视图已达.5487且nDCG更高，不能据此声称复杂改写更优。低成本视图保留为下一统一系统对照的共同A配置，来源选择与必要依赖仍继续优化。固定SDK原文抽取和[12次生成](api_evidence_validation.md)揭示“单方法签名正确而完整初始化错误”，构建器已支持显式初始化合同，后续结合[必要性与局部依赖研究](api_evidence_followup_research.md)完善。研究增量应在所需信息、来源适用范围和完整答案成本的联合效果上体现，不将AST或普通query改写单独称首创。
- **B：用更强的简单决策检验复杂预测。** [正文冗余阈值](redundancy_baseline.md)补齐了技术方案中的B3候选，按固定tokenizer测量旧非required正文，不用完整prompt长度差冒充冗余。它不需要quote或成本模型；与已完成的长度/阶段/预算改进共同进入有限开发比较，用净效果判断复杂计划器的有效区域。
- **C：写入选择会改变被保留的业务工作集。** [中等容量16次重访](host_admission_validation.md)已经分别观察到另一业务前缀被保住、自身尾段被推迟备份后增加重算。业务身份、实际已提交备份预算和原生淘汰事件的结合具有明确作用路径；最终仍需真实多会话闭环确定取舍是否划算，不能只按D2H下降组织贡献。

上述方向允许进一步组合、替换及优化。公开[GLiNER2.5/关系抽取/EXIT研究](topic_extraction_reuse.md)保留为可复用候选，当前优先处理已经定位的召回与重排瓶颈，未下载或宣称实现这些新抽取模型。

最新[独立研究审查](research_review_20261024.md)进一步纳入 SmoothAgent、AKV 与 SCR：注意力工作量解析式、Prefill/Decode 成本分解、上下文变换的准备/提交和缓存状态生命周期均有直接近邻。B 的增量需要落到完整 QA / 同一 required 证据 / 原生精确前缀约束下的实测选择效果；C 的增量需要落到答案持久化后退役的具体事务边界及其实际缓存影响。当前不将通用公式、正系数拟合、提示词修订或 CAS/outbox 单列为研究创新。

**已完成的改进与接续问题：**

| 路线 | 当前真实实现与观测 | 接续的可检验改进 |
|---|---|---|
| A：取证与来源 | BGE/RRF/reranker 已接入共享 P3，开发 qrels R@5 从 BM25/last3 的 .2094 到 hybrid/current 的 .5231，但等会话完成时间 2.581→3.654 s；两组同时改变 query 视角和检索器。[P3 审查](p3_retrieval_integration.md)保留更慢与回答失误。纠正查询将附件限额表从 rank17 提到3，真实回答却仍混淆限额对象并误引历史助手句子；[MiniCheck v2](grounding_v2_assistant_audit.md)已运行并由替换改为保留旧来源的引用增补 | 在已取得相关正文的基础上，检验行/对象/条件绑定与来源 span 核查；与直接提示、单来源 NLI、现有增补比较，计入额外验证与修订成本。并行检验检索成批和布局对共享 GPU 的影响 |
| B：成本与动作 | [候选长度](length_decision_validation.md)和[条件误差](conditional_uncertainty.md)已实现、训练并完成同一 90-pair 的会话 OOF 重评分；shared/global→conditional/candidate 的 regret 为437.05→270.30 ms。[嵌套收益校准](gain_calibration_validation.md)进一步到236.14 ms，但动作与简单稳定期重建规则相同，仍高于固定REBUILD的129.14 ms | 改进可分辨“错误重建/漏判”的提交前候选特征，继续比较外部预测器与直接差值目标；将一次切换的后续摊销和质量纳入真实闭环。保留简单规则，不把低 MAE 或截距修正当成已学到细粒度决策 |
| C：容量与生命周期 | 事务/epoch/退役协议、受控两层路径探针与[原生写策略32K/16K实验](native_policy_validation.md)已完成；32K三组host恢复均0，16K三组均触发，量为10,752/896/13,120 tokens。恢复量最多不等于运行最快 | 在已激活的容量区间复用原生写策略、回访/工作集特征与退役事件，比较保留/退役及必要的写入准入扩展；用固定轨迹解释机制，再用闭环检验质量、资源与完成时间 |

B表中的regret来自隔离c1 fixture的server elapsed、尚未纳入答案质量，不能写成真实P3/c4已获得的完整响应收益；C的三策略数值来自各一次运行，不能按恢复量给策略排优劣。已有dev用于训练探索、会话OOF和必要的嵌套选择；同一case的状态/重复不增加独立会话数。独立test继续冻结，但开发探索不等待人为设定的新会话数量或单一机制先成功。负面结果、版本和参数选择全部保留。

实验分两层组织：同材料矩阵隔离布局/成本/退役收益；检索或质量增强矩阵允许选择与回答过程变化，共享语料、问题、生成模型和资源预算，计入新增开销并审查完整答案。新机制不需要先满足“完全不改变任何材料”才有资格探索。

**有限容量联合实验进一步明确了改进位置。** 新的[16K 标定](capacity_calibration_validation.md)与[六方法 246 轮运行](controller_capacity_validation.md)均已完成。Ours/no-retire 的全部输入和输出一致，暂未观察到退役的时延收益；34 次回退主要来自将 free-slot 不足统一视为不可用估计的门控，其中 29 轮两臂都已有时序支持。后续扩展因此包括压力与传输不确定性的区分、提交前可观测工作量及前置等待估计，而不是只扩大 lookup 网格或放宽误差阈值。A 的表格关系/来源绑定与 C 的工作集/写策略仍独立发展，通过联合效果确定最终贡献组合。

模块 A 的[混合检索](hybrid_retrieval.md)、[纠正识别 query](correction_query_view.md)都已有真实 P3 结果；[Qwen3同池对照](qwen3_rerank_validation.md)也已运行，39个有标签任务上 R@5=.4085，低于 BGE .5231，故保留 BGE 基座。较新方法不按发布日期自动采用；也不因一次负结果停止原理分析和场景适配。下一阶段选项见[检索与来源研究路线](retrieval_research_next.md)。

接续实现已经落到可选主路径：A 的[字面表格关系注释](table_relations.md)保留原文并增加来源行/对象/值/unit span；B 的[阶段成本](stage_cost_model.md)与[工作量等待模型](admission_wait_model.md)将可测执行和待校准的异质等待分开，原生报价区分容量压力与在途传输，并提供当前请求的标量工作量。前者已完成[82 轮 P3 开关审核](table_relations_validation.md)和 8 次冻结历史对照，定位引用缺失与额外解释；进一步直接复用 CRAFT MIT 行文本并扩展当前 query 焦点与行内来源，[12 次冻结三臂对照](table_focus_validation.md)在两历史上均消除了已知目标范围/引用问题，且已可选接线。它是混合干预的局部开发改善，尚不支持总体质量结论；后者保留 strict 模式和固定规则对照，不以解析估计冒充校准。CRAFT、SMART、DRE 的行 focus、操作分解和来源范围研究，以及 TIE、WAIT、Tail-Aware 与 Sarathi 的工作量/内存/阶段分析，为后续扩展提供具体入口；当前实现边界分别列在上述文档，未把它们的完整系统或论文收益计作本项目成果。

本轮接续形成了可运行的增量链路：A 的[用户主题视图](user_topic_query_view.md)在五视图、820条真实检索结果中将 R@5 从 current 的52.31%提高到54.87%，同时保留候选召回退化；[82轮完整P3](user_topic_p3_validation.md)确认套餐/SDK取材改善与跨平台、代码扩写问题并存。进一步[复用少量独立查询与RRF融合思想](query_fusion_research.md)，实现相同query不重复检索、固定总重排预算的可选双路模式，[真实重排](current_topic_fusion_validation.md)恢复了候选覆盖但top5指标持平；[随后完整P3](current_topic_fusion_p3_validation.md)确认GUID定义与库链接的局部改善，同时保留版本/产品回归，不能把它直接列为整体质量收益。

B 的[原生预算与匹配时间观测](native_observation_validation.md)直接复用固定引擎预算函数并保存已有match标量，真实GPU/host探针已验证采集和生命周期合同；它为后续reservation/service分离提供数据，没有新建缓存树、反复hash或改变默认调度。评测侧的[原子诊断实测](atomic_judge_validation.md)则证明格式约束不能修复语义错误，下一轮[专用判别器与数量关系复用](quantity_judge_reuse.md)已落实到固定来源、许可、原生输入格式和具体适配边界。两者属于支持方法改进的工程与评价能力，不以观测通过或JSON合法单列质量贡献。

新增[原生预算与服务分离](native_budget_validation.md)已完成两种模型各41轮真实压力闭环：普通EOS预算公式与请求身份关联均通过；同支持样本的等待MAE未改善，平均完整响应下降同时尾部变差，尚不构成稳定净收益。后续以真实host/mixed采样补足支持域，而非放宽资格阈值。质量侧的[Lettuce](lettuce_judge_validation.md)与[Granite](granite_judge_validation.md)各完成同29项原生诊断，提供了适用区域与失效证据；测量能力改进与A/B/C方法贡献分别记录。接续的[来源结构/API证据路线](ac_extension_priorities.md)及[业务版本感知host准入](host_admission_design.md)已有具体原生接点、复用边界和可证伪验证方案。

## 为什么选择这组基础工作

| 基础工作 | 已有能力与选型依据 | 本项目实际复用及边界 |
|---|---|---|
| SGLang | 同一引擎已提供精确前缀缓存、host tier、原生 admission 和 session 软引用，能保持底层计算语义不变研究上层决策。[原生 session 说明](https://docs.sglang.io/docs/advanced_features/session_radix_cache) | **直接复用引擎代码**；固定 `3f36fc0…`，新增只读报价、epoch 协议和测量接缝。选 SGLang 是为了在一个可核查运行时里比较布局、恢复与退役，不是断言它优于所有引擎 |
| ContextPilot | 直接研究上下文重叠、排序、去重和注释，问题与 A 最接近。其 RAG 实验涵盖多会话/多轮和 MTRAG；离线批量与在线冷启动实验设置有区别，不能混用结论。[论文 §7](https://arxiv.org/html/2511.03475v4#S7) | **方法原理适配**：复用固定源码的 overlap/position 度量、命中前缀对齐和引用注释思想；当前未接入完整层次索引、跨会话聚类或批调度 |
| TokenPilot / LightRSI | 将前缀稳定与保守批量清理结合，说明“删得更短”与“保留复用”需要共同考虑。论文使用 PinchBench/Claw-Eval、GPT-5.4-mini、API 缓存计数和货币成本，属于与本地 GPU 不同的测量体系。[论文 §4.1](https://arxiv.org/html/2606.17016v1#S4.SS1) | **设计思想复用**：稳定内容与动态内容分离、批量/门槛触发、计划后提交；不引入其模型估计器、整套 TypeScript 服务或训练压缩器 |
| EfficientAgent | 对并发 agent 的可复用工作集、host 容量和写入/恢复收益做系统分析，说明 host 有副本并不保证下次能复用。主实验是 OpenHands、SWE-bench、vLLM/LMCache 和多 GPU；本项目不继承其部署结果。[论文 §5.1](https://arxiv.org/html/2609.33762v1#S5.SS1) | **机制适配**：已实现可选工作集/占用/淘汰条件下的写入准入，并扩展业务会话去重与每 Req 累计实际准入页预算。保留原生 SGLang 备份、锁与 ACK；未迁移其 connector；[GPU固定过滤](host_admission_validation.md)通路已验证，也暴露即时回访额外重算，条件压力及总体收益仍待验证 |
| MTRAG + BM25S | MTRAG 含非独立追问、不可回答问题与多领域完整会话，适合同时检验上下文行为和回答质量。[官方论文](https://arxiv.org/html/2501.03468v1) | **直接使用固定数据与检索库**，在线白名单与 P1/P2/P3 协议为本项目适配；检索变化与缓存策略收益分开评价 |

这组选择围绕一条因果链展开：**材料是否相关且被正确使用 → 怎样排列与复用 → 恢复/计算/生成分别要花多少 → 旧版本何时退出保护 → 下轮及其他会话是否受益。** 每项基础工作对应明确环节。ReCAP 当前只实现 warm/idle 启发对照、标为适配版；持久注意力图尚未接入，可作为另一候选选择路线评估，不把未实现的部分计入现有成果。

论文结果也不意味着任意质量指标都不降：例如 ContextPilot 的 QASPER 部分模型结果、TokenPilot 的部分任务质量结果并非全面优于未优化方法。因此本项目需要自行做逐会话质量比较、条件/版本审计和非劣检验，不能只引用上游“保持质量”的概括性描述。详细固定版本、许可证与 API 副作用核查见[相关工作记录](related_work.md)。

## 候选方法筛选：主动核查后的纳入与取舍

筛选标准为：能否解决当前可观察瓶颈、是否保持完整证据及因果前缀语义、有无可核查实现入口、迁移是否改变底层计算/质量条件、能否在硕士课题规模内形成公平对照。文献中报告过收益只能说明值得核查，不足以决定直接集成。

| 候选 | 官方证据与成熟能力 | 可复用入口、纳入内容 | 当前取舍与触发后续扩展的证据 |
|---|---|---|---|
| RAGCache | [arXiv v2](https://arxiv.org/abs/2404.12457v2)、[作者 TOCS 版本](https://chaojin0310.github.io/files/papers/TOCS25-RAGCache.pdf)使用保留文档顺序的知识树、GPU/host 两层、prefix-aware GDSF 与检索/Prefill 推测流水线；其比较覆盖 TTFT、长输出 TBT 和容量消融 | 可核查入口是论文 §5.1 的前缀感知代价/大小/频率策略、§5.3 的检索流水线；纳入“相同文档不等于相同因果 KV”和分项成本/容量消融的依据。当前没有把 PGDSF 说成本项目已经实现的策略 | 它属于精确前缀路线，不能与下面的部分重算方法混为一类。本次在论文/作者主页未确认可维护的官方源码包，不安装同名 PyPI 包。先复用 SGLang 已有两层缓存；若实际数据表明淘汰策略或检索等待成为瓶颈，再单独移植策略或流水线，而非同时替换缓存树和检索执行器 |
| CacheBlend | [原论文 §4/§7](https://arxiv.org/html/2405.16444v1)通过选择性重算融合非前缀 chunk KV，并在多跳问答等任务检验质量与 TTFT；[LMCache 官方当前入口](https://docs.lmcache.ai/kv_cache_optimizations/cacheblend.html)提供 `lmcache server --engine-type blend` / `BlendModule` | 存在真实可运行入口，值得作为另一条研究路线；它不是“只按语义相似度直接拿来用”的简单近似缓存。可借鉴质量—重算比例联合评价；当前未调用该引擎，也未移植其融合/位置恢复逻辑 | 会改变本项目完整前缀恢复后的计算路径，需要 LMCache/引擎集成和选择性重算支持。先不塞入 SGLang 主实验。若排序后仍大量存在非前缀块重叠且 Prefill 主导，可建立独立实验分支；届时要锁定兼容版本、同质量阈值和同资源预算，不能把其论文 TTFT 结果当作本项目端到端优势 |
| Cache-Craft | [作者所属 Adobe 官方页面](https://research.adobe.com/publication/cache-craft-managing-chunk-caches-for-efficient-retrieval-augmented-generation/)、[论文 2502.15734 §3–§5](https://arxiv.org/html/2502.15734v1)核实为 chunk-cache 可复用性、局部修复与层次管理方法；论文实现修改 vLLM/Triton attention，实验包括 LLaMA-3 8B/70B、A100、Sys-X 及公开任务 | 可核查方法入口是 cache variant 选择、重算提前终止和逐层预载；可用于分析“非前缀复用需修复”的代价。论文/作者页没有提供本次可确认的官方源码入口，当前没有代码复用，不把同名 CacheCraft 语义缓存项目当成论文实现 | 这是包含部分 Prefill、动态 attention mask、位置处理和多份 KV 的另一计算路线。可从非前缀重复与修复代价的小型对照评估利用价值，再安排所需实现；目前没有新增该 kernel 或声称复现，质量与迁移开销须单列 |

相较之下，ContextPilot 的纯上下文排序/注释可以在保持原生 attention 的前提下接入；SGLang 的 GPU/host 和 session 能力可以直接复用；TokenPilot 的批量切换与 EfficientAgent 的压力分析可以在控制器/测量层验证增量。这个选择优先形成一条可部署、可消融的完整链路，并不说明 CacheBlend 或 Cache-Craft 效果较差。若实测表明精确前缀路线不能利用主要重叠，应承认适用范围，而不是通过挑选样本回避其他路线。

来源辨认也属于工程前置工作：ContextPilot 的作者、题名及官方仓库由 [MLSys 2026 proceedings](https://proceedings.mlsys.org/paper_files/paper/2026/hash/b0131b6ee02a00b03fc3320176fec8f5-Abstract-Conference.html)与 `EfficientContext/ContextPilot` 交叉确认；不将同名项目或 fork 的接口/许可证默认归给论文。上述筛选没有复制未确认许可的实现，也没有把三个系统的性能数字拼成项目收益预期。

### 补充检索发现的直接相近工作

2026-10-23 进一步检索“cache-aware compression / context compaction / append-rebuild cost”后，发现两项需要纳入新颖性边界的作者论文：

- [Cache-Aware Prompt Compression（CAPC），2607.15516v1](https://arxiv.org/html/2607.15516v1)已经建立缓存与压缩的成本交叉模型，并提供按历史变化率选择稳定/动态区域的 AdaptiveCacheBoundary。其直接目标是 API 计费和命中率，核心验证围绕 Sonnet API；文中 Appendix A 是可核查的算法入口。本项目因此不把“压缩可能破坏缓存”或“利用成本选策略”作为新发现。区别限于本地 GPU/host 的实际驻留与原生恢复、完整回答时延，以及共享证据约束的两种文本动作。未复用其压缩器、归一化或 API 专属阈值，也没有将 API 命中层次推断当作 SGLang 硬件事实。
- [Don’t Break the Cache，2601.06007v1](https://arxiv.org/html/2601.06007v1)已经比较长程 agent 的缓存边界，并讨论动态内容置尾。其 Appendix C 给出策略实现方式。因此“Question 置尾”属于有依据的工程修正，不是独立贡献；本项目 B0 仍保持原生前缀缓存，不照搬通过随机 UUID 刻意破坏命中的无缓存对照作为主要弱基线。

这两项工作目前纳入方法比较与实验设计，没有复制未经确认的代码包。后续发现更接近或更有效的实现时，应直接比较其能力、复用可维护部分，再围绕本场景尚未解决的质量、候选条件成本或资源协同继续优化；用更强对照检验增量，而不是因存在近邻就放弃方法改进。

继续核查的 [CacheSmith（2026-10）](https://arxiv.org/html/2610.18925v1) 已有转移感知、离线标定与队列预算驱动的自适应 KV 重用计划器，其动作是 WarmUp/Probe/Patch 与局部重算。因此 B 也不能宣称首次用实际转移成本进行在线计划，区别应限定为相同 required/full QA 下的完整文本 APPEND/REBUILD 决策。RR-Evict、生产淘汰分析、SpecCache 和 GroundedCache 的原理、边界与五项可落地机制，见[较新工作筛选](research_screening.md)；其中已实施资料锚点作用域纠偏，没有把答案缓存或部分重算混作精确前缀优化。

### 与已有工作的增量目标及检验方式

| 对照工作 | 可直接利用的已有内容 | 本项目继续发展的增量目标 |
|---|---|---|
| ContextPilot | 上下文重排、精确去重、位置/排名注释、与缓存关联的索引 | 重建后当前可见性、版本与条件闭包成为所有候选共用的约束；在其思想适配基线之上，成本选择仍有额外价值。当前线性会话索引不能代表完整 ContextPilot 的跨会话层次索引 |
| TokenPilot / CAPC | 前缀稳定、归档恢复、生命周期门控、缓存感知的成本权衡 | 在不裁剪完整 QA/必需正文的前提下，用本地驻留/恢复测量决定何时整轮重建，而不是复述稳定前缀和批量清理原则 |
| RAGCache / EfficientAgent | 前缀依赖、恢复对重算成本、工作集/容量分析、缓存保留或写入准入 | 决策变量是两个实际提示布局及其后续 Decode，不是固定输入上的替换/写入策略。若只靠缓存容量或原生写策略解释收益，B 的独立贡献不成立 |
| CacheBlend / Cache-Craft | 非前缀 chunk KV 复用与选择性重算 | 保持原生精确前缀计算的独立路线及其代价边界；不声称此路线普遍更快，或完整证据就保证模型质量不变 |
| 原生 SGLang / LightRSI | session generation、软引用、版本检查、CAS 式提交 | 业务语义版本成功提交与缓存 epoch 退役的可恢复绑定，以及它对其他会话工作集的可测影响；事务/幂等本身不是新算法 |

## 贡献 A：证据选择、组织与来源关系的协同改进

**可检验主张。** 将已有检索、重排、证据支持核查、排序与注释适配到会发生纠正、重建、版本变化和历史回访的多轮问答：既让回答所需材料进入当前输入，也把对象、条件和来源绑定到具体论断。布局比较保证两候选共享必需材料；检索/质量分支明确允许材料选择和回答发生变化。目标是减少错配和无据扩展，并使稳定组织的复用收益覆盖额外开销；两项目标分别测量后再检验联合效果。

**已有能力 → 具体复用。** ContextPilot 已有上下文距离、前缀对齐、去重与位置提示。本项目 [`adapters/contextpilot.py`](../adapters/contextpilot.py)按[固定标量距离源码](https://github.com/EfficientContext/ContextPilot/blob/b4c80327f6cfae2286839413731fcb6f53578b2f/contextpilot/context_index/compute_distance_cpu.py)实现 `context_distance`，按[固定前缀对齐与注释路径](https://github.com/EfficientContext/ContextPilot/blob/b4c80327f6cfae2286839413731fcb6f53578b2f/contextpilot/server/live_index.py)实现只读 `order_evidence` 和 `reuse_annotation`，并依据论文 §5.3 用 `retrieval_order_annotation` 在本轮证据正文之后保留原检索优先序，使缓存对齐后的正文顺序不掩盖相关性顺序。这是独立重实现的方法适配，不是直接调用整个上游服务或完整复现官方 hierarchical clustering。上游 `reorder`/`deduplicate` 会更新已见状态，不能在两个未执行候选上直接调用。

**必要改动。** [`evidence.py`](../controller/evidence.py)、[`candidates.py`](../controller/candidates.py)、[`templates.py`](../controller/templates.py)将“历史见过”改为“当前执行账本中仍有正文”，维护稳定证据身份、引用与正文 span；版本冲突使 APPEND 不可行，REBUILD 保留完整原始 QA 并恢复本轮所需证据。两个布局都把证据放在变化的问题之前，并用真实 tokenizer 验证前缀，而不是用字符长度估计。新增 [`corpus.py`](../adapters/corpus.py)适配真实文档/片段/版本及显式条件依赖；保守自然回访只使用已提交历史与当前问题，并记录选择原因。

`CorpusIndex.retrieved_keys` 保留原始 top-k 优先序，`resolve` 再追加条件闭包；候选的 `retrieval_priority` 与完整 required 集合分别传递。B4 因而能保留原检索语义而不为新增条件/回访材料虚构排名，新增映射只进入当前消息尾部。

**假设、基线与消融。**

- 假设 A1：在真实重叠和回访存在时，当前可见性检查消除“正文已移除却仅给引用”的失败，版本/条件错误不增加。
- 假设 A2：共享 required 材料下，稳定排序或 ContextPilot 风格布局增加有效前缀，并使真实 Prefill 的减少覆盖其组织开销；低重叠时允许没有净收益。
- 假设 A3：选择性用户历史能补足省略/纠正问题的意图；在原文保留的条件下，行对象与来源 span 绑定能减少“取回正确表却合并错限额”和“把历史助手句子当来源”的错误。分别与 current/last3、仅提示和单来源 NLI 对照，不以引用数增加代替支持正确率。
- 布局实验固定 B、C，只比较正常全量重建、稳定布局追加和独立 ContextPilot 风格布局；不能把两个都走固定追加分支的策略名称当成独立算法。
- 回访策略单独消融 `explicit` / `conservative`，所有布局使用同一版本与条件规则。省略条件或破坏可见性的错误实现只能作为故障演示，不能充当弱化的正式基线。

**工程产物与验收。** A 交付可独立调用的语料适配器、纯候选生成器、稳定引用和证据审计；[`test_candidates.py`](../tests/test_candidates.py)、[`test_corpus.py`](../tests/test_corpus.py)、[`test_tokenizer_optional.py`](../tests/test_tokenizer_optional.py)验证不丢正文/QA、回访/版本/条件与真实 token 一致性。实验还需报告重复正文量、构建耗时、理论 LCP、实际命中/Prefill、版本/条件/引用错误及完整回答质量。该贡献是场景适配与方法改进，不宣称创造了前缀缓存或通用语义理解方法。

候选构造的[分词优化](candidate_tokenization_profile.md)已在Linux固定tokenizer上验证15个case、30个候选的文本、token IDs及原冻结输入一致；四线程局部渲染/分词wall为22.61→7.58ms，process CPU为22.49→17.44ms（`37828a2`）。这是已验证的局部工程收益；并发control、TTFT及会话完成收益仍需端到端测量，不能用该比例代替整体效果。

**已落地的质量分支。** `adapters/hybrid_retrieval.py` / `retriever_factory.py` 共享模型与 query 配置，`bench/retrieval_eval.py` 和 `query_selector.py`分离检索与标签分析；`bench/grounding.py`直接复用固定 MiniCheck 评分，`citation_repair.py`已改为保留全部旧引用的增补，避免复合断言/段尾作用域的证据删除。真实增补产物已验证原文不变和旧引用保留，但没有把同一分类器分数再当独立质量真值。`bench/code_audit.py`补充不执行代码的结构检查；这些入口均不自动修改默认在线回答。

**继续扩展。** 在现有评分/增补基础上增加可追溯的对象—数值—条件和来源 span 对照，比较单来源、联合来源与原子论断支持，检查需要修订的是引用还是答案本身；[三个较新方法选项](retrieval_research_next.md)提供具体入口。官方 ContextPilot 索引、检索成批和其他质量方法仍可独立推进，不以当前纯排序或提示词为固定终点。先按各自作用定位效果，再检验它们与 B/C 的系统协同。

## 贡献 B：真实驻留与完整回答成本驱动的追加/重建计划器

**可检验主张。** 在同一份 required 材料和同一引擎执行语义下，利用当前 GPU ready / host 可恢复前缀、候选条件输出长度和实测误差，选择 APPEND 或 REBUILD。通过直接成本差与切换摊销建模，争取在质量合格时超过调优固定阈值、周期和简单门控。B 是完整成本决策方向；其研究价值由选择效果及失败解释建立，不由插值或回归函数本身建立。

**已有能力 → 具体复用。** SGLang 原生匹配、恢复、调度和 kernel 保持不变；研究补丁在 scheduler 所有者上提供同批只读报价。TokenPilot 的批量稳定思想成为切换门槛/滞回的依据，EfficientAgent 的恢复成本与工作集分析成为压力和传输测量的依据。两者没有现成的本项目 APPEND/REBUILD 计划器代码：[`cost_model.py`](../controller/cost_model.py)与[`planner.py`](../controller/planner.py)是本项目实现的增量。

**必要改动。** 计划器先检查证据、历史、版本和预算约束；再依据选中输入的真实 token 与报价预测成本：

`J(candidate) = controller_cost + measured_first_token_phase + expected_later_decode`

当前真实探针的 `first_token` 是服务端收到请求至首 token 的联合墙时，包含其中发生的排队、恢复和 Prefill；不能再重复加同一段排队/H2D，也不能将它称为独立 kernel 时间。Decode 排除已计入联合阶段的首 token。输出长度只取开发集和本会话已经完成的回答。未覆盖的测量范围、未知 host 状态或不确定恢复触发显式固定策略回退；预计节省必须超过误差/门槛并满足稳定轮次。容量风险和尾部成本目前属于诊断输出，未据此实现全局最优资源调度。

标定由 [`profiling/sglang_probe.py`](../profiling/sglang_probe.py)和[`profiling/fit.py`](../profiling/fit.py)完成；支持冷/GPU/host 状态准备、独立测量单元留出、误差/覆盖率报告和冻结模型。服务集成见 [`service.py`](../controller/service.py)，固定策略跳过不需要的报价。CPU 解析公式只作诊断，不得冒充目标设备标定。

[`controller/output_length.py`](../controller/output_length.py)已接可选单候选长度预测，`profiling/length_fit.py`复用 NumPy 做 shared/action/candidate/paired-delta 的会话嵌套验证；SSJF、LTR/PARS、TIE 的外部预测边界、相对差值和上限意识为方法参考，不是假称直接用了其训练模型。条件误差保留原点预测、按并发/host状态选经验误差；[`profiling/gain_fit.py`](../profiling/gain_fit.py)进一步以提交前预测成本差学习收益残差和单侧阈值，当前为离线实验，未偷偷改变线上 planner。新增模型和旧共享 prior 的来源边界均在各自报告中保留。

**假设、基线与消融。**

- 假设 B1：相同逻辑前缀在 GPU、host 或不可用状态下的最优动作可能不同，真实报价比单纯输入长度/命中比例提供额外信息。
- 假设 B2：长 APPEND 即使 Prefill 省得多，也可能因后续 Decode 更贵而失去优势；完整成本选择比只看 Prefill 更可靠。
- 假设 B3：误差门槛和滞回能减少收益不足的频繁重建，额外控制开销被端到端计入后仍有可解释收益。
- 假设 B4：候选条件与成对收益模型能区分同问题两布局的生成成本，并在未见会话上胜过动作常数偏移和“时间支持且稳定就重建”。当前增益模型与后者动作相同，构成下一版改进的具体对照；不把这一负结果转成必须放弃建模的结论。
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

**已完成的容量实验与接续。** 受控 GPU/host fixture 在同一淘汰压力下观察到旧/新独有路径保护变化，退役瞬间 pool 变化仍为0；[16K原生写策略实验](native_policy_validation.md)则在真实41轮负载中激活 host 恢复。前者隔离软引用作用，后者建立真实容量压力对照，不能互相替代。继续复用 SGLang 的 write-through/selective/write-back 与 EfficientAgent 工作集分析，研究版本回访、写入/恢复代价和过时保护的联动；是否加入写入准入或更长程保留预测由对照结果决定，不预设 C 只能交付事务正确性。

**工程产物与验收。** 交付 CLI/API、持久账本、恢复命令、固定源码补丁及 CPU 合同/真实引擎冒烟入口，见 [`engine_patches/README.md`](../engine_patches/README.md)。验收链必须逐项有证据：**账本提交 → 退役 ACK → 引用状态变化 → 实际淘汰/唯一 slot 占用变化 → 后续请求恢复/重算变化**。ACK 只说明引用被撤销，不说明立即释放了 KV；预留显存、逻辑 prompt token 和多个引用相加也不能代替唯一物理占用。

## 统一实验与工程完成标准

[`bench/experiments.py`](../bench/experiments.py)负责等预算开发集选择和冻结测试，[`bench/report.py`](../bench/report.py)、[`bench/judge.py`](../bench/judge.py)、[`bench/blind_review.py`](../bench/blind_review.py)负责运行汇总、外部 judge 和人工盲审。P1 固定参考历史用于受控诊断；P2 使用各方法自身回答与固定证据；P3 使用真实检索与自身回答。不能把 P1 的结果等同于完整闭环能力，也不能用测试集答案、未来轮次或质量标签驱动在线策略。

H200 是验证设备，不自动满足“有限缓存”的实验条件：需要记录实际 KV pool、host pool、page size、并发与模型配置，在所有方法间保持预算相同。预先冻结容量和负载网格，包含低重叠、低压力和高压力区域；报告全部区域，不能只挑有利点。质量阈值、调参预算、随机种子和统计单位在测试前确定，按会话做配对区间；完整原文审计条件/版本/回访差异，LLM judge 之外保留人工盲审。

工程完成要求是每个公开入口可执行、模块合同一致、关键失败路径可恢复、证据和测量可追溯；研究完成还要求 A/B/C 的改进获得实证支持。某个版本收益不足时，保留对照、定位原因并继续复用/优化/扩展，用新实验检验改进。最终论文按证据报告实际有效机制及适用条件，不能用更多代码或策略名称代替论证，也不把当前负结果预先写成研究目标的上限。

## 独立审查：对照技术方案仍需补齐的研究证据

下表区分已有可执行入口和尚缺的验收证据。完成某个入口不自动消除后面的缺口；优先级针对论文结论的可信度，不代表需要额外搭建大系统。

| 优先级 | 当前有的能力 | 尚缺证据或必要补充 | 结论约束 |
|---|---|---|---|
| P0 | 候选长度、并发条件误差、嵌套收益校准及90对真实观测的离线重评分均已完成 | 区分复杂模型与简单稳定期重建，扩充有解释力的候选特征并检查切换摊销；继续会话 OOF/独立开发数据检验，再冻结 test 与闭环质量条件 | 236.14ms regret 尚高于固定REBUILD129.14ms；当前结果指向机制改进，不是把训练工程重写成待办 |
| P0 | 已有控制器11 jobs+匹配消融5 jobs、原生B0/B1和32K/16K写策略实验 | 将实际会激活host的可靠写策略与容量纳入更强简单基线；轮换运行顺序、重复测量并共享同一检索/质量设置 | 不同P2答案和一次测量的时延差不能单归因缓存；公平比较允许分层探索后联合优化 |
| P0 | BGE/Qwen同池比较、P3混合检索、纠正query真实运行、MiniCheck评分/增补、代码审计 | 对已取回正文的限额关系和来源污染实施独立干预对照；补完整回答的支持、完整性、拒答和代码可用性评分 | qrels命中、引用存在、NLI概率、语法通过各自只验证一个维度；保持真实负例，不能互相替代 |
| P1 | 退役事件、唯一池采样、受控两层路径与16K真实host恢复 | 先用固定动作轨迹隔离退役/保留对工作集的影响，再测闭环；把版本回访及原生写策略与C共同比较 | ACK不等于即时物理回收，host-loaded量不等于收益；受控机制证据与真实业务效果分别报告 |
| P1 | 开关式 host/Decode/控制成本/滞回/退役消融 | 为正式矩阵冻结同一个 Ours 基础配置，再逐项只改一个开关；补自然回访 explicit/conservative 的共享质量对照。B4 的排序和排名注释暂作为整体适配，不单独归因每一部分 | smoke 中独立命名配置不必然与已选 Ours 只差一个因素；若继承参数不同，不能称单因素消融 |
| P1 | 端到端、服务端与容量统计 | 将自然容量及受控容量、低重叠与高重叠、无/有 host、短/长输出的负例纳入结果；检测 C 收益是否被更大缓存或不同动作轨迹混淆 | 若主效应仅见于特定区域，就按区域报告；H200 大显存上无收益也应作为结论保留 |

对于 C，完整闭环的“同算法开/关退役”给出的是总效果；若要证明“某旧分支被回收使另一会话少重算”的机制，还需要固定选中 prompt/动作轨迹的受控实验。对于 B，低平均时间误差并不保证成本差值方向正确，应分别报告预测差、实际差、选错时的损失和切换频率。当前不把复杂长期预测、全图记忆、host 准入迁移或新的 attention kernel 设为完成上述核心验收的前置条件。
