# 检索、限额关系与来源绑定的研究路线

> 归档记录：保留当时的设计、结果与复用依据；其中“下一步”和执行计划不代表当前任务。当前工作仅以[实现清单](../implementation_plan.md)为准。

2026-10-24 更新。目标是通过广泛的开源复用与必要优化取得较好的完整问答效果；下文区分已实现机制、真实结果和待验证方案，当前实现不是最终可探索范围。

## 已完成到哪一步

| 机制 | 已实现与实测 | 现在应解决的问题 |
|---|---|---|
| BGE-M3 + BM25/RRF + BGE reranker | 固定权重、本地加载、共享 P3 factory 和完整检索计时均已接通；[39轮qrels](../research/hybrid_retrieval_dev_analysis.md)与[41轮真实P3](../research/p3_retrieval_integration.md)已完成 | R@5提升并未自动成为更快、更正确的答案。保留query/retriever交叉对照，研究成批执行、选择性历史及后续证据使用 |
| 纠正识别 query | `correction_aware`已实现；三个词法触发只使用用户文字。真实P3匹配离线路由，附件限额表17→3 | [新答案仍把1MB文档、10MB附件、11MB请求混成一个限制关系](../research/correction_query_view.md)，并发生历史助手句子冒充外部引用；不是重新提出尚未实现的query选择 |
| 用户历史主题视图 | [可选 `user_topic`](../research/user_topic_query_view.md)已接共享 factory/双 CLI/离线评测，保留来源 user turn/span；41轮视图核对中3轮补入原用户主题、1轮复用纠正last3，无额外模型调用 | 尚未验证检索收益；与 current、last3、all_user、correction_aware 同预算比较。大小写/工具名和隐式换题仍有局限，不把词法线索当可靠实体消解 |
| Qwen3-Reranker-0.6B | [官方论文](https://arxiv.org/html/2506.05176v3)与固定Apache-2.0模型已复用；同50池/1024 cap真实评分41轮成功、0截断。[独立审查](qwen3_rerank_validation.md)复现R@5 .4085，低于BGE .5231 | 保留BGE及Qwen负结果。该结论限定于当前池与视角；部分qrels下降仍是相关替代段落未标注，不能当作总体回答错误 |
| MiniCheck + 引用增补 | 固定官方API已运行Q1/Q2完整audit；去掉语义输入中的引用标记后重测。变换已从替换改为保留旧引用、去重新增ID | [两个真实候选](../research/grounding_v2_assistant_audit.md)暴露复合句和段尾作用域。增补合同正确不等于全部子断言得到支持，仍需来源适用性与独立审查 |
| 原文表格行关系与焦点引用 | [确定性 helper](../research/table_relations.md)保留原文及跨度；可选 `focus_refs` 复用 CRAFT 行文本函数，在[12 次冻结探针](table_focus_validation.md)改善目标对象/引用错误，后续[三臂完整 P3 共123轮](table_focus_p3_validation.md)已成功，输入与完整历史均核验 | 目标局部改善仍伴随后续条件标题、跨产品SDK错误；继续长表、条件/版本和未触发轮回归，拆解混合干预。完整P3成功不是总体质量通过 |
| Querit-Reranker | [2026-06论文](https://arxiv.org/html/2606.19037v1)的目标分布合成query、teacher软标签与模型合并已分析；尚未迁移训练或运行权重 | 保留为领域适配选项；不重复计作已完成复用，也不把teacher生成标签称为完全无监督。优先处理当前已可定位的证据使用缺口 |

上述工作可继续在dev迭代。当前/最近历史/纠正路由与模型选型均保留记录，最终test继续冻结；不因五个开发会话的规模而停止实现，也不将这五个已观察会话重新命名为独立最终验证。

## 表格与来源关系的三项原理核查

检索聚焦2025–2026的表格结构保留、细粒度来源和对象归属。它们补足已有ContextPilot布局、CRAG/Self-RAG及MiniCheck支持评分，不再把同一篇泛RAG工作重复列为新机制。只使用作者论文与官方仓库；下面均是待验证的适配选项。

| 工作 | 已核查原理与直接入口 | 最小可复用干预与本项目区别 |
|---|---|---|
| **TabRAG，2025-11**：[论文§2及Appendix A](https://arxiv.org/html/2511.06582v1)；[作者源码固定版](https://github.com/jacobyhsi/TabRAG/tree/c3aba4190f141737f1a52ed1bacce95bec337667) | 区域解析后将单元格与行名、列名、单位绑定，再形成可检索表示。[`src/prompts.py`](https://github.com/jacobyhsi/TabRAG/blob/c3aba4190f141737f1a52ed1bacce95bec337667/src/prompts.py)的`VLMPrompts.vlm_table_prompt`给出row/column/value/units结构；[`LICENSE`](https://github.com/jacobyhsi/TabRAG/blob/c3aba4190f141737f1a52ed1bacce95bec337667/LICENSE)为CC BY-NC4.0。完整论文使用VLM、LLM和embedding，不是简单正则解析 | **结构绑定思想已作小范围适配**：`adapters/table_relations.py`保留原文，在旁路记录行对象、完整原表头、原值、显式单位和span；只有同标记脚注关联具体行，不推断未写条件或合计。支持真实扁平文本，无需视觉解析器。代码为独立实现，未复制上游prompt/code，不声称完整TabRAG复现或质量提升；固定仓库部分评测脚本未公开的边界仍保留 |
| **FullCite，2026-06**：[论文§4.1、§5、限制](https://arxiv.org/html/2606.07130v1) | 每个claim同时引用document ID与原文span，比较普通提示、文法约束和后置近似对齐；研究区分文档正确、span定位和支持关系。全文为CC BY4.0，本次未确认官方源码入口，不声称已有可直接安装包。作者结果包含span改善与语义对齐权衡 | **优先适配可验证引用定位**：对声称逐字引用的内容，用现有EvidenceKey/ref及字符offset在当前`audit_evidence`核对；若只在`actual_history.assistant`出现，记录来源混淆。后置相似对齐只产生待核查span，不继承论文阈值或把相似度当支持。完整QA仍保留，历史助手文本可以帮助理解会话，但不自动成为外部证据 |
| **Deceptive Grounding，2026-07**：[论文§5.4、Appendix A/B](https://arxiv.org/html/2607.09349v1)及[作者说明](https://cedriccaruzzo.github.io/blog/deceptive_grounding.html) | 引用真实、内容有据仍可能归给错误实体；EAV逐claim检查被引来源实体是否适用于问题对象。作者为临床场景，并分析judge在实体替换和部分句子归属上的盲点。论文CC BY-NC-ND4.0；本次未确认公开实现仓库，不复制其补充代码 | **适配对象/作用域核查**：在MiniCheck信号外，核对claim的产品/版本/限制对象与所引行的对象，而非仅检查“1MB”是否出现在同一段。将entity替换扩展为document/attachment/request关系是本项目的待验假设，不能套用临床论文检出率。也记录“转答另一个对象”和“没有来源”两类错误，避免只检测显式错绑 |

TabRAG的论文标题/仓库README略有差异，身份以论文直接链接的作者仓库为准；核查固定到上述commit，不依赖README中不一致的arXiv badge。表格关系helper已完成工程验证，其他方案仍是候选机制；均不将论文报告的性能换算成本项目收益。

## 可执行的干预对照

**选项一：原文行关系旁路。** [已实现helper与测试](../research/table_relations.md)为对照提供入口。在原段落不变、完整QA不变的条件下，比较无辅助表示、对象/数值/单位行表示、再扩展明确条件/例外绑定。现有1MB/10MB/11MB反例是回归样本，另已覆盖相同数值不同对象、非大小单位和源表明确总量行，不使用“一律禁止组合关系”的规则。接续补同对象不同版本与更复杂条件例，验收看无据关系新增率、条件保留、回答完整性和额外token/延迟，不只看JSON格式有效。

**选项二：引用来源定位与有限修订。** 比较现有引用增补、doc+span核对、doc+span+联合/原子claim支持。先在冻结初稿上检验诊断的漏报、误报及段尾覆盖；再将明确错误作为选择性修订输入，与统一修订/不修订对照。核对外部span和助手历史的来源类别，保留原回答与修订记录，不自动删除完整历史。逐字quote与普通paraphrase分开评价，无法逐字匹配的转述不能直接判为无据。

**选项三：对象适用性与NLI联合。** 比较MiniCheck单独评分、对象/关系检查单独使用、两者联合；对未知对象或不明确关系保留未知，不伪造确定的支持结论。该信号先作为离线诊断，再研究它是否值得触发补充检索或修订；在线动作依据误报与实际效果选择。新增信号必须改善实际错误识别/答案，而非通过多报错获得看似更严格的分数。

### 新一轮表格方法筛选后的实施顺序

[表格研究补充](../research/table_relations.md)已核查 CRAFT（ACL 2026）的作者 MIT 代码、TreeThinker（2025）的原表头路径、SMART（2026）的选择/计算/排序操作、DRE（2026）的错取值与漏行诊断、SSC（2026）的选择性复查和对→错评估。它们针对不同缺口，不能把“新增一个注释段”当成所有结构问题已经解决；也不把这些论文在单表短答案或含 oracle 输入的结果套用到本项目多轮 QA。

优先推进**带来源的行选择与范围核对**：每个数值 claim 保留它使用的对象、原行/值/条件 span，将“值可见”与“新增包含关系有依据”分开；先在保存初稿上检验，再做一次选择性修订。CRAFT 的 `build_row_texts` 小模块已经直接复用，现有 `focus_refs` 采用轻量词法选择并保留全部原文；现有 BGE 行评分是下一可比较选项，尚未实现。原始 HTML/LaTeX 仍有结构时，再扩展 header path 和数值操作验证；扁平文本缺失的关系不由解析器猜造。

这条顺序不依赖新底座训练或付费 judge，但有明确额外调用与模型误判风险，必须与不复查、普通 NLI、全量修订共同比较。对条件来源、历史助手污染和相关行遗漏仍分别计数；继续保留所有负面结果与未触发轮回归。DRE 原论文的 gold-assisted critic、CRAFT 的 OTT-QA gold passage、TreeThinker 提示中的“问题必有答案”都不进入在线输入。各固定源码、许可核查和具体改动入口见上述补充；词法焦点引用已实现为可选项，BGE 行评分、后置范围核对和更复杂结构路径仍待实现/验证。

这些实验首先共享冻结的初稿/原始材料以定位机制；真实闭环再共享语料、问题、生成模型、输出预算和资源，对照检索与回答过程的变化。代码块语法、claim支持、对象适用性、拒答和完整性分别报告；同一分类器选择修改后，不能仍由它单独证明质量提升。变动轮与未触发轮都保留，历史传播也计入闭环结果。

## 评估器也需要检验

独立本地judge可作为待盲审包的补充诊断，不应直接替代人工或重复使用MiniCheck自证修复。[GroUSE（COLING 2025）](https://aclanthology.org/2025.coling-main.304/)以具体grounded-QA失败模式评估judge，指出与强模型分数相关不代表能辨别这些错误。[官方固定仓库](https://github.com/illuin-tech/grouse/tree/e4c57ba9eb14548c7642daca3ed4ff24d760c7fa)的`meta-evaluate --train_set`提供16个开发例，另有144个test；代码[MIT许可证](https://github.com/illuin-tech/grouse/blob/e4c57ba9eb14548c7642daca3ed4ff24d760c7fa/LICENSE.txt)已核。现有 `bench/native_judge.py` 与 `bench/grouse_meta.py` 已接本地服务，复用固定原生模板、GroUSE 条件规则和失败语义，未启动上游默认外部 API 链路。

[Prometheus-7B-v2.0官方说明](https://huggingface.co/prometheus-eval/prometheus-7b-v2.0)提供rubric、reference answer和专用Mistral模板，绝对评分输出`[RESULT]`1–5。它和 Selene Mini 8B 均已完成真实首轮开发验证：原生两份 probe 各五维，共各 10 次调用，全部得到合法评分；另各执行 GroUSE train16。两者均未可靠辨别“文档 1 MB 包含附件”这一新增范围错误，因此合法格式和跨模型家族不是质量通过条件。[完整输入核对、原始分数与误差分析](open_judge_validation.md)保留全部结果；未读取 GroUSE test144 或项目最终 test。官方 GroUSE 专门微调模型本次访问未成功，不能写成已下载/可用。

当前需要改进的是已实现评估器的格式适配与语义辨别。Selene 原 GroUSE 协议 45 次调用有 39 次解析成功，96 项条件通过 48 项；Prometheus 的 32 次调用全部解析失败，其中 30 次到输出上限，其 0/96 不能解释为 96 次独立语义判断错误。Selene 的可选类型规范化已实现但尚缺后续真实调用，未生成新的通过率；范围限定词、引用作用域与合理拒答仍需独立改善。后续冻结协议再检查未见元评测并扩到完整开发回答，保留失败和不适用项。模型分数始终标为模型诊断，不称为 human review 或独立事实真值。

## 与B/C形成整体改进

检索增强已进入P3，不再把“接通主路径”列为下一步。接续比较同一选定质量设置下的候选长度/条件误差/收益模型，并重新核查encoder/reranker共享GPU后的成本标定；不能把空闲GPU标定直接当成新负载已验证。C则在已触发host恢复的有限容量区间比较版本退役、原生写策略及必要的写入准入适配。

A带来的额外检索、验证、修订及更长/更短答案，B的预测与切换开销，C的传输和保留影响，都纳入完整响应和会话完成时间。单模块改善不保证联合改善，联合结果也需通过消融定位贡献。若当前组合仍有不足，继续从这些可观察机制优化，不把单个负结果作为停止广泛复用的理由。

## 全开发覆盖扫描与原子条件绑定的新增线索

已对保存的 P3 off/on 各 41 轮做离线枚举，脚本和原始计时位于 `runs/table-focus-dev-review/`。82/82 保存输入可精确重建；每个变体保留各自完整实际历史与全部原段落。两组均仅附件纠正轮触发，两个来源各选中一个焦点行；相同历史下 `all_rows` 增加 417 tokens，`focus_refs` 增加 110，其余 40 轮输入逐 token 不变。这是覆盖与成本验证，不是 focus 的新闭环质量结论。Mac 稳态 parse+focus 的逐轮中位 CPU 再平均约 0.32 ms，首次可选库导入另列；Linux 服务和并发成本仍由真实运行衡量。

扫描还在前一轮发现两张**无编号标题但有明确表头的限额表**，以及一个**带套餐条件的嵌套列表**。前两者因当前扁平表要求编号 caption 而未解析；后者不能直接按数值行展开，否则会把上层套餐条件丢掉。完整原文一直在模型输入中；辅助表示未覆盖不等于检索遗漏。下一步可优先扩展显式无编号标题/表头的结构支持，并为列表建立可核查的父级条件路径，保留来源 offset；对无法确定条件范围的文本仍保持未解析状态。该扫描只产生三个待审查行簇，不将它们冒充表格召回率的完整真值。

以下三项是与此前 CRAFT、TabRAG、SMART 筛选不同的新入口，核查到作者原文及固定公开材料；未迁移其训练、代理工作流或评分结果。

| 新工作与固定依据 | 实质机制、可复用入口和边界 |
|---|---|
| **FACTS，Findings ACL 2026**：[论文](https://aclanthology.org/2026.findings-acl.453.pdf)、[作者固定源码](https://github.com/BorealisAI/FACTS/tree/449625cf73ca809d17fa891eab978f0d767f3eb6) | 将 query/schema 生成的 SQL 与 Jinja2 模板分离，执行查询后用具名列渲染，可复用到同 schema 的表。[`src/facts_framework.py`](https://github.com/BorealisAI/FACTS/blob/449625cf73ca809d17fa891eab978f0d767f3eb6/src/facts_framework.py)包含执行反馈、SQL/template 对齐及 `values` 渲染入口；[许可](https://github.com/BorealisAI/FACTS/blob/449625cf73ca809d17fa891eab978f0d767f3eb6/LICENSE)是 CC BY-NC-SA 4.0，不能当 MIT 模块直接并入。最小思想适配是把对象、值、单位、条件和来源作为具名字段投影再渲染，检查所有字段来源；不启动其默认外部 LLM council，也不把 SQL 成功执行当作语义正确。本场景先需要可靠结构提取，不能把任意扁平段落假设为完整关系表。 |
| **Dawn，SemEval 2026 Task 8**：[论文 §3、Appendix A](https://aclanthology.org/2026.semeval-1.363.pdf) | 将从参考段落抽取原子事实，与从会话恢复问题意图分开，再给出可回答性和回答范围信号；比单一提示词更强调信息来源分工。其多步实现使用 GPT-4o-mini，不能移用成当前 7B 的已验证收益。[官方补充材料](https://aclanthology.org/attachments/2026.semeval-1.363.SupplementaryMaterial.zip)经检查为论文 LaTeX/图及样式文件，并非可运行 RAG 包。可适配为“用户历史解析对象，外部 span 提供事实”的双来源记录：尤其避免历史助手错误成为外部引用。完整 QA 仍保留，事实旁路须绑定现有 ref/span；不直接复制整套多次分类/生成流程。 |
| **Atomic Reasoning / SciAtomicBench，2025-06**：[原文 §4–5](https://arxiv.org/html/2506.06972v1)、[作者固定仓库](https://github.com/CelestineZYJ/SciAtomicBench/tree/5d29811f2e672520dfbf19a61e5be1c0c18d7942) | 先定位 claim 对应单元格，再分别做结构、实体及数值操作，最后合并局部结果。论文主要验证包含微调的科学表格 claim 验证；固定仓库仅有 README 与四个领域 JSON，未见模型/推理实现或独立 LICENSE，不能声称已复用可运行 verifier。可借其分解原则，在现有保存答案上分别检查“对象匹配、数值/单位匹配、条件保持、跨行运算依据”，用显式未知保留无法判断部分。科学表格二分类和因果推理不直接迁移成限额问答真值。 |

据此，优先的可检验扩展是**带条件路径的来源事实记录与局部范围核对**：先让明确表头/父级条款进入原文 span 表示，再将回答中的对象、值和适用条件逐项对齐。与当前 `focus_refs` 的区别在于补足条件和范围，而非只换排序器；长表语义行评分仍可另用现有 BGE 做对照。比较时共享原始材料、初稿和生成预算，分别统计原对象修复、条件遗漏、新增无据包含关系、拒答和完整性，再进入闭环。来源字段可能解析错、原子拆分可能遗漏复合含义，这些失败都要保留。现有词法选择仅直接复用了 CRAFT 行文本函数，不是其 dense 重排算法复现。

## 引用生成的后续筛选

2026-10-24补查了[ALCE作者实现](https://github.com/princeton-nlp/ALCE/tree/246c476a4edfc564266b7346b6e29ef4861ae937)（MIT）、[ReClaim，NAACL Findings2025](https://aclanthology.org/2025.findings-naacl.55/)和[How Do LLMs Cite，2026-06 v1](https://arxiv.org/html/2606.28358v1)。ALCE把检索段放入编号文档格式并单列citation评价；ReClaim交替生成参考片段和回答句；较新的机制研究在Llama-3.1-8B上用activation patching研究引用行为，发现实体匹配等因素会影响引用。最后一项不是当前Qwen/SGLang上的即插即用修复，不能直接移用其组件或报告数字。

对本项目的可用原则是把**引用格式、来源支持、是否真正依赖来源**分开。现有Q1/Q2试验已说明带合法ID仍会错绑来源；API十二次生成与完整开发评分又保留了无引用及评审漏判。简单编号值得作为未来呈现对照，但这些论文没有证明本项目的长hash引用ID就是失败原因。多轮编号还必须保持旧QA中的引用身份，不能每轮从1重新编号、替换原始历史，或靠格式出现率宣称质量提高。

当前不因此加入新的生成模型、训练流程或改变默认引用策略。先完成正在进行的完整质量测量与强基线工程。若后续采用ReClaim式先引用后作答，应保留完整原证据与逐句支持位置，独立计入额外生成成本，并覆盖代码/表格/条件及错误拒答；现有存在性检查和judge高分不能替代这些检查。本次仅研究这些机制，没有复制或执行其源码。
