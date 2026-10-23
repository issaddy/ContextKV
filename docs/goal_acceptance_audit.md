# 完整工程目标的独立验收审计

审计日期：2026-10-24；基线提交 `70f0487`。按原《[完整技术方案](../llm_qa_context_kv_technical_plan.md)》第3–14章、[实施计划](implementation_plan.md)、质量准入、共享实验入口及真实报告核对。没有修改原方案、读取test答案调参或将模拟结果记成硬件结果。基线726项与6项子测试通过，只代表已覆盖工程合同。后续 `41d75c0` 的共享policy、SLO与失败退出修复通过82项及6子测试，后续82轮调度报告已记录显式SLO，非默认共享策略的独立硬件对照尚未完成；`5a009f4` 已实现固定LettuceDetect span适配，随后29/29真实模型调用完成；Granite在tokenizer返回类型修复后也完成29/29原生判分，但真实对象范围漏检等语义边界仍在，见[Granite验证](granite_judge_validation.md)。

**完整目标尚未验收完成，但核心工程不是空壳。** A/B/C、引擎补丁、数据、重放、标定和评测都有运行实现；H200证据已覆盖真实生成、两层缓存与有限容量。当前主要缺口是：可靠的完整答案评价、同条件强B6、B在压力下的净决策效果、C与整套方法的重复联合证据，以及最终冻结test与综合报告。A、B、C继续独立改进；近邻工作不构成收窄贡献方向的理由。

这里将“已证”限定为已运行的具体合同/配置，“弱证”表示实现和局部开发结果存在但不足以支撑总体效果，“未完成”表示尚无所需产物。负面结果是改进依据，不能删除，也不等于整条方向不可行。

后续 `3b852aa` 已通过804项完整CPU测试与6子测试；[普通EOS预算与闭环](native_budget_validation.md)也完成82轮，明确保留等待误差未改善、尾部变差和答案分叉。后续来源结构、mixed采样与BYOC是继续开发项，不能从这轮测试通过推断其结果。

## 一、模块实现与证据等级

| 模块 / 原方案要求 | 已证的工程与实测 | 尚缺的目标验收 |
|---|---|---|
| D：固定数据、在线白名单、P1/P2/P3 | 固定842任务/110会话；全域22 dev/88 test会话，Cloud为5/21会话、41/164任务。真实P2/P3只用自身已完成历史；query视图仅允许用户历史。数据/检索/完整input/receipt关联已多次逐轮核对 | test真实方法比较仍未完成；当前真实研究主要限于Cloud dev，第二模型或其他域关键点尚缺。这里只复算划分数量，未查看test问题/答案内容作选择 |
| A：双账本、共同required、版本/回访/条件闭包、稳定引用、APPEND/REBUILD | `controller/evidence.py`、`candidates.py`、`templates.py`和`adapters/corpus.py`覆盖无副作用候选、正文span、版本冲突、完整QA与溢出；真实输入重建通过。BGE、用户topic、四榜RRF、CRAFT来源焦点已主路径接线 | **弱证：** 原始benchmark是固定snapshot，跨版本与外部条件依赖主要由构造测试证明，尚未形成真实版本化专项结果；检索相关性/局部10MB与GUID改善没有完成整套回答质量验收。表格解析只对显式结构生成关系，不等同一般表格推理 |
| B：批量只读quote、完整成本、误差/滞回、控制成本 | 引擎实测quote、阶段拆分、候选长度、条件误差、收益模型与会话OOF/冻结双臂均存在。90 pairs、16K标定和多轮闭环保留了预测/实际/回退。最新原生预算与match年龄观测已H200核验 | **弱证：** Gate0相对Gate2改善，但仍比Fixed4096慢15.16%；候选regret仍不及固定REBUILD。等待/状态时效/候选长度误差未解决，不能把calibrated/eligible或模型MAE改善当成选择净收益。normal-EOS的ratio/clip边界尚未真实覆盖 |
| C：prepare/guard、成功事务提交、退役outbox/ACK、恢复 | 持久attempt/CAS/outbox、同会话串行、失败不提交半答案均有工程测试；H200 completion恢复、退休拒绝和GPU/host路径探针已跑通。ACK明确不声称即时free；16K真实host恢复已激活 | **弱证：** 受控探针证明软引用机制，真实六方法中Ours/no-retire同输入输出却未见时延收益。缺少更多版本/回访/工作集交互的重复配对及整套质量约束下结论；真实强制进程崩溃/网络故障的覆盖不等于所有CPU故障向量均已硬件验证 |
| E：固定SGLang适配与遥测 | 固定commit、Python FULL、BF16、GPU+host；只读报价、generation/restart guard、原生admission、实际DMA与pool采样已主路径运行；大池、32K、16K均有记录 | kernel/page_first段错误由兼容layer_first配置绕过，未声称修复上游；采样峰值不是allocator瞬时峰值。不同补丁/资源结果不能直接合并作同条件收益 |
| Q：质量来源绑定、独立judge、盲审与CI | 官方格式/导出、全visible audit、MiniCheck、原生Selene/Prometheus、GroUSE、原子诊断、盲审导入与双侧输入绑定均已实现；dev/模拟/错答案的旧漏检已有修复 | **未完成：** 实际五维合格评分/人工盲审和正式非劣。现有readiness五维每方法缺41分，`dev_quality_eligible=false`；原子v3格式修复后prose仍8 matched/5 mismatched/11 unknown，不能充当五维gate |
| X：配置矩阵、开发选择、匹配消融、冻结测试、统一报告 | 矩阵/继承消融/失败保留、会话cluster统计和test配置绑定有实现，模拟完整流程已跑；真实控制器11主配置+5消融共656轮 | **未完成：** 最新共同A、标定B、合理原生写策略下完整B0–B6与Ours比较。已有`runs/*/frozen.json`三个顶层示例均`simulation_only=true`且无质量gate；不能用于真实test选方法 |

相关真实证据分别见[工程验证](verification.md)、[成本双臂](stratified_decision_validation.md)、[稳定期消融](stage_gate_validation.md)、[有限容量联合](controller_capacity_validation.md)、[原生预算观测](native_observation_validation.md)、[最新融合P3](current_topic_fusion_p3_validation.md)。这些已运行工作不可仍列为“待开发”，但也不自动构成完整目标通过。

## 二、B0–B6是否构成足够强的基线

| 编号 | 当前实现/实际运行 | 公平比较还需补什么 |
|---|---|---|
| B0 | 未打研究补丁的原生prefix路径，完整QA+当前材料；fixture与Cloud P2实际成功 | 与最终A公共证据策略、输出预算、硬件初态和重复顺序匹配；最终质量和test仍缺 |
| B1 | stock session-aware+HiCache；32K/16K三种原生写策略各有41轮实测 | 三写策略各一次且答案/等待分叉，不能按恢复量或29ms差选“最强”；与B2同条件解释生命周期增量 |
| B2 | `normal_rag`每轮重建＋研究epoch退役，真实大池/16K/P3已运行 | 当前不是独立native B1的同初态配对因果证明；保留原生基线和各自完整开销 |
| B3 | fixed threshold 4096/8192、period2/4已实现且真跑；始终REBUILD也是有竞争力的简单对照 | 更新到最终同源A/B/C资源后重新调/评；不因其更快但未评分就宣布优质强B6，也不让它承担无用cost quote |
| B4 | ContextPilot排序/注释适配、完整QA、共同required和同等退役；大池41轮已运行 | 明示不是完整层次索引/跨会话聚类复现。强对照需在最终压力和A配置下重测；增加上游完整index属于可探索扩展，不是把现适配冒称完成 |
| B5 | ready-GPU比例warm/idle适配；.25/.75已真跑，.75实际触发两支 | 不是ReCAP完整图。明确host-only/pending信号边界，在有压力和回访的相同负载里检验；.25全APPEND不能证明有用的冷热门控 |
| B6 | 选择与冻结框架存在；当前controller示例显式不设任何`b6_eligible` | **尚未形成真实、质量合格的最强组合。** 应将已验证较强的上下文规则与原生write policy组成有限候选，通过共同dev质量/完整时延选择，再冻结。不能用模拟的B3_periodic别名或大池无host恢复结果代替 |

原方案第12.2节的冗余20/40/60%、token4k/8k/12k/16k、周期2/4/8是“dev起始搜索”，不是最低版本要求穷尽全部笛卡尔积。当前planner支持token/period/warm，但没有独立冗余比例策略，是**可补的强简单对照**，不是核心工程不可运行。

若补该对照，建议精确定义为 **旧的、在APPEND中可见、但本轮不在required中的完整证据正文token占比**：分子为这些旧非必需块的正文span token长度之和，分母为APPEND所有可见证据正文span token长度之和；QA、system、来源头、引用提示、table annotation不进入分子分母。正文计数使用固定tokenizer对原body的明确计量口径，不能把 `len(APPEND)-len(REBUILD)` 称为纯证据冗余。零证据时定义0；超过预设阈值仅在REBUILD合法时选它。该指标是文本冗余代理，不是实际KV可释放比例；最终prompt仍完整编码、不删required。以相同候选/退役/质量门比较有限0.2/0.4/0.6对照，或预先声明其中两点的等预算搜索；不读取未来或标签。

## 三、本次发现的可执行缺口

### P1：原生基线没有完整继承共享证据策略

审计基线中，`bench/experiments.py:281–287`仅把公共prompt、table样式、token预算传给native；`bench/native_baseline.py:400–403`重新创建CandidatePolicy，丢失共享context中的 `revisit_mode`、`revisit_history_turns`、`allow_version_comparison`、`explicit_keys` 和 `reference_prefix_length`。因此合法的公共base配置可能被native静默替换为默认值。

最小复现已运行，无模型/网络：`runs/goal-acceptance-audit/repro_shared_policy.py`直接调用实际matrix/native策略构造链并在加载基线处截获policy。共享 `revisit_mode=explicit`，档案只有旧timeout段，本轮当前段无关，问题为“`What about that limit?`”：controller的required仅`current`，native实际默认conservative的required为`current+old`。输出保存 `shared-policy-repro.json`。这会破坏原方案的共同材料条件；显式版本策略也可能导致一侧失败。

修复应直接透传完整共享EvidencePolicy，保留B4有意布局差异的声明；涉及required/版本合法性/引用映射的字段必须匹配。当前已运行的默认conservative/3/false/10均相同，本审计**不据此宣布历史默认结果无效**。后续 `41d75c0` 已完成完整不可变context policy透传与矩阵公共材料校验；同一最小案例独立复验后两侧required都仅`current`，记录在 `shared-policy-postfix.json`。原复现JSON保留70f行为，未覆盖。

### P2：goodput阈值原来没有进入结果和CLI

原 `bench/report.py:80`隐含TTFT=2s、TPOT=.1s，CLI没有参数，`bench/metrics.py`输出goodput却不输出阈值。Python调用者改变阈值后，单看summary无法知道其SLO。当前未宣称正式goodput收益，既有时延数值不受影响。

本审计已做最小修复：保留默认值，新增CLI `--ttft-slo-seconds` / `--tpot-slo-seconds`；候选与reference使用同阈值，`summary.timing.slo`记录 `ttft_seconds` / `tpot_seconds`，非有限或负值在现有summary边界拒绝。23项相关测试通过。后续 `41d75c0` 已将矩阵顶层 `slo: {ttft_seconds: 2.0, tpot_seconds: 0.1}` 接入现有spec provenance及报告入口。

该提交同时补了直接benchmark CLI的失败退出合同：此前native/controller保存failed report后仍可退出0；现在内部runner保留并返回RunReport，主CLI遇到失败或未完成会退出非零。原始失败记录继续保留，矩阵仍按实际summary计入失败；不是用进程退出码替代业务结果。三项修复合计82项及6子测试通过，尚未以修复后代码重跑相应硬件比较。

### P1流程缺口：质量门、强组合与冻结test尚未闭合

`bench.quality_acceptance`已做到原问题/历史/检索/实际input/答案/评分双侧绑定、simulation拒绝、同权重自评阻止准入、相同完整任务覆盖及会话bootstrap。`quality_noninferiority_established`还要求heldout固定协议与human_blind；自动模型只给evaluator-relative结论。审计未发现当前真实结果被错误标为通过，也不再重复已修复的三项旧漏检。

但目前运行记录没有完成这些评分。模型下载、judge JSON合法、rule无unknown citation和assistant逐例检查均不能填补缺分。`experiments.select_development`的质量gate是操作者提供的开发结果输入，不自动产生或认证分数；原来的模拟freeze与控制器子矩阵都明确未完成质量准入。最终需要实际冻结的评分协议、每个候选完整评分覆盖、保留误判/NA、明确人工/模型来源，再生成真实B6及test配置。

## 四、有限、可执行的剩余流程

1. **先固化已补的比较合同。** native共享policy、矩阵SLO透传与失败退出已在41d修复并有测试/最小案例核验；不要重复运行与之无关的大矩阵。统一最终使用的模型、输出上限、公共证据策略、retriever和资源清单。
2. **形成可用的开发质量测量。** 当前已存真实答案和完整audit，LettuceDetect与Granite均已完成固定29项实际诊断。Granite的22个事实控制吻合，但恰当拒答误报、错引未检和真实对象范围漏检仍存在；完整模型输入已独立核对，不应继续把它们列为待接入。应按诊断边界完善可测量的来源/条件协议，再对候选完整dev集合评分；不要只选择改善的case。模型诊断和人类盲审分别记录，覆盖代码、版本/对象条件、引用和可回答性。无需等待新增固定数量会话才实现，但5个dev会话仍是开发证据。
3. **让B的改进回答实际瓶颈。** 在normal-EOS压力任务核对原生reservation与服务长度、match年龄和tier漂移；保持固定REBUILD、Fixed4096与当前Gate0对照。用既有冻结pair/会话OOF比较等待或差值模型，避免只降低门槛。随后在同一资源上检查完整成本、错选regret和实际回答，不凭eligible率升级宣称。
4. **完成有边界的联合/B6。** 使用已激活host恢复的16K与一个较大容量对照，有限比较上下文规则×三原生写策略；配对会话顺序/停顿、轮换方法顺序并按原方案进行重复。独立保留C开关与共享旧前缀机制探针，计入控制/检索/传输/提交/排空。候选A允许继续改善材料；纯缓存消融则共享选定A。没有必要先展开全部维度网格才能形成可用结果。
5. **冻结后完成正式流程。** 由dev选定B6/Ours及匹配消融，冻结配置、SLO、输出/质量判据和评审协议，再执行Cloud test 21会话/164轮；test结果不返回调参。按会话给质量CI、按独立运行报告延迟/失败/容量趋势，补全域或第二模型关键点。一次明确冻结后的失败或无收益也要报告，而非改名dev重选。

最终交付应能链接：完整源码/许可与部署 → 全量协议/故障合同 → dev选择与失败区域 → 冻结test原始记录 → 五维质量/完整响应/SLO/资源联合表及消融。**不能以“所有测试绿”结束，也不以“必须每个指标提升”制造无限任务。** 达到可复现、质量可接受且作用区域清楚的整体工程结果后，按真实证据组织A/B/C贡献。

## 五、Granite Guardian 4.1原生接入核查

独立核对的固定源为模型 `ibm-granite/granite-guardian-4.1-8b@ab01ccca5dcfb80246369a086a4a87a29198f5af` 与作者仓库 `ibm-granite/granite-guardian@0a4ec708fa916f3e9d7fc7e5b75dff836752f448`，Apache-2.0。它基于Granite 4.1而非当前Qwen回答器，作为独立风险诊断已完成固定开发探针；当前原生协议仍遗漏真实长答案的对象范围缺陷。来源与模型清单也见[专用判别器计划](quantity_judge_reuse.md)。[固定模型卡](https://huggingface.co/ibm-granite/granite-guardian-4.1-8b/blob/ab01ccca5dcfb80246369a086a4a87a29198f5af/README.md)、[作者固定README](https://github.com/ibm-granite/granite-guardian/blob/0a4ec708fa916f3e9d7fc7e5b75dff836752f448/README.md)。

最小adapter不套Selene五维1–5或Atomic JSON。输入保留实际history/current user、最后一个待评assistant，全部visible原文仅通过官方 `documents=[{doc_id,text}]`注入；末user添加 `<guardian>` 与think/no-think模式、criterion、scoring schema。实际实现直接复用 `NativeSGLangClient` 的 `/generate(input_ids)`，不提供session ID、不重复套chat template。记录完整IDs、既有evaluation binding、criterion与极性、finish reason/usage/原输出；不要将此前assistant句子放进documents。官方模板确实支持documents系统区；权重config窗口为131072，示例8192是服务设置，实际部署需要明确预算与完整输入检查。[固定模板](https://huggingface.co/ibm-granite/granite-guardian-4.1-8b/blob/ab01ccca5dcfb80246369a086a4a87a29198f5af/chat_template.jinja)、[固定config](https://huggingface.co/ibm-granite/granite-guardian-4.1-8b/blob/ab01ccca5dcfb80246369a086a4a87a29198f5af/config.json)。

解析只接受正常完成的原生 `<score>yes/no</score>`；groundedness的作者criterion描述的是**不受来源支持的风险**，所以yes=风险存在，不能当作正确/支持。缺score、截断、上下文超限保持未覆盖；think轨迹与最终判定分开保存，不要求输出JSON或伪造概率。先原生风险判别，再独立探索BYOC对象/数值条件；未标定的二元判定不能自动填满五维quality gate。

本次还发现固定HF模型卡的某个`build_guardian_block`代码段漏写`<guardian>`，但该卡格式节、固定作者README/helper均有它，tokenizer模板也不会自动补。因此依据完整格式与作者helper实现渲染合同，避免复制该省略示例。薄适配已落在 `bench/granite_judge.py` 并附Apache归属，未复制HTTP框架或改动Lettuce/Atomic；18项专项与65项相关测试（另6子测试）通过。默认no-think、64输出，显式think与预算可配置。完整29项真实执行已完成：v1在HTTP前因tokenizer返回类型失败，`856a753`修复后v2全部正常结束，29/29输入逐token重建与wire一致。22个事实控制匹配不代表整体质量，真实对象范围/性能保证仍漏检；两版失败/输出均保留。用法见[原生适配](granite_judge.md)，完整事实、拒答、代码与真实例边界见[真实验证](granite_judge_validation.md)。
