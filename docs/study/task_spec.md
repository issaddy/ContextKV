# ContextKV：课题目标与实施范围

课题：面向资源受限多轮文档问答的上下文管理与 KV 缓存协同优化研究。

## 一、背景与目标

多轮问题持续改变主体、条件和版本，长资料与多跳任务需要多次阅读。窗口、累计计算和 GPU/Host 缓存容量共同限制系统能使用的证据。

固定 Qwen3-32B 权重、tokenizer、精度和生成参数，联合优化证据上下文、阅读计算及来源 KV 复用。在共同资源预算下，最终质量提高 5 个百分点，或同质量下完整响应 P95 降低 15%。按数据集报告质量、交付、延迟、token 与实际资源。

## 二、模块职责

| 模块 | 输入与决策 | 输出 |
|---|---|---|
| 任务与证据上下文 | 当前问题、实际历史、条件、版本和资料范围；选择适用来源与已有事实 | TaskState、EvidenceSet、规范来源窗口、事实复用合同 |
| 预算化阅读与计算 | 已有事实、证据缺口、剩余预算和 Native 报价；选择 READ、REDUCE、GAP 或 ANSWER | 完整事实与结论、来源父链、最终回答 |
| 请求编译与缓存协同 | 实际候选输入、未来原文消费者和 GPU/Host 状态；选择来源排列与需求短租 | 已报价请求、保留变更、原生回执 |

统一入口为 `DocumentQA.ask`。HTTP 问答和 benchmark 共用 Controller、SessionStore、请求编译器与执行器。[架构](../architecture.md)

## 三、核心机制

- Need 显式关联主体、条件、版本与前驱，条件更新只改变对应需求。来源选择、事实复用和缺口检索使用相同适用范围。
- 原文窗口由版本、坐标和实际 tokenizer 定义。稳定原文放在变化的历史与任务之前，READ 与原文 ANSWER 共用来源格式。
- 默认按实际检索排名和依赖闭包选材。当前 retriever 配置使用 dense 阶段与查询适配器；混合召回、发布版 reranker 和支持模型分别由配置启用。
- 先扣除 ANSWER 的系统、任务、历史、来源与输出，再分配结论和 READ/REDUCE 份额。完整结论和来源链进入实际回答窗口。
- 同来源、版本与适用范围共享事实提取，各 Need 分别选择群体和度量。CPU 执行去重、单位换算及 count/sum/group，模型执行语义提取、选择和必要归约。
- 同会话原子事实按当前需求复用。完整历史覆盖直接扣除已完成阅读；适用的显式记录可从当前原文完整投影，其余材料继续 READ。
- 动作质量目标为 `Q(首动作 → 固定有界后续 → ANSWER) − Q(当前直接 ANSWER)`。质量头按决策状态等权拟合，成本头拟合唯一真实分支的绝对费用。线上执行首动作后重新观察。
- 候选先编译后报价，选中后提交同一输入。上下文组织和回答表示投影后续 READ/ANSWER 原文需求，C 据此保留可恢复前缀；后续真实报价反馈给 B。
- Native Radix/HiCache 管理物理缓存。需求短租考虑 GPU/Host 容量与可淘汰区间，共享页按并集占容量，未来消费者的同一因果页只计一次。租约估值用于分配，动作费用使用 Native 报价。

实现细节与公式见[架构](../architecture.md)，复用范围见[方法依据](../references.md)。

## 四、辅助学习

Qwen3-32B 保持固定。训练按已定位的瓶颈选择：

| 对象 | 方法与用途 |
|---|---|
| BGE 查询侧与任务头 | 查询侧 LoRA、任务关联监督；文档编码器和向量冻结 |
| 支持与集合效用 | 支持关系监督、真实候选回答偏好；用于 support 选材对照 |
| 动作收益与成本 | 状态等权小回归树、非负成本回归；线性收益头作为对照 |
| 缓存价值 | 同轨 Native/TTL 配对估值；作为需求短租的可选分配器 |

默认使用 teacher 任务解释、ranked 选材与 `next_use` 需求分配。任务学生尚未输出跨轮 READ 复用合同；查询资产可通过 `retrieval.task_query_asset` 单独接入。辅助训练按会话划分 train/dev，200/40 是配置选项，实际规模由有效数据与待解决问题决定。

执行输入不变的既有动作配对可以 CPU 刷新预算特征并重拟合；输入或动作变化时更新对应真实分支。[运行入口](final_runbook.md)

## 五、数据与评价

| 任务轨 | 输入与历史 | 配置 |
|---|---|---|
| 公开 benchmark | 作者原题、参考答案和参考历史 | `configs/study_matrix.yaml` |
| 多轮系统重放 | 原问题及各方法实际提交的回答 | `configs/study_system_matrix.yaml` |
| 构造机制任务 | 条件/版本与长文统计的独立开发任务 | `configs/study_mechanisms.yaml` |
| MTRAG-UN 诊断 | 选中原问题、作者历史及给定材料 | `configs/study_mtrag_un.yaml` |

公开主比较包含 MultiDoc2Dial、MT-RAG Cloud、VersionQA 和 Loong Set1。每个 benchmark 独立选题、评分与报告；下一轮各选 50 题，同一 benchmark 内各方法共用 task ID 清单。实际历史重放按会话顺序选择连续任务。

MultiDoc2Dial 用 F1/EM；MT-RAG 用作者 MTRAGEval；VersionQA 用作者二值 rubric；Loong 用作者百分制 rubric、Perfect Rate 及领域/难度分组。评分记录实际 judge。题目分母来自选中清单，完成率、未评分项和已付成本随结果报告。[数据与评价](../data_and_evaluation.md)

## 六、实验与验收

主比较为普通文档问答、ContextKV 和 LlamaIndex。普通问答使用必要历史、检索重排、窗口内拼接、SGLang Native 和最终回答；LlamaIndex 使用上游历史问题凝练及 compact/refine。各方法的辅助模型与调用开销计入自身在线成本。

默认并发 1/4、每种方法运行一次，按数据集分批推进。会话内等待前轮完成、零思考间隔，会话间受共同并发上限调度。已有真实固定到达记录时复用；每个方法及数据集起始清缓存，轮间保持缓存。

| 对照目的 | 比较方法 | 固定能力 |
|---|---|---|
| 整体效果 | 普通文档问答、ContextKV、LlamaIndex | 问题、历史协议、生成条件、资源与评分 |
| 缓存感知编译 | `context_native` → `context_cache_order` | 任务、选材、事实、阅读决策与回答表示 |
| 需求保留 | `context_cache_order` → `context_cache_demand` | 上述能力及 ContextPilot 编译 |
| 动作收益 | `Rule_contextpilot_joint_demand` → `context_cache_demand` | 选材、动作域、表示、编译与保留 |
| 跨轮事实复用 | `context_no_fact_reuse` → `context_cache_demand` | 实际历史、本轮共享提取及其余模块 |

机制对照按结果需要选择，复用同批完整方案结果。固定输入的缓存配对解释正文命中、回载、重算与控制开销；单数据集队列评价最终质量和完整响应 P95。各条件贡献分别解释。

0–1 质量指标提高 0.05 为 5 个百分点，Loong 在原 0–100 尺度报告差值。完整响应从真实到达计至回答完成，包含排队、检索、任务解释、阅读、归约和控制。在线 token、GPU/Host 物理积分及 unique KV 页分别报告；训练、索引和离线评分费用单列。

交付各 benchmark 的质量、完成率、完整响应 P95、在线 token、GPU/Host 占用，以及少量缓存对照指标和实际方法配置。执行命令见[运行手册](final_runbook.md)，实测入口见[结果](final_results.md)。
