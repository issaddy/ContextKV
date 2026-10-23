# 较新工作调研：原理、适用价值与实际采用

核查日期：2026-10-23。本轮检索覆盖 2025–2026 年的缓存感知上下文管理、文档 chunk KV 复用、成本/SLO 调度和多轮淘汰；以 2024 年 RAGCache/CacheBlend 等作为路线起点。使用作者 arXiv 正文、ACL/MLSys 官方论文页、作者明确关联的仓库和 LMCache 官方文档。查询包括 `cache-aware context compaction`、`prompt compression cost`、`RAG chunk KV reuse`、`adaptive reuse planner`、`agent prefix eviction`，并沿相关工作查近邻。

这是围绕可实现机制的定向调研，不是穷尽所有论文的系统综述。核心项目源码固定版本见[相关工作记录](related_work.md)，A/B/C 的主张、实验与不足见[贡献论证](research_contributions.md)。不因“较新”而默认方法更好；没有直接核查正文/代码的搜索摘要不作为性能依据。

2026-10-24 的[独立补充审查](research_review_20261024.md)核查了新的直接近邻：SmoothAgent 已有变换准备/提交分离及 Prefill/Decode 成本分解，AKV 提供 drop/repos 缓存工程，Context Language Models 的 SCR 可复用编辑后的后缀 KV。其机制、作者固定源码与许可证均已单列。当前只采用适用的建模和验证思想；AKV 的 page size 1 / 排除 HiCache 与 session，以及 SCR 保留旧前缀信息的状态语义，均不兼容直接替换本项目的固定精确前缀路径。不能据此宣称这些路线无价值，也不能把通用成本公式和生命周期协同当成本项目首创。

## 新近邻及其对主线的影响

| 工作与官方来源 | 原理与实际运行边界 | 对 ContextKV 的价值与取舍 |
|---|---|---|
| [CacheSmith，2026-10](https://arxiv.org/html/2610.18925v1) | 将部分重算拆成 WarmUp/Probe/Patch；用离线成本/质量分布、运行时带宽及剩余预算配置重算，并以队列预算组织 batch。正文 §6 使用 vLLM 0.10.2、Triton、RTX PRO 6000，包含部分 KV 分配 | **直接近邻，收紧 B 的新颖性。** 转移感知成本、离线标定和轻量在线计划都已有工作。可采用少量标定区间、预算/误差意识和负载条件报告；不移植重算 kernel、逐层执行或 partial allocation。当前选的是完整 APPEND/REBUILD 输入，保留原生精确前缀与完整回答成本。正文未提供本次可确认的官方仓库，不称代码复用 |
| [SpecCache，ACL 2026](https://aclanthology.org/2026.acl-long.859/) / [正式 PDF](https://aclanthology.org/2026.acl-long.859.pdf) | 用较小推测模型的深层 hidden-state norm 指导目标模型的关键 token 重算，针对浅层或固定规则挑选 token 不准的问题 | **另一条 chunk 修复路线。** 只有非前缀 chunk 复用成为主瓶颈时值得进一步研究。需要额外模型、选择器成本和目标引擎选择性重算，不是拿一个 Python 排序函数就能接入。当前不新增推测模型，也不把论文相对 full recompute 的结果视为相对本项目的优势；未确认作者代码入口 |
| [Grounded Cache Routing，2026-05](https://arxiv.org/html/2605.27494v1) / [作者仓库](https://github.com/syedhumarahim/grounded-cache-router) | 对**答案缓存**使用 query/evidence/version/support 门控；构造 near-miss 与 document-drift 等场景。固定核查提交 `2fbe16e09fee7994b8e0069e87086c912f2adc0d`，README 声明 CC BY 4.0 | **采用边界与测试思想，不加答案缓存。** 同配置名/相似问题不代表相同资料，版本和检索作用域应先明确。当前已据此完善 A 的锚点作用域和相似但不同来源测试。既有身份 hash 继续只在入口计算，不复制多层签名；不采用其词袋覆盖率作为蕴含或正确性证明 |
| [CAPC，2026-07](https://arxiv.org/html/2607.15516v1) | 用实测 API 命中率/读写价格建立缓存与压缩的交叉成本模型；AdaptiveCacheBoundary 按历次版本变更识别稳定区域 | **采用成本交叉与边界扫描实验方法。** API 金额/命中模型不等于本地 GPU/host 驻留模型。不能把“按成本权衡长度与缓存”写成首次提出；不迁移其 API 阈值、正文归一化或语义压缩 |
| [Don’t Break the Cache，2026-01](https://arxiv.org/html/2601.06007v1) | 在长程 agent 上比较缓存边界，讨论动态内容置尾和全量缓存的不同代价 | **已采用稳定/动态分离。** Question 置尾是工程修正，和 ContextPilot 排名注释共同使用；不列为独立创新，不用随机 UUID 破坏 prefix 来弱化主基线 |
| [RR-Evict，2026-09](https://arxiv.org/html/2609.32278v1) | 观察同一轨迹节点的 recency 同步，轮转回收 idle 轨迹尾部，使多个活跃业务会话保留部分前缀；正文报告 SGLang 实现并比较 completion-aware LRU | **采用尾部与分布诊断，暂不重写淘汰。** C 处理已被新版本替代的旧 epoch；RR-Evict 处理仍会回来的 idle 轨迹，二者问题不同。先检查每会话 uncached tokens/TTFT 尾部、轮间停顿和压力分布，再决定是否移植。未确认官方 repo，不称已复用其实现 |
| [When Fancy Eviction Fails，2026-09](https://arxiv.org/html/2609.28870v1) | 在生产 trace 和模拟器中研究多种淘汰策略，强调 recency、非均匀计算代价与容量相关的管理粒度；作者表述会发布 trace/工具 | **采用“先证明需要复杂策略”的标准。** 主实验保留调优的原生 recency 基线；报告实际重算和完成时间，避免只优化命中率。大 host/低压力时可能无增益，这种结果必须保留。当前不声称获得其未确认发布的 trace 或代码 |

CacheSmith 与 SpecCache 都是**部分重算/近似融合的实际计算路径**，不是普通 prompt 排列；GroundedCache 是**输出答案复用**，不是 KV 复用。这些差别决定了不能直接把各自收益相加，也不能仅按论文标题认为它们已被本项目覆盖。

## 五项最值得落地的机制

### 1. 保留排序语义的稳定布局：已实现

来源为 ContextPilot 的对齐/排名与位置注释，以及稳定/动态内容分离的既有工作。`adapters/contextpilot.py` 保留共享前缀顺序；`templates.py` 将原检索顺序通过稳定引用映射放在正文后、Question 前；`CorpusIndex.retrieved_keys` 把 top-k 与追加条件闭包区分。历史消息和正文头不被动态排名改写。

这是原理适配，不是整个官方层次索引的复现。已验证两布局共享 required/full QA、原始排名不丢、支持材料不获得假排名、固定 Qwen 真实 token LCP 包含共享正文。后续用真实回答验证排序/注释是否影响质量，不能仅凭 token 前缀认为收益成立。

### 2. 面向资料漂移与近似指代的恢复作用域：已实现小修

[作者固定 `validator.py`](https://github.com/syedhumarahim/grounded-cache-router/blob/2fbe16e09fee7994b8e0069e87086c912f2adc0d/src/ragcache/cache/validator.py)、[`workload.py`](https://github.com/syedhumarahim/grounded-cache-router/blob/2fbe16e09fee7994b8e0069e87086c912f2adc0d/src/ragcache/workload.py)启发的是“相似输入但资料不同”的测试维度。没有复制答案缓存路由：其 lexical support 只是词袋代理，源码还移除否定词和较短 token，不能作为技术文档条件/数值正确性的保证。

对当前 A 的具体审查发现：多个产品都含 `connect_timeout` 时，单凭相同锚点可能恢复无关旧资料。已在 `controller/evidence.py` 修正为：优先显式文档/版本名，否则限制到当前检索匹配的文档版本作用域；无当前依据且多个历史来源同名时不猜。显式稳定引用仍可恢复指定资料，条件闭包继续保留。新增跨产品 near-miss、历史歧义、指定版本测试，不引入新 hash、嵌入模型或在线 judge。

这只改善可解释的资料恢复边界，不保证自然语言指代都能解析。真实 MTRAG 的质量增益需单列验证，不与布局收益混淆。

### 3. 以标定区间支持轻量成本选择：已实现并优化控制开销

CacheSmith 的成本带和快速在线选择，以及 EfficientAgent 的恢复/重算分析，支持先做可解释的小动作空间。当前两候选、GPU/host 报价、测量覆盖回退、输出长度先验和误差门槛是该思路在本项目的适配。新增研究重点应是两候选成本差是否能判对，而不是继续扩展动作数量。

审查后已优化并重新测量：252 个 Decode 标定点、输出均值 128/P90 256 时，单候选预测由 38.22 ms 降至 0.58 ms；这是本机 CPU 开销，不是 GPU 加速比。实现预编译固定标定表、优先直接计算完整插值单元，仅在稀疏/越界时排序近邻，区间观测直接命中；没有增加近似积分或无界缓存。随机点、区间、边界和完整预测的等价性测试通过，保留原成本/误差/支持范围。正式切换门槛仍须使用目标环境的控制 P95。

已补充[离线成对决策入口](decision_pairs.md)：从开发集 pre-turn 账本冻结两候选，每臂独立清空并预热同一历史，记录差值误差与选错代价。H200 已运行 3 个早期场景的 9 对诊断，7 次滞回、2 次回退，尚不能证明成本切换收益；后续按会话和版本年龄分层抽样。phase 留出测试不能替代这组结果。该机制不要求复制 CacheSmith 的逐层调度器。

### 4. 只在实际压力下评价生命周期效果：测量与消融优先

RR-Evict 提醒关注“谁被淘汰、谁在尾部变慢”；When Fancy Eviction Fails 和 EfficientAgent 提醒关注容量与实际计算节省。当前 C 应先与原生 session/recency 及无 C 路径比较，保持共同资源预算、同动作轨迹的受控实验，再做完整闭环比较。

可落地增量是每会话 uncached-token/TTFT 分布、停顿序列、旧独有分支与共享祖先的区分，以及三种原生 host 写策略的开发集选择。只有观察到 idle 活跃轨迹反复被集中清空，才值得尝试 round-robin tail；只有观察到无效 host 写入，才值得迁移条件准入。当前不同时加入这两项引擎策略，以免无法解释收益来源。

### 5. 质量/负载条件下的可用区间：采用评价方法，在线不读真值

CacheSmith 的质量与时延联合目标、CAPC 的交叉区域分析，可以转成可执行的开发集筛选：先冻结质量可接受配置，再比较不同负载/容量/输出长度下的完成时间和 SLO。现有 judge、盲审、配对会话区间、`quality_eligible` 与完整失败保留支持该流程，但需要真实生成评分来填充。

本项目在线选择不使用当前 gold、reference 或真实答案质量。质量门槛在离线按方法/配置验证；若报告逐请求 quality-gated SLO，只能作为离线评估指标，并保留未达标请求在总分母中。当前不复制 CacheSmith 的逐请求重算质量概率，也不将便宜的词法覆盖当作质量门槛。

## 当前结论和下一步顺序

可明确采用的主线仍是：SGLang 原生精确前缀/两层缓存 → ContextPilot 风格组织与证据约束 → 轻量、实测驱动的两动作计划 → 成功提交后的 epoch 生命周期。上述调研已带来排名注释、top-k/依赖区分、资料锚点作用域纠偏和成本预测开销优化建议。

下一步按证据优先级推进：目标设备标定与动作覆盖 → 调优强基线/单因素消融 → 真实质量和多会话容量实验 → 依据观察决定是否扩展 host 准入或 idle 轨迹淘汰。部分重算、小模型 token 选择、语义答案缓存与多机调度均保留为独立路线，不以“最新方法”名义未经验证混入主工程。
