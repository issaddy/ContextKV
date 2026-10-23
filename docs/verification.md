# 工程验证记录

日期：2026-10-24（Asia/Shanghai）。本地为 macOS arm64 / Python 3.12.13；真实引擎部署于用户提供的 Linux 容器 /home/relay/songjiayun/ContextKV。执行成功与研究收益分开记录。

## 本地工程证据

| 验证 | 实际结果 |
|---|---|
| 最近完整 CPU 回归 | 667 passed，6 subtests passed，已显式开启真实 tokenizer 项，无跳过；runs/pytest-atomic-focus-final.log |
| 固定 Qwen tokenizer 专项 | 原生完整编码、批量编码、边界回退与冻结候选 IDs 等价通过，包含在最新完整回归中；Linux 30 个真实候选也逐 token 相同 |
| Ruff / shell / wheel | Ruff 通过；wheel/sdist 构建通过，已逐字核对 atomic_judge 模块/协议资产、table_focus、CRAFT 行文本与 GroUSE 源码，CRAFT 完整许可在 wheel 中；runs/build-atomic-focus.log |
| 开发集模拟矩阵 | 33 jobs / 99 轮成功 |
| 冻结模拟测试矩阵 | 39 jobs / 117 轮成功 |
| 继承主方法参数的开发集消融 | 15 jobs / 45 轮成功，不参与主方法选择 |
| 官方 MTRAG 格式检查器 | 6 轮 fixture，errors / warnings 为空；只证明格式兼容 |
| 固定 MTRAG 全量复算 | 842 任务 / 110 会话；Cloud 205 / 26；1053 处参考标记 |

矩阵证据在 runs/matched-dev、runs/matched-test 和 runs/matched-ablation-dev。此前 48-job 开发矩阵属于未匹配消融参数的旧版流程，不再作为最终方法对照。runs/official-checker/verification.json 明确使用 fake fixture 答案，不是质量结果。上游 Starlette TestClient 有 HTTPX 弃用提示，不影响通过。

复核使用 .venv/bin/python -m pytest -q 和 .venv/bin/ruff check .；真实 tokenizer 专项设置 CONTEXTKV_TEST_HF_TOKENIZER=1。新增模块的定向测试记录应在下一次完整回归后统一更新总数。

## 已完成的 H200 集成验证

单张 H200、TP=1、Python TreeCore、BF16 权重/KV、page size 64、context 32768、最多 8 个运行请求。实际环境：Python 3.12.3、torch 2.14.1+cu130、CUDA 编译目标 13.0、transformers 5.19.0、tokenizers 0.23.2、驱动 595.58.03。GPU KV pool 为 121920 tokens；host 为 279040 tokens，即 16001269760 字节（约 16 GB / 14.90 GiB）。固定源码的 hicache-size 采用十进制 GB，并按 page 对齐。完整环境保存在 engine-environment.txt。

SGLang 固定 3f36fc08331f64cd63c4fbbdf39cc402f38bb38b，Qwen2.5-7B-Instruct 固定 a09a35458c702b33eeacc393d103063234e8bc28。以下初次 GPU 证据使用补丁 49355b997267b7c88ca7182d53ab4846109d13bf7064d1e179b729fc444362a5；后续补丁须分别验证，不能自动继承结论。

| 验证 | 实际结果 |
|---|---|
| GPU-only 模型协议冒烟 | 完整生成/结果恢复、重复报价、退役后拒绝请求、ACK 不虚报物理释放，均通过 |
| kernel + layer-first host 冒烟 | 通过相同验收 |
| 冷/GPU-ready/host-only 小标定 | 128 输入、8 输出、prefix 0/64、并发 1、每单元 3 次；18 条请求阶段与 12 条完成传输观测 |
| 小样本 P2，GPU-only | 6 成功 / 0 失败 / 0 未完成 |
| 小样本 P1、P3，host 启用 | 各 6 成功 / 0 失败 / 0 未完成；P3 使用真实 BM25S |
| Cloud 开发集 P2，host、并发 4 | 41 成功 / 0 失败 / 0 未完成；512 输出上限下 21 轮触顶截断 |
| 完整 Cloud 语料 P3 开发集，1024 上限 | 固定 72442 个原始 passage，BM25S 冻结索引；41 成功 / 0 失败 / 0 未完成 |
| 未补丁原生 B0 / B1 | 分别独立固定源码启动；各自 fixture 6 轮、Cloud P2 开发集 41 轮均成功 |

本地镜像证据在 runs/remote-h200-initial，包含原始报告、manifest、观测和失败/成功日志。容器导出使用明确文件白名单并核对归档 SHA256，没有导出代理配置或权重。

后续补丁 69bf539f582ad7065d1b16c6fa6efe348c3e69d9d4aec8a296f3206996b76912 已在相同 H200 配置通过 runtime_manifest、协议冒烟、暖机小标定及生命周期配对探针。受控 GPU/host 两层实验均显示：保留旧 epoch 时旧/新/共享路径为 256/128/128 tokens；退役旧 epoch 后在相同原生 128-token 淘汰压力下变成 128/256/128。退役瞬间两个 pool 的占用变化均为 0。该 fixture 故意在新路径后再次访问旧路径以隔离软引用作用，commit 边界由工具模拟；它证明限定配置中的机制，不证明一般工作负载吞吐收益。完整证据在容器 runs/lifecycle-paired.json，解释见 [引擎验证](../engine_patches/README.md)。

2026-10-24 从 5 个开发会话按版本年龄 0/1/2+ 分层冻结 15 个 pre-turn 案例，观察到 APPEND 输入最大 11217 tokens，超出此前 8192 的采样上限。因此在相同引擎配置追加 16384 长度标定：prefix 比例 0/.5/.9/.99、并发 1/4、输出 128/512，28 个兼容 case，每 case 1 次 warmup 和 3 次测量，共 70 个预热请求、210 个测量请求、778 条阶段/已完成传输记录。原 1500 条记录独立保留，补充数据在 runs/remote-h200-expanded；扩大覆盖本身不是方法收益。两个 estimator 的支持域已共同加入实测总长度边界，phase 别名按同一预测特征分组，模拟拟合来源不能被加载为硬件校准。

随后 15 cases × 3 缓存状态 × 2 次重复共 90 对全部完成，82/90 双臂处于标定支持域。Lookup 6 次成本切换，平均单步 regret 366.55 ms；同一观测上的 positive 离线重评分未切换，regret 437.05 ms。固定 REBUILD 为 129.14 ms，8192 阈值为 177.11 ms；当前结果不支持 B 优于简单策略。84/90 对的 Decode 差主要来自输出长度，180 个臂中有 18 个截断；这是并发 1、冻结历史下的诊断，尚无质量调整或闭环基线结论。详见[分层决策验证](stratified_decision_validation.md)。

**运行限制。** 原生 kernel + page_first 在首次 D2H 出现段错误，栈为 staged HiCache → cudaMemcpyBatchAsync / cuMemcpyBatchAsync_v2。保持固定源码和 host 语义，改用已有 layer_first 并设置 SGLANG_HICACHE_TMA_TRANSFER=0 后通过上述验收。尚未隔离具体驱动/运行时原因，不声称修复上游 kernel。正式比较必须统一兼容配置并重新标定。

**评测限制。** Cloud 512-token 配置有 21/41 轮截断，成功 receipt 不等于完整优质答案。按技术方案单列 1024-token 敏感性后，41/41 成功、0 截断、平均输出 536 tokens；记录于容器 runs/cloud-dev-p2-output1024。该运行没有控制与前一次相同的缓存初态，只比较截断现象，不能宣称时延改进。正式对照应给所有方法相同输出上限，512 原始结果保留。首次小标定含初始化影响，仅证明管线可运行；正式采集需独立 warmup、充分范围与留出误差报告。

## 已验证的控制开销改进

新增共享提示词开发验证共六组，每组固定 41 轮：Q0/Q1/Q2 分别运行 P1 与 P2。Q1/Q2 减少截断或缩短回答，但完整原文复核仍发现跨产品错答和错引，未选为默认，也未将输出变短算为质量保持下的加速。组合 ID 引用的实际漏计已修正；所有已知 ID 均存在仍不代表支持论断。完整自动统计、24 条定向论断审查及复算入口见[共享提示词验证](shared_prompt_validation.md)。

证据身份只在入口计算一次：100 个已归档证据＋1 个新证据的合成 REBUILD 中，正文 hash 次数从 304 降至 1。实际 token 绑定、请求幂等和 CAS/epoch 约束保留。

252 个静态 Decode 标定点、输出均值 128/P90 256 的本机 CPU 微基准，单候选中位预测耗时由 38.22 ms 降到 0.58 ms；区间案例由 15.55 ms 降到 0.46 ms。原插值、误差与支持范围语义通过随机稀疏/密集、重复点、区间、边界及完整预测等价性测试。证据为 runs/cost-profile/comparison.json；这是 CPU 开销改善，不是 GPU 端到端加速比。

15 个真实长会话快照的局部 CPU profile 显示重复类型注解解析和标量通用遍历可避免；只缓存静态 schema 并增加标量快路径后，临时 WAL/FULL 部分存储链的等案例均值从 47.22 ms 降为 13.46 ms，JSON 字节和 typed contract 保持一致。候选构建的哈希本身占比很低，没有因此删除证据校验。该局部结果不包含 RPC、并发等待和全部事务成本，不能换算为远端控制 P95；详见[控制路径诊断](controller_control_profile.md)。

远端 P1/always_rebuild/c4 的一次 before→after 对照均 41/41 成功，全部输入 IDs 相同，control P95 为 456.11→168.68 ms；16 个同输入/输出样本的 control P95 为 378.55→157.07 ms。P2 对照虽然也观察到控制耗时下降，但只有 6/41 输入相同，历史与动作变化存在混杂。两组均非随机交叉重复实验，P1 输入仅 2460–4231 tokens；不外推长 APPEND、质量一致或一般吞吐增益。原始数据在 runs/remote-h200-control，逐项校验与统计见同一控制路径文档。

候选联合分词进一步复用本次调用中相同的特殊 token 边界片段，保持完整文本及 IDs 等价；本地完整构建 13.31→4.63 ms。Linux 四线程分词局部 wall 22.61→7.58 ms、CPU 22.49→17.44 ms；默认大线程池则 CPU 增加，不能泛称减少算量。源码依据、逐 token 验证和线程预算见[候选分词测量](candidate_tokenization_profile.md)。

## 新增方法与真实开发验证

- 候选长度、条件误差、会话留一训练及离线 planner 重评分已实现。15 cases / 5 sessions / 90 pairs 的联合探索中，conditional＋candidate OOF regret 为 270.30 ms，低于原 global 437.05 ms，但仍高于固定 REBUILD 的 129.14 ms。完整 GPU P2/c4 集成 41/41 成功；该运行使用 full-fit dev artifact，不是 OOF 验证，且没有触发成本节省切换。见[长度与决策验证](length_decision_validation.md)。
- 原硬件控制器矩阵 11 jobs 和匹配消融 5 jobs，共 656/656 成功。所有实际 host 恢复为零，退役 ACK 不虚报物理释放；该结果不能证明 host 项或容量竞争收益。逐轮机制诊断见[硬件矩阵分析](hardware_controller_matrix.md)。
- 固定 MiniCheck 在 H200 上完成 Q1/Q2 各 41 个完整 audit 的评分。已修复把引用 ID 当事实输入产生伪负例的问题；两个高置信引用替换候选仍经原文审查发现部分支持及段尾绑定歧义，因而默认改为只增补来源。分句/未覆盖/评分与修正局限见[assistant 审查](grounding_v2_assistant_audit.md)，不计为人工盲审或正式质量通过。
- 原始 72442 passage 的 BGE-M3 索引与三种 query 视角、四阶段检索均已在 GPU 执行。39/41 个有 qrels 的开发问题上，last3/BM25 Recall@5=0.2094、nDCG@5=0.1671；current/hybrid_rerank=0.5231、0.4833。原文、候选池、模型版本、时长和全部阶段结果保存在 runs/remote-h200-retrieval。该提升属于已观察开发集的检索相关性，不等于回答质量或缓存收益。
- 嵌套会话留一的增益模型将候选长度分支 regret 270.30→236.14 ms，但决策与简单的稳定期内支持域门控完全一致，尚无细粒度学习增量。见[增益校准验证](gain_calibration_validation.md)。
- P3 BM25 与 BGE 增强各 41/41 成功，检索 ID/次序与离线结果一致；增强分支检索均值 4.94→699.26 ms，完整轮均值 2.546→3.624 s，且回答仍有事实/代码错误。纠错查询又完成 41/41，取回 10MB 附件限制表，但回答仍错误合并文档与附件限制。详见[P3 集成](p3_retrieval_integration.md)、[纠错查询](correction_query_view.md)和[代码审计](code_audit.md)。
- 固定 Qwen3-Reranker-0.6B 在同一 41 问题、2050 个候选对上真实评分；无截断或接口对接错误，Recall@5/nDCG@5 却降至 0.4085/0.3907。因此保留 BGE 为增强分支选择，负面结果及未标注替代相关段落一并保留。见[较新重排器验证](qwen3_rerank_validation.md)。
- 原生三种 HiCache 写策略分别完成 32K 和 16K GPU 池各 123 轮。32K 无 host 恢复；16K 恢复 10752/896/13120 tokens，各组全部成功、无 retraction。单次运行、输出和等待差异不足以选 B6，原始证据见[容量与写策略](native_policy_validation.md)。
- 16K 研究池独立完成 76 个 case 的标定：552 个正式请求、184 个独立 warmup，共 2268 条阶段/完成传输观测；四份 manifest 与合并文件来源一致。c4 首 token 留出仅 50% 受支持，长输入的输出预算变化主要反映在首次 forward 前的联合等待/准备区间；配置匹配不等于预测准确。详见[有限池标定分析](capacity_calibration_validation.md)。
- 同一 16K/279040-host 配置的六方法矩阵共 246/246 成功。Ours 与 no-retire 的 41 轮输入/输出完全相同，但仍有 34 次固定策略回退、零成本切换；完成均值 5.325/5.313 s，未观察到退役的时延收益。Rebuild 为 4.122 s，但与其 no-retire 只有 6 轮输入一致，不能将差值直接归因 C。完整状态、真实恢复/传输、回退因子与初态局限见[有限容量联合验证](controller_capacity_validation.md)。
- 质量验收新增双侧原始问题/历史/检索材料/输入身份绑定，核对原运行的 split 声明与既有摘要，并汇总各层模拟标记。独立反例审查修正了开发数据冒充 heldout、模拟来源被 audit 遮蔽、相同答案挪用到不同提示三个漏检。真实 P3 的 123 轮原始绑定均通过；v2 盲审包评分仍全部为空，不建立质量结论。见[质量验收](quality_acceptance.md)。
- 字面表格注释已完成 82 轮 P3 与 8 次冻结历史实验；随后直接复用 CRAFT 的 MIT 行文本函数，扩展当前 query 的词法焦点与行内来源。两固定历史、三臂正反顺序共 12 次 GPU 运行全部成功，新样式在两历史中均保留正确单附件限制和支持它的原引用，未重现已知范围错误。已可选接入共享候选、原生基线和矩阵；完整正文/QA/预算保持，默认关闭。仅是开发探针上的局部改善，见[焦点行验证](table_focus_validation.md)。
- Selene Mini 8B 与 Prometheus 2 7B 的原生五维协议各完成 10 次真实评分，均正常解析，但关键范围扩张与缺引用仍有漏判。GroUSE 固定 train16 分别完成 45/32 次调用；Selene 条件通过48/96，6次字符串 null 失败；Prometheus 30次输出截断、2次JSON失败，零分代表该协议运行失败。官方解析/DTO与判分重放一致。显式 normalized_v1 已实现并完整重跑 train：51次调用全解析、64/96条件通过，完整性仍仅5/16；不回填旧调用，不按格式完成率宣布语义可靠。见[独立评审验证](open_judge_validation.md)。
- 同一优化源码的 gate2/gate0/Fixed4096/Rebuild 两次对照完成328/328轮；源码、8份冷态和全部原始记录通过独立复核。Gate0等会话完成均值下降19.88%，但比Fixed4096慢15.16%，P2输入输出分叉且仍有截断，不建立质量相等或强B6胜出结论。见[稳定期消融](stage_gate_validation.md)。
- `focus_refs` 完整P3三臂共123/123成功、每臂2轮截断，完整输入/QA/来源重建通过。附件目标有原始引用的10MB回答改善了已知范围错误，但后续套餐条件、SDK跨产品混引仍存在，非触发轮也因闭环生成变化而分叉，见[P3焦点验证](table_focus_p3_validation.md)。
- 新增[原子声明诊断](atomic_judge_proposal.md)复用既有HTTP、完整输入绑定与来源合同，适配RefChecker/Claimify等抽取和来源核验原则。实际代码支持抽取、原文位置、限定槽位、可见/被引来源分判及未知/失败记录；41项专项与独立审查通过，已包含在667项全套中。尚无真实原子模型评测，不进入正式五维质量gate，也未在线修答案。

## 仍需完成的研究验收

当前没有正式回答质量非劣、SLO goodput 提升或论文优于基线的结论。已有 B0/B1、控制器矩阵、匹配消融、成对诊断、原生容量实验及 16K 匹配标定/六方法联合运行。接续重点是区分可观测容量压力与传输不确定性、改善等待成本估计，补 32K 对照、重复测量、同预算 B6、低重叠负例、真实质量评分和人工盲审。profiling.decision_pairs 与生命周期压力探针分别补充成本差和旧独有分支归因；全局 slot 变化不能替代具体路径证据。

完整 Cloud P3 开发流程已运行。官方 qrels 覆盖 39/41 轮，缺失的两轮单列，未按零分混入；在有标签部分，BM25S recall@5 为 0.2094、nDCG@5 为 0.1671。这只是本配置下的检索弱基线，不是回答质量。原始 qrels 诊断、B0/B1 运行与 P3 生成保存在 runs/remote-h200-native；仍须完成回答质量评价、强基线和冻结 test。缺少评分时，质量覆盖、非劣判定和人工审查保持未验证，不以格式检查、模拟回答或词法代理补齐。运行方法见 [部署说明](deployment.md)、[成对决策](decision_pairs.md) 和 [贡献与实验设计](research_contributions.md)。
