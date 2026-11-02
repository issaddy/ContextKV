# ContextKV 实施状态

评审基线：`7388008f56ec7176668321d00f69bd2f07639f0a`。
原版任务书保存在 [task_spec.md](task_spec.md)。
2026-11-02 重新读取的[在线指导](https://chatgpt.com/space/page_2c79ac76b0e8819187e8d0c156ddd634)
已经改为“ContextKV 核心模块重构与实施指导”。

## 当前范围

用户此前取消新增训练；最新目标文件重新授权 reader LoRA 与计划器交替训练。
阶段依赖和验收见 [接续清单](stages.md)。
默认 [study.yaml](../../configs/study.yaml) 使用现有冻结 BGE 查询编码器，
任务解析、组合支持和阅读共用固定 Qwen3-32B 非思考阅读器。
集合搜索复用已经得到的单段和组合支持分；不在每个 beam 分支重新调用生成模型。
R1 通过独立运行配置载入；默认配置尚未因训练完成而切换为 R1。
当前可运行配置属于 Incumbent 候选，尚未通过同一开发 cohort 选出最强配置。

用户指定的远端环境是
`public-ge108-wlf3-hangu-5.idchb2az3.hb2.kwaidc.com` 的 `sjy-0518`。
用户随后按指定交互式 SSH 路径恢复了容器访问。只同步必要源码；数据、模型、
日志和结果留在生成环境。诊断终端输出每次最多 700 字符。

新版指导将 reader LoRA 与计划器联合训练、反事实质量增益驱动的滚动决策、
稳定证据前缀独立所有权列为主线。训练数据/LoRA 与收益策略入口已实现，
R1 已完成真实训练、合并与服务启动；R1绑定的P1 prototype22已训练，B v3也已完成真实反事实采集与拟合。
两者的在线质量收益仍待对应评分；R2当前是混合数据与新P1目标复核阶段。
TaskState、ContextPlan、EvidenceSet、
ReadResult/ReadingOutcome 共同表达统一逻辑状态，不再建立平行 AnswerState 副本。

## 实现与证据

| 内容 | 代码 | 当前证据 |
|---|---|---|
| 任务、历史、条件与版本 | `adapters/history_state.py`、`evidence_pool.py` | 原文坐标、有限指针、条件更新和历史来源目录的本地测试 |
| 单段、组合支持与集合搜索 | `adapters/support_model.py`、`teacher_support.py`、`controller/evidence_planner.py` | single/pair/triple、依赖闭包、真实 tokenizer 预算与多集合测试 |
| READ/REDUCE/ANSWER | `controller/reading.py`、`execution.py`、`reduction.py` | R0新balanced长文全范围答36/666；多跳仍失败，R1尚缺匹配复跑 |
| 原生运行时 | `controller/service.py`、`session_store.py` | 步骤恢复、持久化、一次 QA 提交与成本累计测试 |
| 多候选与 KV 成本 | `controller/candidates.py`、`planner.py`、`context_cost.py` | 同动作多集合、三个布局、合法 APPEND、逐步骤报价与输出 profile 测试 |
| 实验编排 | `bench/study.py`、`study_metrics.py`、`study_scoring.py` | 连续质量选择、公开原始指标、完整队列/生成/辅助开销和物理积分接口 |
| Reader LoRA | `training/reader_data.py`、`train_reader.py` | R1 真实训练完成；构造型开发共同12状态E0/inferred为R0 0.8708、R1 0.7333，首轮未提升 |
| R1 绑定的计划器交替训练 | `training/train_evidence_policy.py`、`a_workflow.py` | prototype22已用22答案/5状态训练；四格各14状态生成完成、待评分；自身训练P3与新目标复核已推进 |
| 来源配对 curriculum | `training/reader_curriculum.py`、`reader_constructed_curriculum.py`、`reader_computed_curriculum.py`、`reader_mix.py` | 已审课程新增153样本，含parent共1945样本；配置课程3倍、新P1 8倍采样，R2尚未报告训练完成 |
| B 收益拟合 | `training/collect_plan_gains.py`、`train_plan_gain.py` | v3 train8/dev4真实176次model call、721867ms，已fit；滚动运行收益待验 |
| 受控机制离线审计 | `bench/audit_reading_mechanisms.py` | 分别核对执行、原文/claim来源、REDUCE量值和最终ANSWER；代码与本地测试完成，实际报告待容器运行 |

上述是已实现的接口与局部验证，不代表全量质量、性能和研究验收完成。

实现细节分别见 [任务与来源](task_sources.md)、[支持与搜索](support_and_search.md)、
[阅读运行时](reading_runtime.md)。

## 已有真实结果

原始 Dense5 在固定实际历史批次上完成 MultiDoc 206 题和 Cloud 21 题。
MultiDoc `RB_llm=0.844903`；Cloud 原始 `RB_llm=0.728571`。
这两个批次未连续采样 GPU/PSS 积分，不能作为完整等成本比较。

受控资格段＋办理段样例完成真实 READ 和最终 ANSWER，一次请求增加一条 QA 和一次版本。
Windows→Linux 纠正样例生成了对应 Linux 步骤。
这些样例使用受控任务和支持标注；自动解析与支持模型的贯通另由
`bench.provider_mechanisms` 运行，尚未取得更新后的 GPU 结果。

长文 v3 完成 12 次 READ、2 次 CPU REDUCE 和一次 ANSWER，但质量失败。
后续 v9 按计数与求和需求分别执行 24 次 READ，最终得到 37 条、667 kg，
期望为 36 条、666 kg。诊断定位到元数据 ID 被当作实体，以及模型生成不同版本标记
导致重复计入。多跳曾在 v5 正确回答 1997；v9 在第二次 READ 的新事实父 ID 上失败。
当前协议将来源版本交由控制器绑定，将新事实的缺口直接挂在该事实下；
识别符型扫描可以显式约束到正文 ID。R1 已按当前实现完成四项受控运行，
其中 R1 长文执行了 24 次 READ、2 次 REDUCE 和一次 QA，实际 REDUCE 得到
36 条、655000 g，但最终 ANSWER 回答 3 条、6 kg；原文期望为 36 条、666 kg。
同代码 R0 的 REDUCE 为 36 条、596000 g，最终也回答 3 条、6 kg。
实际最终 prompt 已包含 Computed 行与根结论；最终答复仍取了代表摘录的局部量值。
需要分别解决 READ 质量总量提取错误与 ANSWER 忽略全范围计算结论的问题，
不能把这组共同失败归结为纯 R1 退化。R1 报告覆盖条目 24、未补需求 7。多跳只推进到
Institute→Arbor，未输出期望年份 1997。流程完成与任务正确分开记录，
这两项属于失败结果，未进行答案或 claim 修补。pair/correction 已生成预期表单词项，
其原文支持与引用审计尚待完成。相同代码上的 R0 配对归因运行由主 agent 推进。

阅读器实际开启紧凑 JSON 约束；原生缓存配置签名已经包含解码配置。
早期多个教师服务已停止，最新受控实验使用同一 Qwen3-32B 服务。
当前确认的 R1 服务为 `30004/GPU4`，可比 R0 为最新引擎上的 `30005/GPU5`；
离线冻结教师为 `30001/GPU2`，C 服务占用 `30002/GPU3`。
这些是主 agent 的最近报告，不代表本地 agent 另行进行了远端健康检查。
官方指标隔离环境已完成单句 smoke：相同句子的 BERTScore F1 与 ROUGE
1/2/L/Lsum 均为 1；尚无这批 benchmark 的 `RB_alg` 分数。
完整主矩阵、开发冻结、公开测试和负载实验仍未完成。

容器训练分区已有 10004 轮、1539 会话的实际历史。已导出 2000 轮 ANSWER，
并从真实检索材料构造 source-first 协议 2 READ 输入。教师复核产物保留在容器，
同调用自评分仅用于训练筛选，不是独立 benchmark 分数。复核超窗样本明确排除，
未替换实际历史或截断训练目标。教师输出达到长度上限的样本保留原始输出、
实际 token 与成本，标记 `teacher_target_truncated` 后排除。

R1 训练数据共 1792 个已复核样本（1559 ANSWER、233 READ），其中训练样本
1773 个、来自 1544 个训练 turn；训练分区另留 19 个 validation 样本。
真实训练完成 56 个 optimizer step，处理 7,800,825 个输入 token，训练墙钟
1,405,076.37 ms（约 23.4 分钟）。checkpoint 为
`contextkv-reader-r1:1eec232b42b67721374c0400b354081842b54c965a20c441b17b4d82a33c567b`。
LoRA 已合并并启动服务，R0 原权重保留。训练分区持出 loss 为
`0.1608121544`；该数值不是 QA benchmark 提升。
当前小规模实际分布上的 R1 仍未解决上述长文和多跳受控任务，尚无回答质量改善结论。

随后新balanced配置的R0长文已实际得到REDUCE 36条/666000g，最终ANSWER 36条/666kg，完整来源范围通过；
多跳仍失败。该配置与上面的早期R1失败运行不同，需要匹配R1复跑后才能做reader归因。

sixgrid-v1 已实际评分152格，16格缺失；主表使用排除2个泛topic后共同的12个构造型动态状态。
工程主分 `project_condition_version_score` 的 E0/inferred 为R0 0.8708、R1 0.7333；
E1/inferred和E2/inferred两reader均为1.0000。同候选池oracle选材的差距为R0约12.9、R1约26.7个百分点，
说明这批状态的实际证据选集仍有明显改进空间。首轮LoRA没有提升，R0保留为较强当前候选。
verified各格及缺失范围见[六格诊断](reader_diagnosis.md)；这些结果不代表公开benchmark或已选出最强Incumbent。

后续P1 prototype22已训练，资产为`models/study-support-r1-prototype22`，监督来自22个真实候选答案、5个训练状态。
P0/P1 × R0/R1 冻结实际历史四格均已完成14状态生成，共56个状态格；当前尚待score，不能据生成完成判定P1提升。
P1自身训练P3首会话完成5条QA后遇到Task教师`completed/length`，已记终态模型预算失败、跳过后继并保留已付费用；
第二会话最新观测为首轮尚未提交QA，CPU在temporal来源过滤停留约13分钟，reader/helper在该观测时均空闲。
已交付按文档族/版本索引、调用内只计算一次temporal结果的补丁；root安排中止旧计算并按同数据重启。
新P1目标复核与R2训练完成仍待实际报告。已有QA复用，原失败没有修补或重试。

课程审计完成后保留新增153个样本，加上parent的1773训练样本和19验证样本，共1945个样本，真实训练轮数仍为1544。
R2混合配置对课程显式重复3份、新P1样本重复8份，从实际R1 adapter初始化，保留原验证文件。
这些是已审数据和采样设置；新P1复核、实际混合计数、R2训练及质量结果仍分别记录。

B v3已在8个训练状态和4个开发状态完成176次真实model call，采集墙钟721867ms，并完成fit。
C已进入完整READ/ANSWER/REDUCE真实成本采集和拟合阶段。两项的最终在线/独立联合效果尚未据此成立。

公开多跳数据已在指定容器下载并转换：作者官方镜像2WikiMultihopQA的12576条有标签dev，按固定hash选40条开发样本和40条互斥project-heldout样本。
这不是官方隐藏test，也没有全量覆盖。原始上下文含distractors进入阅读路径，gold答案、支持句和关系只用于离线评分；
来源revision、内容SHA、cohort IDs和原文句跨度由`bench.public_multihop`保存。
镜像身份见[作者数据卡](https://huggingface.co/datasets/xanhho/2WikiMultihopQA/blob/612bc5039a457880d9e7d84c3b0a4cf154b70e4f/README.md)，
格式与原始评估器见[官方仓库](https://github.com/Alab-NII/2wikimultihop)。
`bench.a_final_batch`提供声明子集、开发冻结、公开评分与报告入口；目前不把入口代码记为公开结果完成。
数据、模型、成本及评测状态见 [A 实验清单](a_experiment_inventory.json)。
隔离评分依赖与 P1 首批采集前置条件见 [R1/P1 运行清单](reader_policy_readiness.md)。

C 的固定 token/受控压力原生配对已真实执行：GPU 组额外命中 128 token，
减少 128 token Host 回载，实测响应减少约 10.4 ms；Host 组额外命中 128 token，
减少 128 token 重算、增加 128 token 回载，实测响应增加约 17.0 ms。
此结果只证明独立前缀 owner 的机制和两层取舍，尚无长期价值策略或答案质量收益结论。
后续 C2 固定 DAG 的实际配对已执行；当前C完整阶段成本采集和拟合正在推进，
固定DAG配对与成本fit均不能代替未来复用价值策略的效果证据。
实际语义 READ 资格脚本首轮在输出审计的反序列化处失败，已修正后待复跑。

早期批次的题数、原始目录和评分缺失原因见
[实验清单](c_experiment_inventory.json)，该文件是暂停远端时的历史快照；传输停止及部分归档见
[传输清单](c_transfer_ledger.json)。

## 验证与下一批验收

首个 QA 提交为`edb20e7`（当时1701 passed）；主agent随后报告过1761项完整回归。
按09:00截止要求，后续仅运行必要管线smoke、编译检查与原生契约资格，不再反复扩大测试。
当前新增管线代码和生成完成状态分别登记，未以旧回归或模型训练代替当前质量评分。

自动任务解析、支持评分、集合选择、READ/ANSWER的实际训练P3已产生结果与明确失败；
长文/多跳、开发评分、公开cohort准备按截止时间并行推进。
六格已完成上述共同12状态的真实评分，两个泛topic的E1/E2仍待精确范围审核。
计划器 × 阅读器四格各14状态已生成、待评分；收益驱动决策 × 前缀保留四格仍缺完整质量/成本结论；
正式 QA heldout、R1 长文/多跳通过验收与独立效果消融仍缺失。

## 实验协议与机制取舍

- 原文跨度与 prompt 跨度分开；共享正文保留各自版本、对象和来源身份。
- 最终窗口保留根结论和关键原文。超大 READ 单元可按真实 token 预算切片，
  优先从实际抽取事实的原文位置开始；完整叶子图保存在步骤账本。
- 计数、求和使用确定性 REDUCE；抽取错误仍会传入汇总，不能靠计算模块补成期望值。
- 无训练配置采用支持覆盖效用。生成式集合重排作为保留集合上的独立实验，
  避免把大量生成调用隐含在搜索分支中。
- 已知 READ 输入在当前缓存状态报价；未来 REDUCE 输入与 OpenNeed 内容在实际执行时记录。
  READ、REDUCE 和摘要步骤使用各自的开发期计时、输出长度和控制开销 profile。
- 开发集在完整预定题集和预算内按连续质量选择；测试保留 B0、Btoken、B3、B6、Lmap
  及开发入选方案。原始 benchmark 指标与工程 Q 分开报告。

## 入口

```bash
python -m bench.study prepare --config configs/study.yaml
python -m bench.provider_mechanisms --help
python -m bench.reading_mechanisms --help
python -m bench.study evaluate --split dev --matrix configs/study_matrix.yaml
python -m bench.study freeze --results runs/study/dev/results.jsonl
python -m bench.study evaluate --split test --frozen runs/study/frozen.json
python -m bench.study report --frozen runs/study/frozen.json
```

以上为执行入口；开发冻结、公开测试和报告命令尚未完成真实运行。
