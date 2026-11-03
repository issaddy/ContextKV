# ContextKV 最终结果登记

更新于 2026-11-03，模型调用批次于 08:27 结束。评审基线为
`7388008f56ec7176668321d00f69bd2f07639f0a`。数据、模型、日志和结果保留在指定主机的
`sjy-0518` 容器；本文路径沿用各批次 manifest。最终源码与配置 SHA 由交付提交和冻结记录固定。

任务状态、条件修正、历史、版本与来源目录已经接入集合搜索和 READ/REDUCE/ANSWER。
运行时持久化真实步骤、原生回执和一次 QA 提交。B 开发比较选择 B3，C 开发选择 native。
公开批次在源码 `487ce5d` 冻结，因开发候选均超出 P95 约束，运行方案保留 P0_R0；测试结果保持该选择。
最终运行源码为 `6ff46fc`，只读分析随交付提交登记。模型服务进入收尾，训练资产与原始实验产物保留。

## 数据与模型

训练原始轨迹为 `task-a/turns.jsonl` 的 10004 轮、1539 会话，配套 SQLite 保存实际生成历史与检索原文。
R1、R2 reader LoRA 和绑定 R1 的 P1 prototype22 均已完成训练；R0 保留原 Qwen3-32B BF16 权重。

| Reader | train / validation | 输入 / 监督 token | optimizer steps | 同步训练墙钟 ms | validation loss |
|---|---|---:|---:|---:|---:|
| R1 v1 | 1773 / 19；1544 个实际 native train turn | 7,800,825 / 355,628 | 56 | 1,405,076.37 | 0.1608121544 |
| R2 | 2264 个采样例 / 原 19；1544 个实际 native train turn | 10,759,242 / 597,099 | 71 | 1,952,039.975 | 0.157603919506073 |

R2 从 R1 adapter 继续训练，使用 GPU `0,4,6,7`。2264 个采样例由 1773 个 parent、
153 个课程样本各 3 份、4 个新 P1 实际历史样本各 8 份组成。原 validation 文件保持原样；loss 对应训练分区持出集。

| 资产 | identity / 绑定 | 容器目录 |
|---|---|---|
| R1 v1 | `contextkv-reader-r1:1eec232b42b67721374c0400b354081842b54c965a20c441b17b4d82a33c567b` | `models/study-reader-r1/`、`models/study-reader-r1-merged/` |
| R2 | `contextkv-reader-r1:7521b3cffa58df5c0973b361f12d9138bcf358c6404a1b6b6a4c6b440f62e768` | `models/study-reader-r2/`、`models/study-reader-r2-merged/` |
| P0 | actual-prefix 支持资产 | `models/study-support-actual-prefix/` |
| P1 prototype22 | 22 个真实候选答案、5 个训练状态；绑定 R1 v1 | `models/study-support-r1-prototype22/` |

R2 沿用 driver 的 identity 前缀，以逻辑名称和完整 digest 区分。P1×R0、P1×R2 登记为跨 reader 迁移。
原 R1 服务进程 `133486` 在其全部实验完成后停止，adapter、合并权重与原结果保留。
数据 split、目标复核、排除清单、采样倍率、adapter、合并权重、许可与配置路径见
[training_assets.json](training_assets.json)。
R1/R2 adapter 目录内的 `BASE_MODEL_CARD.md`、`LICENSE`、`BASE_NOTICE` 实测均缺失；
基座 revision 和来源 scope 保留在资产清单中。

## A：固定实际历史的四格

原始目录为 `../fourgrid-prototype22/frozen/`。四格共用实际 TaskState、候选池和历史，
主分为 `project_condition_version_score`。

| 方法 | 完成 QA / 预定状态 | 共同评分状态 | 主分 | 相对 P0_R0 的百分点差 |
|---|---:|---:|---:|---:|
| P0_R0 | 13 / 14 | 13 | 0.9230769231 | 0 |
| P0_R1 v1 | 14 / 14 | 13 | 0.9461538462 | +2.3077 |
| P1_R0 | 14 / 14 | 13 | 0.9615384615 | +3.8462 |
| P1_R1 v1 | 14 / 14 | 13 | 0.9769230769 | +5.3846 |

同 reader 的 P1−P0 分别为 R0 +3.8462、R1 v1 +3.0769 个百分点。P0_R0 的第 14 个状态为预算失败，
原 attempt、步骤和费用保留。早期 sixgrid-v1 共同 12 状态的 E0/inferred 分数为 R0 0.8708、R1 v1 0.7333；
两批按各自输入和运行版本独立登记。

自身 P3 的旧代码四格采用独立离线协议 `goal_reference_correctness_conditions_v1`，
使用实际问题、回答、文本历史、参考目标与当前条件，主分为正确性和条件版本分的均值。
groundedness、answerability、citations 三维为 null；该分数登记参考正确性与条件符合度。

| 旧代码自身 P3 方法 | 已存 QA / 预定 14 | 已评分 | 共同 8 题主分 |
|---|---:|---:|---:|
| P0_R0 | 14 / 14 | 14 | 0.93750 |
| P0_R1 v1 | 8 / 14 | 8 | 0.93125 |
| P1_R0 | 11 / 14 | 11 | 1.00000 |
| P1_R1 v1 | 12 / 14 | 12 | 0.83750 |

当前代码 R0/R2 四格采用同一独立 goal-reference 协议，已完成：

| 当前代码自身 P3 方法 | 已存 QA / 预定 14 | 已评分 | 共同 8 题主分 | 各自已评分均值 |
|---|---:|---:|---:|---:|
| P0_R0 | 11 / 14 | 11 | 0.81250 | 0.863636 |
| P1_R0 | 11 / 14 | 11 | 0.73125 | 0.804545 |
| P0_R2 | 12 / 14 | 12 | 0.90625 | 0.854167 |
| P1_R2 | 14 / 14 | 14 | 0.74375 | 0.782143 |

P0 下 reader 替换的共同集差为 +9.375 个百分点；P1_R2 相对 P0_R0 的联合差为 −6.875 个百分点。
原始结果与共同题 ID 见
`runs/study/a-own-p3-current-r2-final/goal-reference-scoring/comparison_report.json`。

## B：真实增益与滚动选择

早期 v3 真实反事实采集包含 train 8、dev 4 个状态，176 次模型调用、721867 ms。
当前 READ prompt 对应 `models/study-plan-gain-r0-bridge-v4.json`；
成本资产为 `models/study-plan-r0-bridge-v3-execution.json` 与
`models/study-plan-r0-bridge-v3-native/native-assets.json`。

| 当前滚动开发方法 | 共同 QA | RB_llm |
|---|---:|---:|
| B3 | 6 | 0.800000 |
| S3，λ=0 | 6 | 0.633333 |
| S3，λ=1e-5 | 6 | 0.566667 |

该开发比较选择 B3。S3 两档与 B6 继续按预登记测试方案执行。
已在开发阶段选定的 B3 完成同一公共 13 题独立测试，13/13 完成并评分，RB_llm 为 **0.5230769231**，
完整响应均值 **56275.347 ms**、P95 **78036.13 ms**。全部在线模型合计 **2,242,604 token**；
按请求分摊的 GPU/Host 物理积分分别为 **129416.704 / 39326.432 GiB·s**。
产物为 `runs/study/incumbent-b3-public-final-v1/`，方法选择沿用测试前的开发冻结。
滚动策略结合真实 native quote、完整阶段时窗与物理成本；统计预测保留观测范围、预测区间及外推标记。
ANSWER 使用选择完成后的 remaining-control，已支付控制与后续成本分开入账。
R2 成本资产为 `models/study-plan-r2-bridge-v3-execution.json`，含 54 个完整阶段 profile，
覆盖 READ、REDUCE、ANSWER、CPU_REDUCE；native timing 资产位于
`models/study-plan-r2-bridge-v3-native/native-assets.json`。

## C：独立前缀与价值消融

稳定来源前缀按实际执行输入确定，端点按 page size 向下对齐。独立 owner 跨业务 epoch 保留，
记录 TTL、GPU/Host 原生 soft refs、释放与重启核对。原生命中收益、共享页边际逻辑容量和连续物理积分分别登记。

固定 token 资格探针中，GPU 组增加 128 个命中 token、少回载 128 token，响应约减少 10.4 ms；
Host 组增加 128 个命中 token、少重算 128 token、多回载 128 token，响应约增加 17.0 ms。

当前语义轨迹为 `runs/study/c-current-r0-g5-v4/`，已真实重采 train 7、dev 14 轮，
完成 4 个固定 token/DAG 配对窗口，train/dev 各 2，TTL 为 15 秒。输入、采样和输出预算一致性证明、
目标再访问、其他来源与控制成本、整窗 `Native−Retain` 差保存在各 `paired-rows.jsonl`。

| train-only 拟合规则 | dev 预测覆盖 | dev MAE ms | dev 正收益留存动作 |
|---|---:|---:|---:|
| 完整特征 | 0 / 2 | 未覆盖 | — |
| prior_access_count | 0 / 2 | 未覆盖 | — |
| scope / TTL | 2 / 2 | 4452.695606 | 0 |

开发选择保留 native。粗化将预测覆盖提高到 2/2，本批净价值判断选择不增加留存。
三个资产及其 train 来源均保留；模型训练发生在 GPU5，GPU3 联合运行的实际 scope 与动作由每步 `evidence_prefix` 记录。

## 自身 P3 联合与容量

`runs/study/capacity-r0-joint-v1/` 的 8 个 job 已完成，共 56 QA。
并发 1 包含 B6/S3×native/value 四格，各 2 QA；并发 4、8 分别使用 4、8 个真实客户端，各运行两种 native 方法。
首轮同时到达，第二轮在自身首答提交后停顿 30 秒，再按自身回答形成条件纠正 P3。

该批共有 55/56 条已有质量评分：

| 并发 | 方法 | 完成 QA | 已评分 | 容量批次评分均值 | 完整响应 P95 ms |
|---:|---|---:|---:|---:|---:|
| 1 | B6_native | 2 / 2 | 2 / 2 | 1.000000 | 见该 job |
| 1 | B6_value | 2 / 2 | 2 / 2 | 0.950000 | 见该 job |
| 1 | S3_native | 2 / 2 | 2 / 2 | 1.000000 | 见该 job |
| 1 | S3_value | 2 / 2 | 2 / 2 | 1.000000 | 见该 job |
| 4 | B6_native | 8 / 8 | 8 / 8 | 0.962500 | 见该 job |
| 4 | S3_native | 8 / 8 | 7 / 8 | 0.985714 | 见该 job |
| 8 | B6_native | 16 / 16 | 16 / 16 | 0.940625 | 113811.5896 |
| 8 | S3_native | 16 / 16 | 16 / 16 | 0.906250 | 94399.0444 |

八并发点 S3 的 P95 减少 19412.5452 ms，下降 17.0567%，对应已有评分均值降低 0.034375。
全部 c1 四格、c4/c8 的实际到达、吞吐、
prefill、回载、Need/source 覆盖和批次费用见 `workflow.json` 与 `jobs/*/capacity-summary.json`。
此批为两轮容量协议；完整七轮五场景 P3 另批登记。联合四格允许自身回答、分支和 DAG 变化。

## 公开开发冻结与测试

开发冻结使用 MultiDoc2Dial、2Wiki 各 0.5 的预定权重：

| 开发 reader | 加权 Q | 完整响应 P95 ms | 约束结果 |
|---|---:|---:|---|
| R0 | 0.642963 | 135177 | 超出 P95 上限 |
| R2 | 0.615185 | 170321 | 超出 P95 上限 |

冻结文件 `runs/study/a-public-final-v4/frozen.json` 保留 P0_R0 作为运行方案；无候选满足全部开发约束。
公开测试对 R0/R2 执行预登记对照，测试后不重新选方法。

| V4 主测试轨道 | reader | 完成 / 预定 QA | 已有评分 | 原指标均值 |
|---|---|---:|---:|---:|
| MultiDoc2Dial | R0 | 6 / 13 | 6 | RB_llm 0.6500000 |
| MultiDoc2Dial | R2 | 6 / 13 | 6 | RB_llm 0.3666667 |
| 2Wiki | R0 | 40 / 40 | 37 个有效字段 | answer F1 53.8060588 |
| 2Wiki | R2 | 40 / 40 | 37 个有效字段 | answer F1 60.2110352 |
| Loong | R0 | 1 / 4 | 1 | rubric 30 |
| Loong | R2 | 2 / 4 | 2 | rubric 30 |

2Wiki 精确共同 34 题的 answer F1 为 R0 **55.61247576**、R2 **62.58259714**，R2 提高 **6.97012138** 个百分点。
各自 37 个有效字段的均值与共同集结果分别保存，字段失败保留原始输出。

2WikiMultihopQA 原始有标签 dev 共 12576 条，固定 hash 形成互斥 40/40 开发与 project-heldout cohort，
保留全部原文与 distractors。全文 V4 使用相同 8192 body 预算与 32K 模型窗口，预算内直接回答、超预算执行 READ。
该批采用同输入 reader 对照，P0/P1 策略头不参与证据选择。

| 全文 V4 开发 reader | 完成 QA | 已有字段评分 | answer F1 |
|---|---:|---:|---:|
| R0 | 12 / 12 | 12 / 12 | 59.259 |
| R1 v1 | 12 / 12 | 10 / 12 | 49.167 |

F1 按各自已有评分分母登记；字段失败、原始答案与费用保存在对应评分目录。
Loong set1 测试预选 4 题，包含 1 paper、2 legal、1 financial，全部 17 个原文实例保留；
样本选择与导出回执见 `data/loong/set1-cohort4/{cohort_manifest.json,study_spec.yaml,source/}`。

主冻结预算为 `4K/8K`；长文 `12K/8K`、32K 窗口预算消融独立登记。当前受控机制结果为：

| 当前受控运行 | pair | 条件纠正 | 长文 literal audit | 多跳 literal audit |
|---|---|---|---|---|
| R0 v3 | 通过 | 通过 | 失败 | 失败 |
| R2 v1 | 通过 | 通过 | 失败 | 失败 |

早期 balanced R0 v2 的 36 条、666 kg 属于已保存的单次通过记录。当前运行按各自 literal audit 登记。

| Loong 批次 | 预算 | R0 完成并评分 / 4；均分 | R2 完成并评分 / 4；均分 |
|---|---|---:|---:|
| 主冻结测试 | 4K/8K | 1 / 4；30 | 2 / 4；30 |
| 输入预算消融 v1，旧代码 | 12K/8K，32K 窗口 | 3 / 4；53.3333 | 4 / 4；25 |
| 输入预算消融 v2，当前 `6ff46fc` | 12K/8K，32K 窗口 | 4 / 4；38.75 | 3 / 4；41.0 |

各批保留自己的完成分母与实际 TaskState，测试后方法选择保持原冻结。
R2 v2 的一个 READ 在累计使用 4,177,440 token、311 个步骤后，下一步请求 20,296 token 超过
4,194,304 总预算，按预算终态保存失败与费用。

集成提交 `41491a4` 补齐实际辅助 gap 用量，`6ff46fc` 修正 Need 命名空间变更后的 SourceSpan 原坐标几何去重；
原冻结失败与原始费用保持原记录。最终逐题、评分覆盖和比较来源为
`runs/study/a-public-final-v4/report/{public_summaries.json,paired_comparisons.json,artifact_inventory.json}`，
输入 ID、预算与资产 SHA 由该批 cohort/freeze manifest 固定。

## 完整成本与交付索引

各阶段保存真实采集、训练、验证、merge、在线辅助模型、READ/REDUCE/ANSWER、失败调用及离线评分费用。
R1/R2 的 `training_cost*.json`、逐 rank 步骤、`train_physical_resources*.json` 和 `merge_cost*.json` 与模型资产共同保留。
容量总成本在 `capacity-cost.json`；请求份额和嵌套批次积分由去重总账索引，观察缺口保留为 unknown。

最终汇总使用 `profiling.final_study_index` 读取已有评分和 manifest，生成 `final-index.json/md`；
`profiling.study_cost_ledger` 将同一时间轴的物理卡/PID 样本合并为唯一积分。
完整成本数值与采样覆盖引用最终 `full-cost-ledger.json`，费用口径见
[cost_retention_report.md](cost_retention_report.md)。

首个 QA 源码提交为 `edb20e7`。交付索引登记最终源码、运行配置、训练数据清单、两套 reader 资产、
策略资产、冻结 manifest、逐题结果、原生回执和成本来源。当前取舍为：B3 保留开发质量优势，native 保留 C 开发选择；
S3 八并发响应更快，对应容量评分较低；R2 的 2Wiki 共同集 F1 提高，MultiDoc2Dial 已评分均值较低，
Loong 完成覆盖增加。公开运行方案沿用测试前冻结的 P0_R0。

## 封存成本与收尾

最终物理账本为容器内 `runs/study/full-cost-ledger.json`，索引 596 个来源文件，无缺失链接；
八卡 UUID 各自唯一。同一物理卡/PID 的重叠采样去重后，已观察 GPU 积分为
**10,120,861.9885 GiB·s**，已观察进程 PSS 积分为 **2,307,280.1306 GiB·s**。
历史全八卡同步采样的合并缺口为 **25,533.5349 秒（7.0926 小时）**；各卡缺口原样登记，未补零。
Host 数值采用实际观察到的 PID 并集 PSS，覆盖范围见账本。

| 成本类别 | 唯一完成记录 | 输入 token | 输出 / 监督 token | 账本 |
|---|---:|---:|---:|---|
| 原生 ANSWER/READ/REDUCE | 17,180 | 68,354,498 | 3,066,658 输出 | `final-native-usage-index.json` |
| API 教师、复核、离线评分 | 3,046 | 25,227,835 | 1,142,775 输出 | `final-api-cost-index.json` |
| R1＋R2 训练 | 两轮 | 18,560,067 | 952,727 监督 | 各模型 `training_cost.json` |

原生记录按重启 ID＋实际调用 ID 去重，并核对输入 hash；覆盖 378 个 SQLite 来源，冲突为 0。
数据库中 422 个缺失、不完整或未进入该完成口径的记录项保留在 inventory 中。
API 按实际响应身份去重；训练前向和各请求的辅助编码器用量另表保留，各类不直接相加。
辅助模型输入、输出和时间仍以逐请求全在线成本及原始调用记录为准。

模型与评分批次于 08:27 前结束；本任务模型服务已全部停止，没有强杀剩余 worker。
收尾观察中八卡各占用 4 MiB。两套 reader、策略资产、数据和结果均留在指定容器；本地仅保留源码与交付文档。

## 组件调整索引

| 组件 | 入口 |
|---|---|
| 共享任务/证据/计划合同 | `controller/context_schema.py`、`controller/schema.py` |
| 历史、条件、版本、来源目录 | `adapters/history_state.py`、`adapters/evidence_pool.py` |
| 单段/组合支持、集合搜索 | `adapters/support_model.py`、`controller/evidence_planner.py` |
| 计划收益、预算与滚动选择 | `controller/plan_policy.py`、`controller/plan_profiles.py`、`controller/rolling_policy.py` |
| READ/REDUCE/ANSWER 与持久化 | `controller/reading.py`、`controller/execution.py`、`controller/service.py`、`controller/session_store.py` |
| 前缀目录、所有权与原生缓存 | `controller/prefix_retention.py`、`engine_patches/` |
| 训练、批次与成本复现 | `training/`、`bench/`、`profiling/`；命令见 `final_runbook.md` |

本轮按时间约束保留编译检查和真实批次验证；未再运行完整单元测试集。
源码与只读分析分别记录在 `6ff46fc`、`7b3e4f1`，交付文档由最终提交封存。
