# ContextKV 最终结果登记

更新于 2026-11-03，登记截至 06:50 已确认的容器产物。评审基线为
`7388008f56ec7176668321d00f69bd2f07639f0a`。数据、模型、日志和结果保留在指定主机的
`sjy-0518` 容器；本文路径沿用各批次 manifest。最终源码与配置 SHA 由交付提交和冻结记录固定。

任务状态、条件修正、历史、版本与来源目录已经接入集合搜索和 READ/REDUCE/ANSWER。
运行时持久化真实步骤、原生回执和一次 QA 提交。当前 B 开发比较选择 B3，C 开发选择 native；
reader 与 A 策略的公开最终选择由最终批次 `frozen.json` 登记。

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
数据 split、目标复核、排除清单、采样倍率、adapter、合并权重、许可与配置路径见
[training_assets.json](training_assets.json)。

## A：固定实际历史的四格

原始目录为 `b/fourgrid-prototype22/frozen/`。四格共用实际 TaskState、候选池和历史，
主分为 `project_condition_version_score`。

| 方法 | 完成 QA / 预定状态 | 共同评分状态 | 主分 | 相对 P0_R0 的百分点差 |
|---|---:|---:|---:|---:|
| P0_R0 | 13 / 14 | 13 | 0.9230769231 | 0 |
| P0_R1 v1 | 14 / 14 | 13 | 0.9461538462 | +2.3077 |
| P1_R0 | 14 / 14 | 13 | 0.9615384615 | +3.8462 |
| P1_R1 v1 | 14 / 14 | 13 | 0.9769230769 | +5.3846 |

同 reader 的 P1−P0 分别为 R0 +3.8462、R1 v1 +3.0769 个百分点。P0_R0 的第 14 个状态为预算失败，
原 attempt、步骤和费用保留。早期 sixgrid-v1 共同 12 状态的 E0/inferred 分数为 R0 0.8708、R1 v1 0.7333；
两批按各自输入和运行版本独立登记。自身 P3、R2 与公开对照的完成数、评分分母和成本引用最终 A 批次报告。

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
滚动策略结合真实 native quote、完整阶段时窗与物理成本；统计预测保留观测范围、预测区间及外推标记。
ANSWER 使用选择完成后的 remaining-control，已支付控制与后续成本分开入账。

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

| 并发 8 方法 | 完成 QA | 完整响应 P95 ms |
|---|---:|---:|
| B6_native | 16 / 16 | 113811.5896 |
| S3_native | 16 / 16 | 94399.0444 |

该负载点 S3 的 P95 减少 19412.5452 ms，下降 17.0567%。全部 c1 四格、c4/c8 的实际到达、吞吐、
prefill、回载、Need/source 覆盖和批次费用见 `workflow.json` 与 `jobs/*/capacity-summary.json`。
此批为两轮容量协议；完整七轮五场景 P3 另批登记。联合四格允许自身回答、分支和 DAG 变化。

## 公开多跳、长文与最终批次

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
受控 balanced R0 长文已得到 36 条、666000 g，最终回答为 36 条、666 kg；
早期 READ 数值错误与 ANSWER 忽略全范围结果的运行保留在机制审计中。

最终公开候选为 P0_R0、P1_R0、P1_R1、P0_R2、P1_R2。R2 开发后启动公开测试；
全批逐轨质量、评分覆盖、实际选择和测试状态引用 `bench.a_final_batch` 最终批次的
`frozen.json`、`report/public_summaries.json`、`report/paired_comparisons.json`、`report/artifact_inventory.json`。
MultiDoc2Dial 的预定开发为 15 turn / 2 conversation，长文开发为 2 个独立训练来源案例；
最终输入 ID 与资产 SHA 以该批 cohort/freeze manifest 为准。

## 完整成本与交付索引

各阶段保存真实采集、训练、验证、merge、在线辅助模型、READ/REDUCE/ANSWER、失败调用及离线评分费用。
R1/R2 的 `training_cost*.json`、逐 rank 步骤、`train_physical_resources*.json` 和 `merge_cost*.json` 与模型资产共同保留。
容量总成本在 `capacity-cost.json`；请求份额和嵌套批次积分由去重总账索引，观察缺口保留为 unknown。

最终汇总使用 `profiling.final_study_index` 读取已有评分和 manifest，生成 `final-index.json/md`；
`profiling.study_cost_ledger` 将同一时间轴的物理卡/PID 样本合并为唯一积分。
完整成本数值与采样覆盖引用最终 `full-cost-ledger.json`，费用口径见
[cost_retention_report.md](cost_retention_report.md)。

首个 QA 源码提交为 `edb20e7`。交付索引登记最终源码、运行配置、训练数据清单、两套 reader 资产、
策略资产、冻结 manifest、逐题结果、原生回执和成本来源。当前机制取舍为：B3 保留开发质量优势，native 保留 C 开发选择，
S3 在已测八并发容量点具有更低 P95；公开质量与最终配置以最终批次产物登记。
