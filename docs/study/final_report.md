# ContextKV 最终报告（结果登记稿）

更新日期：2026-11-03。评审基线：`7388008f56ec7176668321d00f69bd2f07639f0a`。
本稿登记主 agent 在指定 `sjy-0518` 容器报告的实际结果；待填项由对应原始产物补齐。
容器项目根为 `/home/relay/songjiayun/ContextKV/runs/study-20261102/project`，下文产物路径均相对该目录。

## 已完成的训练与冻结诊断

R1、R2 reader LoRA 和 R1 绑定的 P1 prototype22 已完成训练。冻结实际历史的
P0/P1 × R0/R1 四格在共同 13 个状态上的工程主分依次为
0.9231、0.9462、0.9615、0.9769。P0_R0 的第 14 个状态发生预算失败，其他三格完成 14/14。
这组结果来自相同实际 TaskState、候选池与历史的受控开发诊断；公开开发和测试结果在后表单独登记。

| Reader | 实际训练数据 | 输入 / 监督 token | 训练墙钟 | 训练证据 |
|---|---|---|---|---|
| R0 | 保留原 Qwen3-32B BF16 权重 | 无本轮训练 | — | 原模型 scope 与运行配置保留 |
| R1 v1 | 1773 train examples，1544 个 native train turn；另 19 validation；总样本 1559 ANSWER＋233 READ | 7,800,825 / 355,628 | 1,405,076.37 ms；56 optimizer steps | LoRA、合并权重、manifest、连续物理采样已按训练入口保存 |
| R2 | 2264 个采样后的 train examples＝1773 parent＋153 课程×3＋4 个新 P1 样本×8；1544 个实际 native train turn | 10,759,242 / 597,099 | 1,952,039.975 ms；71 optimizer steps | 从 R1 adapter 继续，实际四 GPU `0,4,6,7` 训练、合并和服务启动完成 |

R1 训练分区持出 loss 为 `0.1608121544`。R2 的持出 loss 为 `0.157603919506073`。
两轮训练都保留实际历史、来源原文坐标、目标复核与排除清单；原 validation 文件由 mix 入口原样复制。
R2 的 2264 是显式过采样后的训练例数，不是 2264 个独立 turn。

| 资产名 | checkpoint identity | 容器产物 |
|---|---|---|
| R1 v1 | `contextkv-reader-r1:1eec232b42b67721374c0400b354081842b54c965a20c441b17b4d82a33c567b` | `models/study-reader-r1/`、`models/study-reader-r1-merged/` |
| R2 | `contextkv-reader-r1:7521b3cffa58df5c0973b361f12d9138bcf358c6404a1b6b6a4c6b440f62e768` | `models/study-reader-r2/`、`models/study-reader-r2-merged/` |
| P0 | 实际 prefix 支持资产；具体 identity 读其 manifest | `models/study-support-actual-prefix/` |
| P1 prototype22 | 22 个真实候选答案 / 5 个训练状态；绑定 R1 v1 | `models/study-support-r1-prototype22/` |

R2 的 identity 字符串由现有 reader driver 生成，前缀仍为 `contextkv-reader-r1:`；
报告与配置用独立名称 R2 和完整 digest 区分。P1×R0、P1×R2 均保留跨 reader 迁移标记。
最近实际健康报告：R2 `30006/GPU4`、原 R1 v1 `30004/GPU6` 均为 HTTP 200。
完整训练、数据和许可产物清单见 [training_assets.json](training_assets.json)。

## A：同历史四格

原始目录：`b/fourgrid-prototype22/frozen/`。每格包含实际 generation、audit、reference、
steps/attempts 与离线评分记录。主分为 `project_condition_version_score`。

| 方法 | 完成 QA / 预定状态 | 共同评分状态 | 主分 | 相对 P0_R0（百分点） |
|---|---:|---:|---:|---:|
| P0_R0 | 13 / 14 | 13 | 0.9230769231 | 0 |
| P0_R1 v1 | 14 / 14 | 13 | 0.9461538462 | +2.3077 |
| P1_R0 | 14 / 14 | 13 | 0.9615384615 | +3.8462 |
| P1_R1 v1 | 14 / 14 | 13 | 0.9769230769 | +5.3846 |

同 reader 的 P1−P0 分别为 R0 **+3.8462**、R1 v1 **+3.0769** 个百分点。
各格的在线总 token、GPU/Host GiB·s、完整响应 P95 与逐项错误数待从该批成本及评分产物填入。
预算失败的原 attempt、原始步骤与已发生费用保留；共同集表不把该题补成零分或成功。

| 后续对照 | 状态 | 产物 / 待填 |
|---|---|---|
| P0/P1×R2 冻结实际历史 | 待实际运行/评分 | 使用独立 R2 run config 和 identity |
| 各方法自身 P3 开发历史 | 待汇总 | 实际问题/回答、终态失败、会话后继跳过与费用逐条登记 |
| P1 自身训练 P3 → R2 | 已得到 4 个有效新样本，按 8 倍混入 R2 | `runs/study/reader-r2-data/new-p1-runtime/` |

早期 sixgrid-v1 的共同 12 状态 E0/inferred 为 R0 0.8708、R1 v1 0.7333；
这批早期诊断没有显示首轮 reader 改善。它与本表的状态/运行版本分开报告。

## B、C 与联合决策

| 项目 | 已实测 | 尚待登记的结果 |
|---|---|---|
| B v3 真实反事实收益拟合 | 8 个 train 状态、4 个 dev 状态，176 次 model call，721867 ms，fit 完成 | reader/checkpoint 绑定、持出误差、实际滚动决策质量与成本 |
| C 原生前缀机制 | 固定 token 配对已执行；完整 READ/ANSWER/REDUCE 成本采集与拟合已推进 | 最终成本资产、留存/TTL/压力实验与未来复用价值结果 |
| 更新 B × 更新 C | 入口与合同已实现 | 四格实际完成数、共同题集质量、完整成本与相对当前候选的收益 |

固定 token 的 C 配对中，GPU 组多命中 128 token、少回载 128 token，响应约少 10.4 ms；
Host 组多命中 128 token、少重算 128 token、多回载 128 token，响应约多 17.0 ms。
实际 READ/REDUCE/ANSWER 与负载实验另列，避免将该局部配对直接外推为在线收益。

## 公开 cohort、开发冻结与测试

当前声明五个公开开发候选：`P0_R0`、`P1_R0`、`P1_R1`、`P0_R2`、`P1_R2`。
R0/R1 分支已运行，分数与完成数待主 agent 汇总；R2 分支排在成本/收益阶段之后。
开发选择记录、最终入选配置、source/weights SHA 与冻结时间待实际 freeze 产物填入。

| 数据轨道 | 开发 cohort | 测试 cohort | 已完成数据准备 | 质量结果 |
|---|---|---|---|---|
| MultiDoc2Dial | 15 turn / 2 conversation | 预定 2 conversation；turn 数读冻结 manifest | 实际 native split 与完整实际历史 | 待填 |
| 2WikiMultihopQA | 40 条 | 互斥 40 条 project-heldout | 作者官方镜像有标签 dev 共 12576 条，固定 hash 40/40 | 待填官方 answer EM/F1、support/evidence/joint 与完成率 |
| Loong / 长文 | 2 个独立训练来源构造的 long_dev | 公开 set1 固定 hash 4 题 | 4 题的全部 17 个原文实例已获取 | 待填 0–100 rubric、完成率、全部 READ/REDUCE/ANSWER 成本 |

2Wiki 的 40/40 来自官方有标签 dev 的项目划分，保留全部原始段落和 distractors。
其 source、cohort 和评分协议分别存于
`data/2wikimultihopqa/source/source_manifest.json`、
`data/2wikimultihopqa/cohort40/cohort_manifest.json`、`study_spec.yaml`。

Loong 的选择仅按 `sha256(canonical_json([42,"loong","test",original_id]))` 排序取 set1 前 4 题，
没有按参考答案选择题目或文档。实际域分布为 **1 paper、2 legal、1 financial**，
**17 个原文实例全部保留**；Range 导出墙钟 **61,392 ms**。
官方 ZIP HEAD 大小 `33,915,268` bytes，ETag `872F4FCE1127BB930A41D7ECEC293562-4`。
此次只获取中央目录和所需完整成员，不记为整 ZIP 下载或全 323 题覆盖。
导出入口产物为 `data/loong/set1-cohort4/{tasks.jsonl,cohort_manifest.json,study_spec.yaml}`；
逐段 Range、ETag、CRC 和成员 SHA 由其 `source/` 原始回执保存，实际输出目录由主 agent 最终核对。

| 冻结/评分项 | 结果 |
|---|---|
| 公共开发逐方法质量、完成/评分分母 | 待填 |
| 预定权重与开发约束验收 | 待冻结 manifest 填入 |
| 入选方法 / 冻结时间 / 源码提交 / 资产与配置 SHA | 待填 |
| 公开测试逐题指标与共同集对照 | 待填 |
| 同物理容量并发、停顿、切题、资料回访负载 | 待填 |
| 消融：任务状态、组合、效用、ANSWER plan、READ、成本 | 待填 |

## 机制结果与错误保留

早期 R1 长文运行的实际 REDUCE 为 36 条、655000 g，最终 ANSWER 却给出 3 条、6 kg；
同代码早期 R0 为 36 条、596000 g，最终同为 3 条、6 kg。新 balanced R0 已实际得到
36 条、666000 g，最终 36 条、666 kg；匹配 R2 结果待填。
多跳任务此前停在 Institute→Arbor，没有完成 1997 的三来源链；新配置结果待填。
原始错误答案、claim 来源、最终输入与计算结果均保留，未改写成期望值。

Task/READ 明确预算终态与单题 judge 坏 JSON 分开登记：前者保留已付费模型调用、
0 QA 和后继跳过；后者保留原 judge 响应、token/时间、失败状态与缺失指标，继续其他题。
基础设施错误保留 failed cell/stage，调度继续独立方法与数据集，最终状态仍为失败或部分失败。

## 完整成本登记

| 阶段 | 已知 token / 时间 | 物理积分、调用账本与待填项 |
|---|---|---|
| 数据导出 / 构造 / prepare | CPU 阶段 GPU 设备列表为空 | 各 data 目录的 `*_cost*.json`、`*_physical_resources*.json` |
| 教师复核 / 课程配对 | 逐次原始调用与接受/排除结果已留存 | `review_calls.jsonl`；实际 input/output tokens、GPU/Host GiB·s 待汇总 |
| R1 train | 7,800,825 输入；1,405,076.37 ms | `models/study-reader-r1/{training_cost.json,train_cost*.json,train_physical_resources*.json}` |
| R2 train | 10,759,242 输入、597,099 监督；1,952,039.975 ms | `models/study-reader-r2/` 中同名逐 rank、连续物理与阶段账本；积分待填 |
| 验证 / checkpoint / merge | 与 train 分开保存 | 各模型目录的阶段窗口、merge 成本；待汇总 |
| 在线 A/B/C 与 reader | 辅助、READ、REDUCE、ANSWER 和失败调用全部计入 | 各 attempt/StepRecord/native receipt/batch measurement segments；待汇总 |
| 离线评分 | 单独于在线生成 | `scoring/` 的 judge 原响应、token/耗时、物理观察与 scoring-stages；待汇总 |

训练 CUDA 同步墙钟与连续 GPU/Host 积分分别保留；不将 torch 峰值显存当作完整成本。
首个 QA 源码提交为 `edb20e7`；最终交付提交与冻结后的必要验证结果待填。

## 交付核对

- 代码与配置：最终源码提交、运行配置、Task/READ 协议、冻结预算与设备映射。
- 数据与资产：[training_assets.json](training_assets.json)，包含实际 split、原文坐标、目标来源、采样倍率、checkpoint 和许可来源。
- 结果与审计：逐题答案、来源、步骤、原生回执、评分失败、质量分母与共同集。
- 费用与取舍：训练采集、训练、验证/merge、在线辅助/reader、失败与评分各自独立记账。
- 待填写的最终结论：公开开发/测试质量、完整成本、负载与消融完成后据冻结产物填入。

## 最终批次登记（06:50）

完整负载批次 `runs/study/capacity-r0-joint-v1` 已结束，预定 56 QA。8 并发下 B6_native 和 S3_native 均完成 16/16，完整响应 P95 分别为 113811.5896 ms 和 94399.0444 ms，后者低 17.0567%。其他格及质量由最终只读索引按原产物导出。

B 的当前 READ prompt 真实收益资产为 `models/study-plan-gain-r0-bridge-v4.json`，执行与 native 成本资产为 `models/study-plan-r0-bridge-v3-execution.json`、`models/study-plan-r0-bridge-v3-native/native-assets.json`。滚动开发共同 6 QA 中 B3 / S3 λ0 / S3 λ1e-5 的 RB_llm 分别为 0.8 / 0.633333 / 0.566667，开发冻结保留 B3；两档 S3 和 B6 在另行预登记的测试对照中继续评估。

C 当前固定 token/DAG 数据为 `runs/study/c-current-r0-g5-v4`，共 4 个配对窗口，train/dev 各 2。完整特征和 count-only 的 dev 覆盖均为 0/2；scope-only 为 2/2，MAE 为 4452.695606 ms，开发策略未选择正收益留存动作。此开发样本支持继续使用 native incumbent。

公开多跳 V4 保留全部原文和 distractors；仅在同一 8192 body 预算及实际 32K 模型窗口允许时直接回答，超预算维持 READ。此时 P0/P1 的证据策略头不参与选择，明确视为同输入 reader 对照。12 题开发：R0 完成和字段评分均为 12/12，答案 F1 59.26；R1 完成 12/12、字段评分 10/12，F1 49.17。原始全文答案指标、字段失败与费用独立保存。
