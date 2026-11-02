# R1 与 P1 运行清单

R1 已真实训练并合并，当前 reader checkpoint 为
`contextkv-reader-r1:1eec232b42b67721374c0400b354081842b54c965a20c441b17b4d82a33c567b`。
训练数据为 1773 个样本、1544 个训练 turn；另有 19 个训练分区持出样本。
56 step、7,800,825 input token、1,405,076.37 ms 和 heldout loss
`0.1608121544` 属于训练证据。正式 QA heldout 与六格质量对照尚未完成。
原始数据、权重、日志和结果均留在 `sjy-0518`。

## 已确认与仍缺的证据

| 项目 | 现有证据 | 仍缺 |
|---|---|---|
| R1 训练与部署 | LoRA、merge、30004/GPU4 服务就绪 | 独立 QA heldout 和与 R0 的完整可比质量/成本 |
| R1 受控长文 | 24 READ、2 REDUCE、1 QA；REDUCE=36条/655000g，ANSWER=3条/6kg | 目标36条/666kg尚未达成；READ和ANSWER错误分别定位 |
| 同代码 R0 长文 | REDUCE=36条/596000g，ANSWER同为3条/6kg | 完整配对审计与其他格的质量指标；不能称纯R1退化 |
| R1 多跳 | 执行结束，推进到Institute→Arbor | 最终1997与三来源链未通过 |
| pair/correction | 已生成预期表单词项 | literal/source/citation离线审计 |
| P1 utility | 采集、配对、训练和下一轮P3入口已实现 | 当前R1实际偏好、训练资产和持出收益 |
| round2 curriculum | 独立native/constructed_train配对构造与审计代码 | fresh teacher标签、round2训练和效果评估 |
| Incumbent | 当前可运行候选和历史Dense5参考 | 同一开发cohort的选型；尚未宣布最强配置 |

## P1 采集的数据与资产前置条件

配置为 [evidence_policy_r1.yaml](../../configs/evidence_policy_r1.yaml)，初始化资产为
`models/study-support-actual-prefix`。旧 prefix12 偏好不能直接作为当前 R1 奖赏。
采集器从 `models/study-reader-r1/reader_manifest.json` 读取真实 R1 身份，
核对 run config、实际 engine fingerprint 与冻结 judge checkpoint，随后按
reader/profile/scorer 建独立 cohort。

| 输入 | 用途与检查 |
|---|---|
| `models/study-reader-r1/{reader_manifest.json,model_scope.json,r1_run_config.yaml}` | 实际checkpoint、32B配置、模板与服务身份；R1端口30004 |
| `models/study-support-actual-prefix/{support-model.json,heads.pt,encoder/}` | P0支持/集合网络初始化；不把它登记为当前R1的P1 |
| `runs/study/data/task_labels.jsonl` | 原生train、真实QA、已复核grounding与TaskState；旧Need悬空边/turn0占位必须整行排除，不修改原标签 |
| `runs/study/data/sources.jsonl` 与原始 `multidoc2dial_doc.json` | 核对Unicode原文跨度并建立原生结构目录 |
| `data/multidoc2dial/bge-m3-sections` | 固定文档索引；原始检索身份随准备清单记录 |
| `models/bge-m3`、`models/bge-m3-dialogue-query-v1` | GPU1查询编码；复用既有查询LoRA |
| `30001/GPU2`冻结R0 judge | 独立生成后的四分量评分；使用固定R0权重与rubric |

`audit` 是只读资格筛选：检查当前原文指针的实际消息来源/边界、活动条件引用、
Need前驱闭包和环、actual-history审核状态，以及verified grounding逐字匹配。
它输出原样的合格行、整行排除原因和独立采集配置；不删除边、不修补缺失原文。
合格行少于预定批次时应补新的有效Task标签，不能改成reference history或改低数据门槛。

## 设备与首批范围

| 工作 | 逻辑设备/服务 | 物理成本采样 |
|---|---|---|
| R1 READ/ANSWER | 30004，GPU4 | collection包含GPU4 |
| 冻结judge | 30001，GPU2 | collection包含GPU2；不使用六格比较R0的GPU5 |
| BGE query、后继P3的task/history embedding | `cuda:1` | collection包含GPU1 |
| 支持、utility推理 | `cuda:6` | collection包含GPU6 |
| P1 utility训练 | `cuda:6` | 仅GPU6；不计其他驻留reader |

这些配置要求设备编号保持物理原序。可显式使用
`CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7`；不能设单卡可见性后继续使用`cuda:6`。
GPU3留给C服务，GPU5留给匹配R0诊断。P1 collection连续采样集合为`[1,2,4,6]`，
进程PSS包括采集进程与reader/judge服务进程组；utility训练只采本进程组和GPU6。

建议先审计8个真实状态，首个状态执行全部可用干预以检查I/O和计量，
再扩到同一审计文件的8个状态。默认最多6类：relevance、direct_support、
complementary、redundant、真实wrong_condition_version、deleted_premise。
缺少真实条件/版本冲突来源时该类跳过并记录，不改写背景材料冒充负例。
正文4096、最终输出1024、source-first READ保持不变；最多48个候选回答及其judge，
READ/REDUCE数量按真实计划另计。这是小规模训练采集检查，不是持出效果结论。

```bash
python -m training.train_evidence_policy audit --config configs/evidence_policy_r1.yaml --limit 8
# 使用audit输出的collection_config路径。
POLICY_CONFIG=runs/study/reader-bound-preferences/READER_COHORT/audited_collection_config.yaml
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m training.train_evidence_policy collect --config "$POLICY_CONFIG" --limit 1
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m training.train_evidence_policy collect --config "$POLICY_CONFIG" --limit 8
python -m training.train_evidence_policy prepare --config "$POLICY_CONFIG"
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m training.train_evidence_policy train --config "$POLICY_CONFIG"
```

训练前核对：真实质量分量已生成、同题actual history/正文与输出预算一致、
存在非零偏好、相同input hash已排除、原生回执和物理/token成本齐全。
输出为独立`models/study-support-r1`，保留P0。P1训练后再执行`rollout`，
由其自身P3回答产生下一轮reader数据；该步骤尚未实际完成。

## 隔离的官方指标环境

实际smoke环境为`.venv-eval`：evaluate 0.4.6、transformers 4.57.6、
huggingface-hub 0.36.0、datasets 3.6.0、bert-score 0.3.13。
核心输入pin见[requirements-study-eval.in](../../configs/requirements-study-eval.in)；
完整解析锁及共享torch/CUDA库版本应与容器实验清单一起保留。
主运行/训练环境仍使用transformers 5.19.0。

整个线上环境的`contextkv-torch-runtime.pth`已停用，避免将线上kernels一起导入
旧版transformers的评分环境；仅通过明确链接共享torch、nvidia和triton库。
DeBERTa xlarge MNLI固定revision为
`5b07a9086c1dbb79981ff7b05b4d1ad83b3af51c`，直接在指定容器下载，未传输模型。
单个相同句子的BERTScore F1和ROUGE 1/2/L/Lsum均为1，只证明指标环境可运行，
尚未产生该批benchmark的`RB_alg`分数。
