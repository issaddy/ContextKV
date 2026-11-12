# 当前课题运行入口

统一业务链为 `DocumentQA.ask → StudyEvidenceProvider → Controller → RequestCompiler/ReadingExecutor → SessionStore`。固定 Qwen3-32B，使用容器默认 `python3`。模块与方法见[架构](../architecture.md)，实验口径见[任务书](task_spec.md)。

## 运行资产

| 内容 | 当前入口 |
|---|---|
| 模型服务与原生补丁 | [部署](../deployment.md)、`scripts/launch_sglang.sh` |
| 问答与阅读配置 | `configs/study.yaml`、`configs/qwen3_32b.yaml` |
| 公开 benchmark | `configs/study_matrix.yaml` |
| 实际历史重放 | `configs/study_system_matrix.yaml` |
| 构造条件/版本及统计任务 | `configs/study_mechanisms.yaml` |
| 辅助训练 | `configs/auxiliary_training.yaml` |
| 数据集评分 | `bench.study_scoring` |
| 最终对比汇总 | `profiling.final_study_index` |

代码、配置和必要结果数据可以同步；模型、向量及大文件保留在实验容器。GPU 使用范围为 0、1、2、3，实际服务和辅助模型按配置分配设备。

ContextPilot 复用作者 CPU 索引与排列：

```bash
python3 scripts/prepare_contextpilot.py --checkout vendor/contextpilot
```

`execution.cache_compilation` 使用 `{mode: contextpilot, checkout: vendor/contextpilot}`；`prefix_retention` 使用 `mode: demand`。默认分配器为 `next_use`，期限候选 15/60/180 秒，保护容量 8192 token；源配置可以声明实际取值。`next_use` 与 `paired_value` 均通过 `pricing_asset` 加载配对估值，没有有效净收益定价时使用 Native。

joint observed 动作默认使用 `action_policy: validated`。同 reader/profile 的旧协议或未通过开发集准入的资产只读取元数据，实际回退规则并记录原因；显式 `learned` 用于严格绑定的研究消融。新协议需要重新采集 frozen branches，见[机制修复与验收](mechanism_repair.md)。更新 Native overlay 后再运行并发软配额实验。

## 数据与选题

优先复用已有原始数据、语料和索引。MultiDoc2Dial 任务按当前 user 紧接的一条有效 agent 回复导出，见[接入说明](../guides/doc2dial.md)；MT-RAG Cloud、VersionQA 和 Loong 使用[数据与评价](../data_and_evaluation.md)所列入口。

每个 benchmark 独立选择 50 题，同一 benchmark 的方法共用清单。任务数由清单决定。实际历史重放按会话顺序保留连续问题。`--cohorts` 的 YAML 格式为：

```yaml
cohorts:
  test:
    multidoc2dial: [task-id-1, task-id-2]
    mtrag: [task-id-1, task-id-2]
    versionqa: [task-id-1, task-id-2]
    loong: [task-id-1, task-id-2]
```

示例中的 ID 替换为对应数据集的实际 task ID。使用 `--datasets` 选择本次数据集，`--split` 默认 `test`。参考历史和实际历史分别运行、分别报告。

## 主比较

默认方法为普通文档问答 `RAG_native`、ContextKV `context_cache_demand` 和 LlamaIndex `LlamaIndex_native`。工作流为普通基线与 LlamaIndex 物化混合召回、发布版重排和常规上下文配置；ContextKV 使用本次冻结的任务、选材和收益资产。

```bash
python3 -m bench.capacity_workflow \
  --config "$STUDY_CONFIG" --matrix configs/study_matrix.yaml \
  --datasets multidoc2dial --cohorts "$COHORTS" \
  --gain-asset "$B_GAIN" --reader-base-url "$READER_URL" \
  --online-gpu-devices "$ONLINE_GPUS" \
  --contextpilot-checkout vendor/contextpilot \
  --concurrency 1 4 --output "$PUBLIC_OUTPUT"
```

按数据集分批执行上述命令。默认每种并发运行一次。多轮实际历史使用 `configs/study_system_matrix.yaml`，构造任务使用 `configs/study_mechanisms.yaml --split dev`。MTRAG-UN 按需使用独立配置。

默认调度沿会话前后依赖推进、思考间隔为零；已有真实固定到达记录时传 `--arrival-plan`。只运行普通基线、LlamaIndex 或规则决策时提供 `--reader-profile-id`，完整方案使用 `--gain-asset` 绑定 reader、输入协议和输出额度。

## 机制对照

| 问题 | `--methods` |
|---|---|
| 来源排列的收益 | `context_native context_cache_order` |
| 需求保留的新增收益 | `context_cache_order context_cache_demand` |
| 动作收益决策 | `Rule_contextpilot_joint_demand context_cache_demand` |
| 跨轮事实复用 | `context_no_fact_reuse context_cache_demand` |

对照共用未改变的模块及本次选题，完整方案结果可以复用。根据主比较结果选取对应数据集和并发补充。

少量固定输入缓存配对使用相同原文、预热、竞争请求和最终问题：

```bash
python3 -m bench.cache_cooperation --config READER.yaml \
  --base-url "$READER_URL" --output runs/cache-cooperation/result.json
```

该入口默认低压力、单请求竞争、四请求竞争各一对，报告正文获取工作、完整时间和控制开销。更细的期限估值沿用 `profiling.retention_trace` 与 `profiling.retention_pair_workflow`，Native/TTL 使用同一请求轨迹和观察窗口，窗口覆盖所选期限。

## 收益资产与辅助训练

真实分支完成后生成紧凑训练样本，再在 CPU 拟合收益与成本：

```bash
python3 -m training.plan_gain_completion finalize \
  --collection TRAIN-CAPTURE --output TRAIN-LABELS --template-config READER.yaml
python3 -m training.plan_gain_completion finalize \
  --collection DEV-CAPTURE --output DEV-LABELS --template-config READER.yaml
python3 -m training.plan_gain_completion fit \
  --train TRAIN-LABELS/effective-gains.jsonl --dev DEV-LABELS/effective-gains.jsonl \
  --output GAIN.json
```

样本保留预测特征、真实质量与费用标签、预算及训练划分。完整上下文和执行记录只用于采集与标注，完成后清理。动作、分包或输入语义改变时补采对应分支。质量按决策状态等权拟合，成本按具有完整实测费用的唯一终止分支拟合，包含成功与失败、未知费用不补零；线上执行一个动作后重规划。`fit --dev` 把开发报告与资产和隔离会话绑定，仍需独立 test 复验。

| 需要解决的问题 | 入口 |
|---|---|
| READ、REDUCE、ANSWER 的阶段错误 | `training.reader_audit export/review` |
| 任务和查询适配 | `training.a_workflow prepare-capture/prepare-labels`、`training.task_labeling`、`training.train_models --stage task/task-eval` |
| 支持关系和集合偏好 | `training.build_examples`、`training.train_models --stage support/support-eval`、`training.preference_rollout` |
| 当前真实状态的动作配对 | `training.collect_plan_gains`、`training.plan_gain_workflow --mode observed_actions` |
| 固定输入的证据策略对照 | `training.a_workflow compare`、`bench.evidence_policy_comparison` |

辅助训练按会话隔离，沿当前 cohort 复用真实历史。任务学生尚未输出跨轮 READ 合同；只接入查询编码器时使用 `retrieval.task_query_asset` 并保留 teacher 解释。支持模型只由 `selection.mode: support` 加载，需求保留通过 `pricing_asset` 加载缓存配对估值。

## 结果与记录

本轮结果见[完整报告](final_results.md)和[指标文件](../../results/evaluation-20261112.json)。容器保留 `runs/current/cohorts.yaml`、`runs/current/config/` 的实际配置及 `runs/current/training/` 的必要监督；当前查询适配器为 `models/study-task-budget-loop`，动作资产为 `models/study-plan-gain.json`。Qwen3-32B 权重不变。

当前容器以 `source runs/current/environment.sh` 选择已安装的 CUDA 12.9 工具链与主机驱动库，仍使用默认 `python3`。任务与支持训练输入各保留一个压缩 JSONL，重新训练时用 `gzip -dc INPUT.jsonl.gz > INPUT.jsonl` 解压；收益训练直接读取紧凑 `plan-gain-train.jsonl`、`plan-gain-dev.jsonl`。

每个 job 对应数据集、方法和并发。`capacity-plan.json` 保存本次配置，`workflow.json` 保存最终指标。

生成与评分共用一个 `results.jsonl`；`task_metrics.jsonl` 按 task_id 合并成功、失败与未提交阻塞题，保留每题分数、耗时、费用已知下界和未知原因。完整历史、候选池、计划、token 序列和评分原文只在执行或实际训练采集中使用。Native 退役或需求租约释放尚未确认时保留恢复状态，确认清理后再删除 job 临时目录。

```bash
python3 -m profiling.final_study_index --artifact NAME=WORKFLOW.json --output RESULTS
```

最终报告各 benchmark 的原始质量指标、完成率、完整响应 P95、在线 token、GPU/Host 占用；缓存对照增加 GPU 命中、Host 回载和重算 token。直接比较本次测得的质量差与 P95 降幅。
