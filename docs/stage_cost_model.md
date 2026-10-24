# B 的阶段成本 artifact 与开发验证

`profiling.stage_fit` 已实现可运行的离线阶段训练入口，产出 `runs/stage-model-dev-v2/cost-stage.json` 和逐请求 OOF `validation.json`。它提供执行到首 token 的 lookup、联合前置区间的经验单元、总 TTFT 对照和已完成开发输出长度分布。`CostModel` 已接入该 artifact，使用显式 `admission_estimator: native_workload` 选项；旧运行配置已退役；保留拟合器与原始证据供只读复核，新Qwen3运行不得加载该7B artifact。离线训练复用已采集数据，不改写原始 artifact。

在线成本使用本表的 forward、原实测 decode，以及[原生工作量等待估计](admission_wait_model.md)。候选的单请求准备项按 `(P,U,L)` 插值 c1 的 pre_forward 单元，合并两个输出预算；没有把同质 c4 的平均等待套到异质队列。`selection_eligible` 表示该显式实验模式可比较，`calibrated` 仍为 false；默认 strict 模式继续要求完整成本已校准。缺少工作量快照、配置不匹配、所需时间超支持域、在途传输或部分恢复时回退。经验时间误差只覆盖 forward/decode，等待均值/P90 差是启发式风险情景，不会加进经验置信误差冒充覆盖保证。

## 标签、特征与数据来源

- `forward_first`：服务端 forward start 到首 token 的 wall interval，包含该阶段实际执行与调度交互，不称作 CUDA kernel 时间。
- `pre_forward`：`server_ttft_ms - forward_to_first_token_ms`，是联合 admission/等待/恢复前置区间；现有记录不能将它再精确分成独立 queue、host restore 与控制耗时。
- `total_ttft`：原 server TTFT，保留为对照。三者不与客户端 TTFT 混用，也不包含控制器本地候选构造。

输入为本地既有 `stage-timings.jsonl`、完整 16K measurements 和四个原始 manifest。compact 有 736 条，其中 184 条 warmup 全部排除，552 条正式请求来自 228 个 batch。compact 的 repetition 全为 null，因此入口复用原 collector 的 request 写入顺序，逐字段核对原 first_token measurement（run ID、case、TTFT、实际 prefix/load/output），恢复原 `(run_id,repetition)`。未按 TTFT 大小排序，不用排序位次或 index 预测等待。

forward 的已知特征为 `(P,U,L,c)`，其中 `n=P+U`；联合前置模型增加 `requested_output_budget`、`gpu_capacity_tokens`。P、L 来自请求提交前的 prepared case/quote，预算来自 case 的请求上限，GPU 容量来自实际 manifest。当前只拟合实际 16384 池，不外推不同 pool/configuration。

36/552 请求虽然准备时是 GPU-ready，执行时其前缀已改从 host 恢复。它们的总可复用 prefix 与 prepared 值仍一致。**不删除这 36 行，也不把其事后 actual_host_loaded_tokens 填回特征**：GPU-ready 的已知 L 仍为 0，实际恢复变化只记诊断，由前置区间标签体现报价到执行之间的压力变化。当前实际输出长度、未来队列顺序和 TTFT 排序也都不入特征。

forward 有 38 个特征单元；加入两个预算 128/512 后，pre_forward 有 76 个单元。每单元先按 batch 求均值，再对 batch 等权；同时保留请求经验 P95/P99/max。并发成员具有共同的可知输入，不能用其中最快/最慢成员的实测排序作为在线已知量。

## Artifact 稳定接口

沿用原 CostModel 的 `version/settings/decode_points`；新的 `prefill_points` 是已有 LookupPoint 结构，`scope="forward_first"`、`source="hardware"`。Decode 点及旧输出先验逐项保留，不用当前矩阵答案重拟合。`settings.prefill_estimator="lookup"`。

新增 `stage_model`：

```text
version: 1
waiting_calibration_scope: "homogeneous_burst_only"
forward:
  features: [prefix_tokens, suffix_tokens, host_load_tokens, concurrency]
  cells: [{features, mean_ms, p95_ms, p99_ms, max_ms, batches, requests}]
  error_source, oof_residuals, independent_feature_cells
  empirical_error_p95_ms, empirical_error_p99_ms, empirical_error_max_ms
  confidence_coverage_guarantee: false
pre_forward:
  features: [prefix_tokens, suffix_tokens, host_load_tokens, concurrency,
             requested_output_budget, gpu_capacity_tokens]
  cells: [...]
  online_heterogeneous_queue_calibrated: false
total_ttft_reference: {features, cells}
total_ttft_budget_reference: {features, cells}
development_output_lengths: {lengths, source, tasks, sessions, truncated_outputs, ...}
validation: {protocol, folds, metrics, ...}
```

前向点的 `error_ms` 使用 whole-cell OOF 中受支持请求的绝对残差经验 P99。没有“至少八个 cell 才有资格”门槛；同时给出独立 cell 数、残差数和 max，明确经验分位不是置信覆盖保证。只有完全没有可用 OOF 残差才沿用点成本本身作为保守误差，并记录原因。

`pre_forward` 的分位值描述同质 burst 的实际离散程度，不是等待硬上限。在线异质队列不能据此设置 calibrated=true，尤其当前 1024 输出预算也不在这两个标定预算的范围内。主控制器可以另用原生已知运行/等待工作量构建显式 analytical/estimated 等待项，但必须与本经验单元的校准范围区分，并另外验证。不能简单把同质 batch 均值加给每个异质请求后宣称队列预测已验证。

## 不泄漏的开发验证与结果

5 折按完整 forward 特征单元划分；同一单元的两个预算、全部重复与 batch 内全部成员始终在同一折。每折只用训练折拟合表，再预测其 heldout cell。独立测试扰动某折所有目标标签后，该折的预测完全不变。这里不是把 552 个相关请求当作 552 个独立样本。

OOF 支持 402/552 请求（72.83%）、168/228 batch、28/38 forward cell。下表是受支持请求以及对应 batch 均值的 MAE；超出留出支持范围的诊断误差仍保存在 JSON，未删去。

| 预测目标/对照 | 请求 MAE ms | batch 均值 MAE ms |
|---|---:|---:|
| forward_first | 27.76 | 24.55 |
| pre_forward | 483.44 | 130.23 |
| forward + pre_forward | 485.55 | 144.86 |
| 总 TTFT，原 `(P,U,L,c)` 特征族 | 523.79 | 211.63 |
| 总 TTFT，同预算/容量特征对照 | 485.53 | 144.87 |

最后一行控制“新增预算特征”的影响：拆阶段后的联合预测与同特征的总 TTFT 几乎相同，**不能把约 38 ms 的请求 MAE 差异全部归因于阶段拆分算法**。分阶段的直接价值是隔离出可部署的执行成本与仍需研究的前置等待，而不是已经证明了整体决策收益。总 TTFT 对照重新使用同一 prepared 特征族，避免用事后恢复量给它额外信息；原旧 artifact 仍原样保留并有来源 SHA。

forward 的 402 个支持残差来自 28 个独立 cell，绝对误差经验 P95=78.53 ms、P99=168.48 ms、max=169.98 ms。全部请求（包括不支持点）的 forward MAE=39.62 ms。联合前置区间在请求级比 batch 均值更难预测，反映同一同质 burst 内先执行与后等待成员差异；不应通过事后 rank 特征把这部分误差“拟合掉”。近零前置区间使 MRE 很敏感，判断时应同时查看绝对毫秒和分组统计。

这些是同池、同质 burst、开发数据的 feature-cell 留出结果，没有异质真实会话队列或跨机器验证，没有使用质量 test，也没有据此宣称在线性能改善。

## 已完成输出长度的剩余工作先验

可选 `--development-report` 必须与 base artifact 已绑定的 development report SHA 一致。当前保存 5 个 dev 会话的 41 个已完成输出长度，截断标记为 0，来源为原 `cloud-dev-p2-output1024/benchmark-9ad65448c01e442aac00b757bc17bcb3.json`，SHA `3703184ba68479100a93d0540a0e0c9d346c0caf2e556bc76bd42406d442c964`。

这些是已完成的实际服务长度，仍受生成预算制约，不称为无限预算自然长度。它们可供主控制器计算有存活样本时的 `E[L-g | L>g]` 及经验 P90，避免已生成长度超过无条件均值后把剩余工作置零。超出观测尾部应使用已知剩余输出预算并标记 fallback；不能拿标定时 `ignore_eos` 强制生成的 128/512 来假装自然分布。

## 运行与验证

`profiling.sglang_probe`的measurement、telemetry和manifest输出使用新的文件前缀。已有任一产物会在访问引擎前拒绝；manifest在缓存准备和生成前独占创建，防止重复命令混写不同run ID并覆盖原始来源。失败仍保留已写测量与分母，不自动覆盖重跑。该工程修复不产生新的标定值。

当前`SGLangProbe`直接在原生telemetry中保存`run_id/repetition/warmup`，request记录还有原index、receipt和`forward_to_first_token_ms`。阶段拟合已能直接读取这些产物，不再要求先运行仓库外的compact转换脚本。下一次**确有必要的新Qwen3采集**可使用以下入口；这里不启动采集，不使用旧模型artifact：

```bash
python -m profiling.stage_fit \
  --telemetry selected=runs/NEW-QWEN3/profile.telemetry.jsonl \
  --measurements runs/NEW-QWEN3/profile.jsonl \
  --profile-manifest selected=runs/NEW-QWEN3/profile.manifest.json \
  --base-artifact runs/NEW-QWEN3/cost-lookup.json \
  --output runs/NEW-QWEN3/cost-stage.json \
  --validation runs/NEW-QWEN3/stage-validation.json
```

多组采集可重复`--telemetry GROUP=PATH`，与`--profile-manifest`逐组一致。直接入口核对原生计时、完整generation、显式run/repetition、原first_token记录的写入顺序及完整并发成员；不是按耗时排序或任意乱序join。warmup不进入拟合。实际GPU→host漂移保留为标签/诊断，输入仍取prepared case的host量。缺阶段标签、缺请求或身份不符明确失败，不从客户端时延补造原生阶段。来源文件在离线边界记录摘要，重拟合的metadata只声明本次阶段输入，原base仍单独绑定。

collector→telemetry→stage artifact的CPU协议闭环及旧compact合同通过41项针对性测试；其中合成时间只验证数据接线，不是硬件标定。记录在`runs/b-current-integration-audit/`。当前仍没有Qwen3可用的实测成本表或在线净收益结论；这一补齐消除了将来必要采集的不可复现人工步骤，没有因此扩大采集矩阵。

下述旧compact命令仅用于复算上文既有历史数据，不启动旧模型或新增实验。旧文件没有repetition时保留这一显式兼容入口，不伪造新telemetry身份：

```bash
python -m profiling.stage_fit \
  --compact runs/remote-h200-capacity-matrix/runs/capacity-calibration/16384/stage-timings.jsonl \
  --measurements runs/remote-h200-capacity-calibration/runs/capacity-calibration/16384/measurements.jsonl \
  --profile-manifest single=runs/remote-h200-capacity-calibration/runs/capacity-calibration/16384/single.manifest.json \
  --profile-manifest concurrent-small=runs/remote-h200-capacity-calibration/runs/capacity-calibration/16384/concurrent-small.manifest.json \
  --profile-manifest concurrent-long-device=runs/remote-h200-capacity-calibration/runs/capacity-calibration/16384/concurrent-long-device.manifest.json \
  --profile-manifest concurrent-long-host=runs/remote-h200-capacity-calibration/runs/capacity-calibration/16384/concurrent-long-host.manifest.json \
  --base-artifact runs/remote-h200-capacity-calibration/runs/capacity-calibration/16384/cost-lookup.json \
  --development-report runs/remote-h200-final/cloud-dev-p2-output1024/benchmark-9ad65448c01e442aac00b757bc17bcb3.json \
  --output runs/NEW-STAGE/cost-stage.json --validation runs/NEW-STAGE/validation.json
```

文件采用独占创建，已有结果不会覆盖。完整 manifest/run_id/源文件 SHA 保留在 artifact metadata，source绑定只在离线入口执行。v1 原型保留，v2 更新了 manifest case 核对与 metadata 内阶段 validation 的标识；数值表和 OOF 结果相同。10 项新测试及 13 项既有 provenance 测试通过，覆盖 warmup 排除、原批次重建、事后恢复不入特征、全 cell/batch 分折、heldout 标签扰动、预算同折、旧点/先验保留，以及开发输出来源拒绝错配。
