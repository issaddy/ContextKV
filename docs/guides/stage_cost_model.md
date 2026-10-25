# B 的阶段成本与运行配置

`profiling.sglang_probe → profiling.fit → profiling.stage_fit → CostModel` 是唯一的采集、拟合和装配链。当前已产出 H200/Qwen3-8B 的 `configs/calibration/qwen3-8b-h200.json`；`configs/qwen3_8b_cost.yaml` 使用显式 `admission_estimator: native_budget`，并在容器通过装配及保存运行身份/采样绑定检查。完整系统结论维护在[当前实现清单](../implementation_plan.md)，本指南只说明成本机制和复现入口。

## 当前实测范围

- 固定 SGLang、Qwen3-8B、BF16、GPU 16384 槽、实际 Host 54272 槽、page 64、FA3、chunk 8192、FCFS；HiCache 为 layer_first/kernel/write_through、TMA 关闭，`sample_pool=true`。
- 20 个明确 case，覆盖选定冷/GPU/Host/mixed 前缀及并发 1/2；60 个正式请求、30 个预热请求、84 个前缀物化调用。并发 mixed 的页对齐与总工作集在执行前检查，未执行组合网格。
- 采集 run ID 为 `452258dcb15a42bdbcd333878dd141fc`，共 256 条 measurement；拟合得到 20 个 forward 点和 14 个 decode 区间。原始 `profile.jsonl`、`profile.telemetry.jsonl`、`profile.manifest.json` 和 `runtime.json` 在 `runs/mainline-delivery/`。发行包中的 artifact 保留这些来源绑定，字节与原测量产物相同。
- 复用同模型、同采样 recipe 的 159 个既有开发输出长度，来自 22 个原始 dev 会话。全局均值先验为 580，P90 为 1024；在线再按已完成会话历史收缩。该先验来自旧回答协议，比当前简洁协议的实际输出更长，因此冷启动长度估计仍有偏差；本次没有用评测输出回填并重跑。没有用当前待生成答案或质量标签预测长度。
- whole-cell OOF 的 forward MAE 为 52.54 ms；留出支持域内 38/60 请求的 MAE 为 47.54 ms，经验 P99 绝对残差约 202.35 ms。这是插值支持诊断，不是策略净收益或统计覆盖保证。

测点由已有候选的提交前输入/前缀形状选择，未按新答案或时延挑选。原始 case、选择记录和运行命令保存在同目录；本地公开副本位于 `runs/mainline-delivery/remote/mainline-delivery/`。旧模型 artifact 保留为历史证据，不用于当前运行，也不触发旧模型实验。

## 阶段边界与决策含义

- `forward_first`：服务端 forward start 到首 token 的 wall interval，包含该阶段实际执行与调度交互，不称作 CUDA kernel 时间。
- `pre_forward`：`server_ttft_ms - forward_to_first_token_ms`，是 admission/等待/恢复的联合前置区间，现有记录不将它伪拆成独立精确时间。
- `decode`：实测上下文区间内每个后续 token 的成本；首 token 已由 forward 计入。
- 控制器本地候选构造、报价与提交开销另计，不和服务端 TTFT 重复累加。

在线使用实测 forward/decode、c1 的单请求准备项，以及[原生工作量等待估计](../research/admission_wait_model.md)。同质 burst 的前置时间不能直接替代异质会话队列；`native_workload` 与 `native_budget` 都是显式解析估计，后者还使用原生预算观测。`selection_eligible` 表示允许该模式参与选择，`calibrated` 仍为 false，不将这两个字段混用。

无工作量快照、配置/采样不匹配、所需执行时间超支持域、在途传输或部分恢复时，沿用已有简单回退。经验时间误差只覆盖 forward/decode；等待均值与 P90 的差描述风险情景，不冒充经验置信误差。选择价值由完整闭环效果判断，单臂预测 MAE 仅供定位问题。

输出长度是已完成的实际服务长度，受生成预算限制。等待估计使用 `E[L-g | L>g]` 及对应经验 P90；若已生成长度超过观测尾部，使用已知剩余预算并标记原因。标定时 `ignore_eos` 强制产生的固定长度不充当自然回答长度分布。

## Artifact 接口与来源

沿用 `version/settings/prefill_points/decode_points`；forward 点仍是 `LookupPoint`，`scope="forward_first"`、`source="hardware"`，采用现有 lookup 插值。`stage_model` 保存：

```text
version: 1
waiting_calibration_scope: homogeneous_burst_only
forward: features [prefix_tokens, suffix_tokens, host_load_tokens, concurrency]
         cells, OOF residuals, empirical error quantiles
pre_forward: 上述特征加 requested_output_budget、gpu_capacity_tokens
             online_heterogeneous_queue_calibrated: false
total_ttft_reference, total_ttft_budget_reference
development_output_lengths, validation
```

特征使用准备时已知的 `(P,U,L,c)` 和实际声明容量/预算。执行中 GPU 前缀漂移到 Host 时保留记录，不删除不利样本，也不把事后恢复量填回已知特征。当前实际输出长度、未来队列顺序、耗时排序不入特征。

拟合先按 batch 均值、再对 batch 等权。同一 forward 特征单元的预算、重复和并发成员在 OOF 中保持同折。经验 P99 使用受支持留出残差，同时保留残差数、独立单元数及最大误差；不把相关请求数当作独立样本数。

`CostModel` 绑定实际 fingerprint、cache configuration 和采样 recipe；这些身份检查不等同于自动确认另一台 GPU 的硬件等价。运行硬件与环境来自对应 runtime manifest。更换模型、池配置或 recipe 后，应先判断现有范围能否复用，再决定是否需要定向补采。

## 复现与必要补采

已保存产物可以直接加载和离线复算，无需重启模型。只有实际决策遇到关键成本空白才补采，使用独立服务及 `CONTEXTKV_ENABLE_PROFILING=1`，因为准备接口会改变缓存状态。正常服务不启用冷缓存标定；本次成本配置的启动条件见[部署说明](../deployment.md)。

`--cases-file PATH` 接收有序 `PrefillCase` JSON 数组，每项包含 `prompt_tokens`、`prefix_tokens`、`output_tokens`、`concurrency`、`cache_state`；mixed 另含 `device_prefix_tokens`。它不与网格参数混用，执行前按真实页大小和上下文预算校验。manifest 的 `planned_generation_requests` 分别记录正式、预热、前缀准备和合计；失败时实际完成分母以 telemetry 为准。

原生 telemetry 保存 `run_id/repetition/warmup`、request index、receipt 和阶段时间，直接进入拟合，不需要外部 compact 转换步骤：

```bash
python -m profiling.stage_fit \
  --telemetry selected=runs/NEW-QWEN3/profile.telemetry.jsonl \
  --measurements runs/NEW-QWEN3/profile.jsonl \
  --profile-manifest selected=runs/NEW-QWEN3/profile.manifest.json \
  --base-artifact runs/NEW-QWEN3/cost-lookup.json \
  --development-report runs/NEW-QWEN3/development-lengths.json \
  --output runs/NEW-QWEN3/cost-stage.json \
  --validation runs/NEW-QWEN3/stage-validation.json
```

新采集/拟合使用新的文件前缀，已有产物不覆盖。warmup 不入拟合；缺阶段标签、请求缺失或来源身份不一致明确失败，不从客户端时间补造原生阶段。`--development-report` 必须与 base artifact 已绑定来源一致。多组采集可重复 `--telemetry GROUP=PATH` 与同名 `--profile-manifest`。原 `--compact` 入口仅供读取旧测量协议，不改变当前模型范围。
