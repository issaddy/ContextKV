# B 的阶段成本与运行配置

`profiling.sglang_probe → profiling.fit → profiling.stage_fit → CostModel`负责采集、拟合和装配。H200/Qwen3-8B的成本文件为`configs/calibration/qwen3-8b-h200.json`；`configs/qwen3_8b_cost.yaml`使用`admission_estimator: native_budget`，已在容器完成装配与运行身份/采样绑定检查。完整系统结果见[数据与评价](../data_and_evaluation.md)。

## 统一装配

`controller.config.assemble_cost_model` 接收配置和已读取的 artifact/measurement，不创建引擎、数据库或目录。服务、成对采集/重放和长度拟合均使用它：artifact 提供已拟合设置，显式成本开关和估计器覆盖对应设置，最后统一解析采样、system prompt 与有效长度先验。文件读取由 `load_cost_model` 完成；调用方继续核对模型、缓存和数据身份。

成本 artifact 的采集规模、拟合误差、历史长度分布和原始文件入口集中见[阶段成本标定记录](../data_and_evaluation.md#阶段成本标定记录)。耗时测点与输出长度先验分别检查适用范围，不因可复用时间测点而沿用不匹配的回答长度分布。

## 阶段边界与决策含义

- `forward_first`：服务端forward start到首token的墙时，包含执行与调度交互。
- `pre_forward`：`server_ttft_ms - forward_to_first_token_ms`，包含admission、等待和恢复的联合前置区间。
- `decode`：实测上下文区间内每个后续 token 的成本；首 token 已由 forward 计入。
- 控制器本地候选构造、报价与提交开销另计，不和服务端 TTFT 重复累加。

在线使用实测forward/decode、c1单请求准备项和[原生工作量等待估计](../research/admission_wait_model.md)。`native_workload`与`native_budget`以解析模型估计异质队列，后者使用原生预算观测。`selection_eligible`表示可参与选择，等待项的`calibrated`为false。

无工作量快照、配置/采样不匹配、所需执行时间超支持域、在途传输或部分恢复时，沿用已有简单回退。经验时间误差覆盖 forward/decode；等待均值与 P90 的差描述两种风险情景。选择价值由完整闭环效果判断，单臂预测 MAE 用于定位问题。

输出长度是已完成的实际服务长度，受生成预算限制。等待估计使用 `E[L-g | L>g]` 及对应经验 P90；若已生成长度超过观测尾部，使用已知剩余预算并标记原因。标定时 `ignore_eos` 强制产生的固定长度不充当自然回答长度分布。

实际提交后，`observe_output` 记录最近 128 个服务完成长度，含预算截断的真实服务长度；幂等重放不重复计入。这些在线统计优先用于后续原生工作量的服务时间预测，原生容量预留和 forward/decode 模型保持不变。追加和重建的后规划控制耗时各保留最近 32 个观测；无该动作观测时使用共同均值。

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

预测特征使用准备时的`(P,U,L,c)`及声明容量/预算；执行中的前缀位置变化、实际恢复量、输出长度和耗时单独记录为结果。

拟合先求batch均值，再对batch等权。同一forward特征单元的预算、重复和并发成员在OOF中保持同折。经验P99使用支持域内的留出残差，同时报告残差数、独立单元数及最大误差。

`CostModel`绑定fingerprint、cache configuration和采样recipe，runtime manifest记录测量硬件与环境。更换模型、设备、池配置或recipe时，按实际条件评估测量可复用范围，并对决策所需的缺口补采。

## 复现与必要补采

已保存产物支持直接加载和离线复算。补采使用独立服务并设置`CONTEXTKV_ENABLE_PROFILING=1`；准备接口会改变缓存状态，正常服务保持关闭。启动条件见[部署说明](../deployment.md)。

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
