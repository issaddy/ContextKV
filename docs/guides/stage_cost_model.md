# 完整成本与标定

B在线加载`CostModel`及其数据；`profiling.sglang_probe → profiling.fit → profiling.stage_fit`是生成这些数据的离线工具链。当前成本文件对应H200、Qwen3-32B、32768个GPU KV槽位和16GB Host配置，原始数据与验证范围见[评价记录](../data_and_evaluation.md)。

## 决策

对共同证据下的候选a，比较：

`T(a) = 控制开销 + 准备/等待 + forward到首token + 后续decode`。

- `forward_first`是服务端forward开始到首token的墙时。
- `pre_forward = server_ttft − forward_first`，包含准入、准备与恢复。在线取单请求准备测点，并按当前队列估计额外等待。
- decode按实测上下文区间逐token累计；首token已计入forward。
- 本地候选、报价和提交耗时另计，按动作维护完成观测。

总TTFT原始测点用于校验，在线阶段模型不再重复加总TTFT。动作按预计完整耗时选择；预测误差接近时考虑工作集和输入长度。缺少观测或超出测量范围时执行固定回退策略。

## 原生预算与服务长度

`admission.py`使用原生运行预算、请求身份、队列、页大小和可用容量，按FIFO计算请求何时具备准入空间。原生输出预留使用请求预算与引擎比例；服务时间使用已完成输出的经验分布。对已生成g个token的请求，取`E[L−g | L>g]`及经验P90情景；没有可用尾部样本时使用剩余请求预算。

`CostModel`保存近期128个完成长度，优先用于等待预测；当前会话近期回答与开发集均值/P90形成生成长度先验。强制输出长度的标定请求只测时间，不进入自然回答分布。

等待是解析估计，实测forward/decode支持范围和等待适用性分别记录。`selection_eligible`表示该估计可用于当前选择；不能把它当作异质队列下已经验证的时延精度。

## 装配与数据

`controller.config.load_cost_model`读取文件，`assemble_cost_model`统一线上与离线装配。成本文件保存测点、准备阶段表、输出长度分布和来源；模型、缓存配置、采样及回答提示在启动时核对。重新标定使用实际目标模型和资源配置。

阶段拟合先对同一并发批次求均值，再对批次等权。验证按完整特征单元分折，预算变体及同批请求留在同一折；留出误差与覆盖率另存验证文件。

## 命令

使用独立生成服务并设置`CONTEXTKV_ENABLE_PROFILING=1`，运行常规[启动配置](../deployment.md)。`--cases-file`接收`PrefillCase`数组：prompt/prefix/output tokens、concurrency和cache_state；mixed另带device_prefix_tokens。

```bash
python -m profiling.sglang_probe --config configs/qwen3_32b_scope.example.yaml   --cases-file CASES.json --output runs/calibration/profile.jsonl
python -m profiling.fit runs/calibration/profile.jsonl   --profile-manifest runs/calibration/profile.manifest.json   --development-report runs/calibration/development-lengths.json   --output runs/calibration/cost-lookup.json
python -m profiling.stage_fit   --telemetry selected=runs/calibration/profile.telemetry.jsonl   --measurements runs/calibration/profile.jsonl   --profile-manifest selected=runs/calibration/profile.manifest.json   --base-artifact runs/calibration/cost-lookup.json   --development-report runs/calibration/development-lengths.json   --output runs/calibration/cost-stage.json   --validation runs/calibration/stage-validation.json
```

`development-lengths.json`来自同模型、采样和回答协议的已完成开发回答；包含实际输出长度及模型身份。原生telemetry中的run、批次、预热标记和回执直接进入阶段拟合。正式服务关闭profiling。

需要诊断APPEND/REBUILD决策时，可用`profiling.decision_pairs freeze/collect/analyze/rescore`复用冻结输入和真实结果。其输出解释动作差值与错误选择代价，完整问答效果仍由公开任务评价。
