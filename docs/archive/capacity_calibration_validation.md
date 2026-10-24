# 16K 有限池标定核查与 B 的改进方向

> 归档记录：保留当时的设计、结果与复用依据；其中“下一步”和执行计划不代表当前任务。当前工作仅以[实现清单](../implementation_plan.md)为准。

2026-10-24。独立核对了 16K 标定的四组原始 measurements、manifests、合并文件、runtime 和冻结 lookup artifact。**来源、数量与配置绑定均一致，但不能据此称预测准确。** 当前留出结果中 first-token 覆盖率为 57.1%，受支持样本 MAE 为 46.22 ms；c4 长输入还呈现由调度压力主导的多峰分布，现有特征把不同等待条件合成一个点。

本次只分析已经下载的文件，没有修改成本模型、重新拟合、调参、访问引擎或下载完整 telemetry。后补的阶段 compact 也只读分析；六方法矩阵使用的原冻结 lookup 保持不变。

## 来源、数量与暖机

源目录为 `runs/remote-h200-capacity-calibration/runs/capacity-calibration/16384/`。四份 manifest 均 complete/hardware，原始文件 SHA 与 artifact provenance 一致，合并文件逐字节等于四组原文件按既定顺序连接。四个 measurement run IDs 与四份 manifest 恰好对应。

| 分组 | 总观测 | first-token | decode | D2H | H2D |
|---|---:|---:|---:|---:|---:|
| single | 534 | 120 | 120 | 246 | 48 |
| concurrent-small | 792 | 240 | 240 | 245 | 67 |
| concurrent-long-device | 462 | 96 | 96 | 240 | 30 |
| concurrent-long-host | 480 | 96 | 96 | 192 | 96 |
| 合计 | 2268 | 552 | 552 | 923 | 241 |

这对应 552 个正式 probe 请求、各一条 first-token/decode 记录，以及 1164 条 DMA 观测；不能称为 2268 个独立请求样本。76 个 fixture case 各暖机一次、正式重复三次，184 个 warmup probe 请求没有进入训练 JSONL。依据每组 case 顺序和 repetition 编号核对，所有正式行均落在三次测量位置，没有混入暖机位置。prefix 预热请求是另外的准备工作，也不等同于正式 probe 请求。

四组 fingerprint、cache config 和 `cache_config_hash=0233254c…` 与 runtime 一致：GPU 16384 slots、host 279040 slots、page64、write-through/cache、layer-first/kernel、TMA=false、sample_pool=true。runtime 记录 H200、driver 595.58.03、torch 2.14.1+cu130/CUDA compile target 13.0。配置 hash 校验不替代这些硬件记录，也不证明性能等价。各 manifest 的 pool snapshot 是该组入口时的快照，不能把它当成每个 case 的冷启动证明；逐 case prepared telemetry 仍在远端。

## 留出误差必须按阶段与并发解释

现 artifact 将完整 predictor feature cell 留出，seed42，45 个训练 cell、11 个留出 cell。总计 40 个 first-token cell 和 16 个 decode cell；不同重复、请求索引没有拆开泄漏到同一 cell 的训练/验证两边。本地按 feature cell 与实际测量值逐条关联保存的 validation，246 条记录全部唯一匹配，不重新训练模型。

| 阶段 / 并发 | 留出观测 / cell | 受支持观测 | 支持覆盖 | 全部 MAE | 受支持 MAE |
|---|---:|---:|---:|---:|---:|
| first-token / c1 | 30 / 5 | 24 | 80.0% | 19.70 ms | 16.67 ms |
| first-token / c4 | 96 / 4 | 48 | 50.0% | 1047.18 ms | 60.99 ms |
| decode / c4 | 120 / 2 | 120 | 100% | 0.210 ms/token | 0.210 ms/token |

first-token 合计 126 条，受支持 72 条；受支持 MAE 46.22 ms，等 cell 的受支持 MAE 31.44 ms。c4 受支持样本的平均相对误差仍为 337%，因为两个小输入 GPU cell 的预测明显高于实际值。decode 留出只覆盖 **n=4096、c4、输出128/512** 两个 cell；没有留出的 c1 或长输入 decode cell，不能把其 3.98% 平均相对误差推广到全部网格。原 validation 总体 `mae_ms=411.16` 混合了 first-token 的 ms 和 decode 的 ms/token，不能作为单一整体时间误差使用。

冷/GPU/host 分层也不是全覆盖：c1 冷态留出只有 n1024 一格，MAE 1.54 ms；c1 GPU 留出是 n8192/P4096，一格 MAE 21.46 ms；c1 host 小输入一格 MAE 0.23 ms，长输入两格中仅一格受支持，其 MAE 43.44 ms。c4 受支持的两格都为 n1024 的 GPU fixture，MAE 分别为 46.18 和 75.80 ms；c4 两个长 host 留出 cell 全部不受支持，诊断 MAE 为 2033.36 ms。留出没有覆盖到的状态应明确保留为空，不能从其它状态补出精度结论。

## 支持域与执行时驻留变化

三个不受支持的 first-token cell 都仍在各坐标轴包络内，但位于稀疏几何空洞。其最近归一化距离分别为：

| n / P / host-load / c | 最近距离 | 当前阈值 |
|---|---:|---:|
| 8192 / 8064 / 8064 / 1 | 0.6632 | 0.5 |
| 12288 / 6144 / 6144 / 4 | 0.5044 | 0.5 |
| 12288 / 12160 / 12160 / 4 | 0.6737 | 0.5 |

不能因为 0.5044 接近阈值就放宽支持域来改善数字：该长 host cell 的留出误差本来就很大。最终部署 artifact 在记录留出结果后使用全部 dev 观测，所以这些已采样 cell 会回到最终表中；这增加精确点覆盖，但不消除 cell 内多峰，也不是这些点取得了新的独立准确率证据。当前最大标定输入仍为 12288；配置身份匹配不授权对更长上下文宣称已标定。

所有正式 first-token 记录的实际总 prefix 与 fixture 指定值相同，但 **36/192 个 `gpu_ready` 记录实际 `host_load_tokens>0`**：n4096/P4032 共6条、n8192/P2048共12条、n12288/P3072共18条，均为 c4。它们的实际 host load 等于整个 prefix。这说明准备时的 GPU-ready 不保证执行时仍留在 GPU；初始 fixture 名称与实际执行 tier 必须分别看，不能把这些请求当作始终纯 GPU 命中。

原始 Measurement JSONL 没有 `actual_prefill_tokens` 或 `num_retractions`，不能单独从 host load、额外 DMA 或长 TTFT 推断重执行。后补 compact 提供了实际 extend 工作量；其核查见下节，native retraction 计数仍不可用。

## 同一个特征点为何出现多峰

一个直接例子是 `(n=12288,P=12160,host-load=12160,c=4)`。其 n/P/load/c 完全相同，三个重复的排序后 TTFT 都呈现相似台阶：

| 生成输出预算 | 每批四请求 TTFT 约值 | 12 条观测均值 |
|---|---|---:|
| 128 | 24 / 642 / 1271 / 1900 ms | 959.49 ms |
| 512 | 24 / 2471 / 4930 / 7389 ms | 3703.91 ms |

对应 decode TPOT 分别约 4.770 和 4.767 ms/token。等待台阶与其它请求生成工作量相符；后补阶段字段进一步把台阶定位在首次 forward 之前，但仍不足以归因到某个具体 scheduler 分支。这里使用了强制输出固定预算的 profiler，而线上真实 EOS 长度未知。

当前 first-token cell 不包含输出预算、队列位置、竞争请求剩余工作量和可用驻留容量，把这些条件下的样本合成同一个均值和误差。因此 **不能把其全部 TTFT 称为内在 Prefill 成本，或称为 host 恢复时间**。仅在 c1 与 c4 上分层不够；同为 c4 的排队/驻留形态也不同。全部观测中，长输入（8192/12288）TTFT 均值在 c1 冷/GPU/host 约为 244/70/77 ms，而 c4 为 2858/2744/2453 ms；这些是不同形状混合的描述统计，不是控制变量后的容量因果效应。

## 后补阶段字段：台阶主要发生在首次 forward 之前

`runs/remote-h200-capacity-matrix/runs/capacity-calibration/16384/stage-timings.jsonl` 含 736 个请求：552 正式、184 warmup。其 repetition 字段全部为 null，未把它当作实际计数。脚本用 group、case shape 和唯一 TTFT 与原始 measurement 逐条匹配，取得正式测量的 repetition，再核对每批 index 恰为 `0..c-1`；所有 552 条 decode 时间也与原始记录逐条对应。warmup 的序号只由 manifest 调用顺序推导，并与正式统计分开。

定义联合前置区间 `T_before = server_ttft_ms − forward_to_first_token_ms`，即 recv→记录的首次 forward；另一个字段为 forward→首 token。两者均为同一服务器时钟的 wall 区间。前者可含等待、admission、host 恢复和准备，后者也不是纯 GPU kernel 时间。本批所有差值非负，且 server elapsed 与 TTFT+decode 相等。

| 正式测量 | 请求数 | TTFT 均值 | 联合前置均值 | forward→首 token 均值 | 前置占累计 TTFT |
|---|---:|---:|---:|---:|---:|
| c1 | 120 | 67.92 ms | 1.15 ms | 66.77 ms | 1.70% |
| c4 | 432 | 1313.61 ms | 1219.44 ms | 94.17 ms | 92.83% |

这是各自网格的描述统计，不能把 c1/c4 的混合形状差当作只改变并发的单因素效应。c4 前置区间中位数仅 2.00 ms、P95 为 7265.13 ms；其 forward→首 token P95 为 300.18 ms，说明长等待集中于部分请求。

对上面完全相同的 12K/high-host-prefix 特征点，阶段均值为：

| 输出预算 | TTFT | 联合前置区间 | forward→首 token |
|---|---:|---:|---:|
| 128 | 959.49 ms | 936.93 ms | 22.56 ms |
| 512 | 3703.91 ms | 3681.99 ms | 21.92 ms |

输出预算增大带来的额外 TTFT 几乎全部落在联合前置区间，forward 区间并未相应增大。每批按实际 TTFT 升序后，前置均值台阶在 output128 时约为 `2 / 620 / 1249 / 1878 ms`，output512 时约为 `2 / 2450 / 4909 / 7368 ms`，forward 区间均保持约 22 ms。这里的排序位次仅用于离线观察，不能作为提交前已知的 queue-position feature；原 request index 也不恒等于这个完成次序。

实际 Prefill 工作量方面，**552/552 正式请求和 184/184 warmup 的 `actual_prefill_tokens` 均恰好等于 `prompt − device_hit − host_loaded`，0 条超过 prompt**。该计数由成功 forward 的 extend lengths 累加，包含分块和发生的重执行工作；所以本批 receipt 没有观察到额外的 Prefill extend 重算。不能进一步称为所有 scheduler retraction/抢占次数为零，因为 compact 没有该计数和完整 scheduler 时间线。此前 GPU-ready→host 的 36 次变化也不应据此叫作 Prefill 重执行。

## 可执行的 B 改进机制

1. **联合前置区间与 forward 区间分别建模。** 上述只读分解已定位同 cell 的主要多峰，不需要立即重跑大网格。后续可以用同一批真实分量建立新的对照 artifact，同时保留原总 TTFT lookup；等待/准备项不能混入内在 Prefill 项，forward wall 也不能冒充 CUDA kernel 时间。缺少原生更细边界时仍保留“联合前置”命名，不人为把两段分别改名纯 queue 和纯 restore。
2. **给调度等待单独的、提交前可观测特征。** 复用当前 active/queued counts、free/evictable GPU slots、真实 GPU/host prefix；必要时扩展少量 scheduler 聚合量，例如排队待处理 input tokens、活动请求已知剩余预算。预算/历史长度 prior 可以预测竞争工作量，当前真实未来输出长度只能作离线标签，不能作 feature。两候选导致的驻留需求不同，等待也不能一律作为相同常数抵消。
3. **分别验证执行成本与等待项，再验证整体决策。** 保留固定 lookup 为对照；对拆分项使用同一硬件身份、同一训练/留出 cell 规则，在新 artifact 中比较支持覆盖、分位误差及误切换 regret。并发同批的四个请求应作为相关观测看待，不能用 request 数制造独立样本充分性；针对压力分层标注不确定性，缺层回退，而不是通过放宽距离阈值获得“校准”。

这些方向针对已观察到的缺失机制，不是因为负面结果停止 B，也不是在本轮矩阵过程中调整门槛。暂不修改冻结成本表，让实际六方法结果与这一诊断保持可追溯。

## 与 121920 旧标定的限定比较

旧表 first-token 留出 132 条、支持 78 条，支持 MAE 62.91 ms；新表为 126/72、46.22 ms。不能据此说小池预测更准：网格、留出 cell、状态分布不同，新表两个 c4 长 host 难点被判为不支持，受支持子集反而排除了它们。旧 decode 留出也只有两个 cell，其 MAE 0.292 ms/token，与新表 0.210 的差异不能独立归因容量。

记录显示两次 GPU UUID、driver/CUDA/torch 相同，host/layout/backend 等配置相同；但除了 GPU pool，旧/新 `skip_server_warmup` 为 false/true、`enable_cache_report` 为 false/true，采样分组与请求顺序也不同。它们是不同运行条件下的开发观测，不是只改变一个容量变量的随机化实验。

## 复核产物

```bash
PYTHONPATH=. python runs/capacity-calibration-analysis/analyze_16k.py
```

脚本和 `runs/capacity-calibration-analysis/16384.json` 被保存在 ignored runs 目录中。输出含四组绑定/数量检查、完整留出分层与 geometry、raw 阶段分布、GPU→host 变化计数、compact 的唯一对应与阶段统计、实际 extend 工作量及上述台阶例子。脚本只读取冻结文件，断言均通过，Ruff 通过；没有改变 artifact、线上策略或实验结果。
