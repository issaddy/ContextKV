# 两 peers 下的 APPEND / REBUILD 原生准入实测

完整 v4 在受控 16K GPU KV 池中完成 **72/72 episodes、264/264 模型调用**。27 个正式 A/R pair 都出现非零 peer 完成时间差；但 **27/27 的候选自身与三请求总 flow-time 排序相同**。因此本次证实“候选动作不影响其他请求”的零影响假设在这些 fixture 上不成立，尚未证明加入 peer 项会改变决策、降低 planner regret 或改善问答质量。

最显著的结果是 L case 的 APPEND 在两 peers 完成前未能进入首次 forward；其自身多耗约 2.35–2.51 秒，两个 peers 合计少耗约 0.32–0.37 秒，总计仍较 REBUILD 多约 2.03–2.15 秒。GPU-ready 与 host-only 相近不表示 GPU 前缀被逐出：实际 receipt 分别保留 D=7808/H=0 和 D=0/H=7808，首轮原生容量检查呈现相同的 1999-token 缺口。

## 原件、来源和审计范围

完整公共导出位于 [remote-h200-peer-interference](../runs/remote-h200-peer-interference/)，39 文件归档 SHA256 为 `9774f5c0f77663c55fbb1d6232fc8f4e1beb2b552fdc516d3ef38abda628664b`。本报告使用完整 v4，没有拼接运行中的部分结果；原始答案、失败、quote、receipt、terminal 时钟和 gate 快照均保留。

- [独立审计 summary](../runs/peer-interference-review/h200-v4/summary.json)：`audited_complete`、issues=[]，本地对固定 Git `4e689eb173fc72d92570a6b2e5d473f952c84c5c` 的 321 个 tracked blobs 与导出的部署声明逐项核对，另核 5 个实际组件 hash。它验证已导出的来源证据，不是本地实时远端 attestation。
- [原 ledger](../runs/remote-h200-peer-interference/runs/peer-interference-v4/episodes.jsonl)、[逐 pair 结果](../runs/peer-interference-review/h200-v4/pairs.jsonl)、[机制统计](../runs/peer-interference-review/h200-v4/mechanism-summary.json)可复算下表。脚本为 [analyze.py](../runs/peer-interference-review/analyze.py) 与 [mechanisms.py](../runs/peer-interference-review/mechanisms.py)。
- SGLang 固定 `3f36fc08331f64cd63c4fbbdf39cc402f38bb38b`；实际 patch SHA256 为 `730ead04291bb23e5ab09ee6acb15df9225c0d1942a14dd3516f20c5a103c474`。[远端 sourcecheck](../runs/remote-h200-peer-interference/runs/admission-sourcecheck-h200.json)保留原文件绑定、14 文件 patch/apply 检查。v4 ignored runner SHA256 为 `cd67751563f5e53e6498daef91ce3433d19b1ae1e9762a4158b83d2176a5a8db`。
- [supervisor](../runs/remote-h200-peer-interference/runs/peer-interference-v4-supervisor.json)记录 fixture completed / returncode 0 / restore 0。原 manifest 的 `status=prepared` 是启动时元数据，完成性由实际 summary、ledger 和 supervisor 共同确认，未改写它。远端 tar 部署没有 Git HEAD，不将这个可选字段缺失误当部署失败。
- [ledger-file-check](../runs/remote-h200-peer-interference/runs/peer-interference-v4/ledger-file-check.json)记录远端逐一比较 72 个 episode 副本与 ledger 相等；本次未导出这些重复个体 JSON，本地没有声称重新读取它们。私有 startup 日志未导出；运行配置依据公开 runtime、sourcecheck 与 supervisor。

## 冻结设计与完整分母

同一 GPU0 H200、Qwen2.5-7B-Instruct `a09a35458c702b33eeacc393d103063234e8bc28`、权重/KV BF16，实际 GPU KV **16384** / host **279040** tokens，page64、chunk8192、FCFS、max-running8、FA3、context32768、seed42。写策略为 write-through，layer-first/kernel，TMA 关闭；host admission 关闭，原生 admission diagnostics 与 profiling 开启。配置与实际池信息保存在 [runtime-before](../runs/remote-h200-peer-interference/runs/peer-interference-v4/runtime-before.json)。诊断本身有固定观测成本，两臂均开启，未测关闭时的开销差。

从已冻结的 15 个开发案例按 APPEND 长度选择最短、中位、最长三项，不依据本次性能或答案选择：

| 简称 / case ID | APPEND 输入 tokens | REBUILD 输入 tokens | 旧 execution seed 输入 tokens |
|---|---:|---:|---:|
| S / `4a536f36e92e4fe7bece637967e3178c` | 4977 | 3615 | 3741 |
| M / `679bad8291324fe3a3401561e1443b3e` | 6693 | 3842 | 6277 |
| L / `2744b5f4572d4b42a1865647afbf31f6` | 11217 | 5397 | 8024 |

每个 case × cold / gpu_ready / host_only 有 1 个配对 warmup 和 3 个正式配对重复；两臂均使用各自独立的 cold 初始化，warm 状态额外执行同一旧 execution seed。共 **18 warmup episodes + 54 formal episodes = 72**，每 episode 有两个固定 peers 和一个候选，合计 **216 target + 48 seed = 264**。全部 target 实际输出 512 tokens、`ignore_eos=true`，seed 输出 1 token；这是固定工作量机制实验，不能称自然 EOS、无截断问答或质量实验。264 个 retire 均完成，72 个 episode 均确认 drained，未用 ACK 推断缓存立即释放。

冻结 plan SHA256 为 `2285ed761357fd9448255254d7fd71158164e9e279c685dc8d12794b3a2e3f49`。审计核了实际输入/输出 ID、receipt/input hash、原快照、请求身份、三状态初始 shape、输出预算、唯一 terminal 及原生时间顺序。A/R 在同一 snapshot 的一个 quote batch 中读取；两个 peers 各自独立 quote。三次 idle 观测均保存，但不将其说成跨三组的原子快照。

候选固定在客户端起点后 100 ms 发出，不根据 peer 首 token 自适应。正式轮实际客户端 deadline lateness 为 0.049–1.074 ms；原生 candidate receive 相对两 peers 的到达差为 107.231–123.026 ms，均满足先有 peer 且至少一个未完成。**相同初态 shape 不保证候选实际到达时整个调度/缓存状态相同**；实际时钟与 gate 为后续分析依据。warmup 单独保留，未并入以下正式均值。

## 9 个分层与 27 个原始差值

以下均为 **APPEND − REBUILD，单位 ms**；负数表示 APPEND 更小。own 为候选 `completion_time − scheduler_recv_time`；peer 为两个 peers 各自该区间之和；total 为三请求 flow-time 之和，**不是 makespan**。client 为三次客户端发出至完整响应时间之和，使用客户端自己的时钟；不跨 client/server 时钟相减。总差严格等于 own + peer（显示值有舍入）。每行均值有 3/3 正式 pair，无缺失补零。

| case | 初态 | own 均值 | peer 合计均值 | total server 均值 | client 合计均值 |
|---|---|---:|---:|---:|---:|
| S | cold | +41.42 | +80.80 | +122.22 | +121.81 |
| S | gpu_ready | -40.92 | -82.83 | -123.75 | -123.43 |
| S | host_only | -39.36 | -78.02 | -117.38 | -116.50 |
| M | cold | +91.58 | +177.60 | +269.17 | +266.84 |
| M | gpu_ready | -54.97 | -109.49 | -164.46 | -163.40 |
| M | host_only | -53.33 | -107.67 | -161.01 | -160.37 |
| L | cold | +2514.16 | -368.04 | +2146.12 | +2139.36 |
| L | gpu_ready | +2354.82 | -323.67 | +2031.15 | +2031.71 |
| L | host_only | +2358.50 | -322.85 | +2035.65 | +2037.63 |

| case | 初态 | repeat | own | peer 合计 | total server | client 合计 |
|---|---|---:|---:|---:|---:|---:|
| S | cold | 0 | +40.84 | +82.38 | +123.22 | +125.31 |
| S | cold | 1 | +42.08 | +82.81 | +124.89 | +122.92 |
| S | cold | 2 | +41.34 | +77.20 | +118.55 | +117.19 |
| S | gpu_ready | 0 | -40.44 | -79.86 | -120.30 | -118.94 |
| S | gpu_ready | 1 | -40.77 | -80.89 | -121.66 | -120.24 |
| S | gpu_ready | 2 | -41.54 | -87.75 | -129.29 | -131.09 |
| S | host_only | 0 | -39.93 | -79.91 | -119.84 | -118.87 |
| S | host_only | 1 | -40.04 | -77.44 | -117.48 | -115.63 |
| S | host_only | 2 | -38.12 | -76.71 | -114.83 | -115.01 |
| M | cold | 0 | +90.83 | +175.01 | +265.84 | +262.74 |
| M | cold | 1 | +92.62 | +180.22 | +272.84 | +270.85 |
| M | cold | 2 | +91.27 | +177.55 | +268.83 | +266.93 |
| M | gpu_ready | 0 | -55.35 | -114.32 | -169.67 | -170.67 |
| M | gpu_ready | 1 | -55.83 | -109.20 | -165.03 | -162.68 |
| M | gpu_ready | 2 | -53.72 | -104.96 | -158.69 | -156.84 |
| M | host_only | 0 | -54.48 | -104.65 | -159.13 | -155.90 |
| M | host_only | 1 | -53.49 | -111.92 | -165.41 | -166.63 |
| M | host_only | 2 | -52.03 | -106.45 | -158.48 | -158.58 |
| L | cold | 0 | +2513.00 | -368.55 | +2144.45 | +2135.22 |
| L | cold | 1 | +2514.60 | -368.34 | +2146.26 | +2136.55 |
| L | cold | 2 | +2514.88 | -367.22 | +2147.66 | +2146.32 |
| L | gpu_ready | 0 | +2353.55 | -320.85 | +2032.71 | +2033.44 |
| L | gpu_ready | 1 | +2354.42 | -328.19 | +2026.22 | +2026.55 |
| L | gpu_ready | 2 | +2356.48 | -321.97 | +2034.51 | +2035.14 |
| L | host_only | 0 | +2354.52 | -319.60 | +2034.91 | +2039.16 |
| L | host_only | 1 | +2359.02 | -333.08 | +2025.94 | +2024.60 |
| L | host_only | 2 | +2361.95 | -315.86 | +2046.09 | +2049.13 |

peer 差 6 个为正、21 个为负，没有零；own 与 total 没有一次排序反转。S/M 在 cold 下 APPEND 增加自身及 peer 时间，warm 下则两者减少。L 的 peer 差与 own 方向相反，但幅度不足以抵消自身准入等待。三次重复方向稳定只适用于这三个固定案例；不将 27 个 pair 当 27 个独立会话，不计算这样的会话 bootstrap 或宣称总体显著性。

两臂的 peer 输入相同、输出长度都固定，但 peer0 的 output IDs 在 27/27 pair 不同，peer1 仅 18/27 相同。保留这项实际轨迹差异，不能称“所有生成逐 token 相同”的纯时间复演，也没有据此判断答案好坏。

## 原生准入与状态证据

S/M 各状态的候选均一次 `accepted:CONTINUE`。S 的 warm APPEND 实际复用 3712 tokens，prefill 从 cold 4977 降到 1265；M 复用 6272，prefill 从 6693 降到 421。与此对应，S 的 forward-entry 至 prefill-finished 均值由 cold 117.44 ms 降到 GPU 36.44 / host 37.46 ms，M 从 164.82 降到 19.21 / 21.72 ms；warm REBUILD 没有该命中。这支持复用工作量改变了 forward 区间，尚不能把 peer 全部差值精确归因到某一个 kernel。

L 的原始 receipt 在同分层的三个重复完全一致，D/H/U 分别为实际 device-hit / host-loaded / prefill tokens：

| L 初态 / arm | 实际 D / H / U | 首次容量检查 requested / remaining | 准入前联合区间均值 ms | forward-entry 至 prefill-finished 均值 ms |
|---|---|---|---:|---:|
| cold A | 0 / 0 / 11217 | 11793 / 9794 | 2435.74 | 273.18 |
| cold R | 0 / 0 / 5397 | 5973 / 9794 | 0.92 | 127.48 |
| gpu_ready A | 7808 / 0 / 3409 | 3985 / 1986 | 2435.28 | 91.91 |
| gpu_ready R | 1216 / 0 / 4181 | 4757 / 8578 | 1.13 | 105.06 |
| host_only A | 0 / 7808 / 3409 | 11793 / 9794 | 2435.87 | 95.69 |
| host_only R | 0 / 1216 / 4181 | 5973 / 9794 | 1.34 | 104.57 |

GPU-ready 的 A 初始 quote 已有 7808 device tokens，实际仍为 D7808/H0；host-only 则确实加载 H7808。**没有证据把前者解释为被 peers 逐出后又从 host 恢复**。首 gate 中 GPU A 的剩余预算 1986，比 host A 的 9794 少 7808，requested 也恰少 7808，两个容量缺口都为 1999。这与原生 prefix 临时 pin 影响可用预算、host 恢复需要 GPU 空间的语义一致；它解释了为什么缓存层不同仍可能同时卡在容量检查。这里引用的是实际 gate 预算，不能拿未 pin 的空闲池直接替代。[诊断字段语义](native_admission_diagnostic.md)

另一项[同请求 gate 前后复核](../runs/joint-cost-design/native-pin/audit.json)进一步核对 54 个正式 candidate：45 个首次接受、9 个首次拒绝，拒绝全为 L A。在同一次拒绝中，pin 内容量 gate 至 unlock 后 `adder_not_accepted` 的剩余预算增加量为 GPU 7808、cold/host 0，offset 与 generated 不变。这支持上述 pin 解释；其范围只到这 9 个拒绝样本，不证明所有共享前缀下的 pin 估计都精确。

每个 L cold A 有 4 次 `not_accepted:NO_TOKEN`，最终 `accepted:OTHER`；GPU/host A 各 3 次未接受，最终 `accepted:CONTINUE`。cold 的 `OTHER` 是已加入后 chunk8192 预算耗尽，不是此次请求失败；后续 chunk 不再经过同一 add_one_req outcome。L R 均立即 `accepted:CONTINUE`。每个 L A 还有 510 次 `scheduler_batch_full` 队首观测，**这个计数不是 510 次完整准入，也不是 2435 ms 的时长归因**。

原生时钟独立显示：L A 在两个 peers 都完成后才进入首次 forward（repeat0 各状态约迟 2.0–2.8 ms）；L R 则在 peers 完成前约 2.59–2.62 s 已进入 forward。这是比“到达时 peer 未完成”更具体的重叠差异。A 的前置联合区间包含原生准备/等待/可能的恢复，现有观测不足以再精确拆分纯排队、CPU 和 DMA。peer 在 A 侧更早完成与其未同 A forward 重叠相符，但本次没有 batch-step kernel 轨迹来单独识别各项因果贡献。

## 失败运行独立保留

| run | 实际状态 | 生成分母与解释 |
|---|---|---|
| v1 | 入口 Git HEAD 元数据失败；restore 0 | 0 episode、0 generation；完整 tar 来源已核验，非必须的 `git rev-parse HEAD` 导致入口失败，后改为可选观测 |
| v2 | 启动 bind `Address already in use`；restore 0 | 0 generation、无新 server PID；探针后采用 SO_REUSEADDR 并等待监听口可用，不能归因模型或新调度算法 |
| v3 | 72 failed，全部 `MIXED_SNAPSHOT`；restore 0 | 48 seed 成功、0 target；不能称 264 调用成功。原 driver `42016c0d…` 将 snapshot 2/7 的 A/R 与 snapshot 0 的 peer 混批，原生正确拒绝 |
| v4 | 72 completed；fixture/restore 均 0 | 216 target + 48 seed 全部成功；拆为 2/1/1 quote，不改快照、不放松原生协议 |

[v3 独立审计](../runs/peer-interference-review/h200-v3-failed/summary.json)保存 72 个失败和 27 个正式 `not_comparable` pair，审计 issues=[] 表示失败记录与原件一致，**不表示实验成功**；其旧实际 runner 保存在原 export，未用修复后的 v4 源冒充。v4 supervisor 在结束后以强制信号清理 owned server（server returncode −9），与所有请求已完成、fixture 0 / restore 0 分列，不称服务自身正常退出。

## 对下一机制的约束

这批观测支持继续研究 joint objective 的 peer 项，却不支持直接增加一个固定 peer penalty：局部影响可以为正也可以为负，且 27 个 pair 都没有改变自身成本所给的排序。本 fixture 没有运行冻结 cost artifact 的选择预测，也没有计算其误差/regret，不能把观察到的大等待直接称为某一已运行 planner 的预测失败。

优先可证伪路径是用同一冻结输入区分 **原生容量准入、候选到达时状态、已有请求剩余服务**：先在离线记录中核对实际 first-gate 请求量/可用量及首次 forward 是否必须等 peers 释放，再固定参数预测未见分层的准入与等待。若模型能解释 L 的阶跃但仍无法解释 S/M 的 peer 差，再用有界 batch-step 观测检验增量服务成本；将收益、符号和排序反转都纳入验收。新机制仍需自然 EOS、多会话独立重复和质量协议验证，本报告既不宣布 planner 改进，也不据这些固定长度结果选择 B6。
