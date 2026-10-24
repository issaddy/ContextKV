# B6 phase1 的真实计时分解

2026-10-24。**下一步应优先补足原生 admission/恢复就绪的时间归因，而不是先扩大 forward 拟合。** 九个控制器格的写策略差异主要落在首次 forward 之前；现有时间戳不能把这段分成纯排队、容量准入与缓存准备。本结论只定位观测与模型缺口，不指定策略赢家，也不证明某写策略或 DMA 造成了这些差异。

数据来自 `runs/remote-h200-b6-phase1/runs/b6-joint-dev-phase1-v1`：Rebuild、Fixed4096、Redundancy40 × write_through/ selective/write_back，P2、c4、seed42、单次运行，每格5会话/41轮。实际 GPU/host 池为16,384/279,040 tokens，page64；各方法有自己的生成历史。没有读取质量评分、qrels 或 test。复算入口和6项CPU合同测试为 `runs/b6-timing-decomposition/{analyze.py,test_analyze.py}`；完整逐轮、分层和配对结果保存于同目录 `analysis.json`，原始结果未改。

## 绑定与计时含义

全部 **369/369** 成功轮按 `result.attempt_id` 绑定 benchmark 内 `telemetry.after.events.request_terminal`。核对 rid、完整 receipt、restart、audit 实际 input IDs 的既有 token hash、output IDs、长度与原生时间来源，均一致；九格各41个终态，无额外终态混入，无事件截断。各轮 D+H+实际Prefill 恰好覆盖输入，无重复执行工作增量；8个截断答案仍留在全部统计中。`servers/*/metrics-after.json` 不用于确定主分母，因该类快照可能包含运行前探针。

计时定义来自固定项目 `0893134` 的 `engine_patches/contextkv_runtime.py:native_timing`，该文件 SHA256 为 `71ddcbf9b3e11368d465860a11caa7476bf619c5e86633828759a2e20b8bd97f`：

- **pre** = server TTFT − forward-to-first：scheduler 收到请求到首次 forward；包含其中的准入等待和缓存准备，不能称纯 queue。
- **forward** = 首次 forward 入口到首 token 完成；不是纯 GPU kernel，部分缓存准备已经在入口前发生。
- **decode** = 首 token 完成到最终 token 完成。369轮全部满足 `pre + forward + decode = server_elapsed`，均非负；服务端 TPOT 与 `decode/(N−1)` 一致。
- **controller complete** 为控制器入口到答案提交；**E2E** 与既有 report 一致，从 runner 的 `t_arrive` 到提交。**外部残差** = controller complete − server elapsed，已含 control_ms，不能再加一次控制耗时。客户端时间有序，服务端 elapsed 不超过客户端 HTTP 区间时长；未相减跨进程绝对时钟。

全部369条的描述性均值：pre **1,381.6ms**、forward **94.6ms**、decode **2,673.2ms**，分别占server elapsed的33.30%、2.28%、64.42%。pre占server TTFT的 **93.59%**。controller complete均值4,275.3ms，E2E 4,275.6ms，control 99.6ms，外部残差125.9ms。合并P95为pre4,056.1ms、forward177.9ms、decode4,805.3ms；它们来自不同轮，不能相加得到总P95。369条是同5会话跨9配置的重复观察，不是369个独立会话。

## 九格全量分解

以下各段及E2E均为**等会话均值，单位ms**；H>0/截断是41轮全量计数。尾值为全41轮P95。WT=write_through，SEL=write_through_selective，WB=write_back。每格P99、各会话与首轮/后续完整分布均保存在JSON；41点的P99不构成尾部保证。

| 格 | pre | forward | decode | E2E | pre P95 | E2E P95 | H>0 | 截断 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Rebuild WT | 1790.1 | 99.5 | 2643.5 | 4657.9 | 4484.5 | 7248.5 | 14 | 1 |
| Rebuild SEL | 1027.6 | 96.6 | 2701.4 | 3954.2 | 3275.2 | 6667.5 | 1 | 1 |
| Rebuild WB | 883.1 | 89.3 | 2715.5 | 3802.2 | 3103.5 | 6597.2 | 15 | 1 |
| Fixed4096 WT | 1699.0 | 104.8 | 2650.9 | 4585.5 | 4461.9 | 7048.6 | 12 | 0 |
| Fixed4096 SEL | 978.7 | 95.5 | 2807.6 | 4006.2 | 3202.8 | 6657.0 | 2 | 0 |
| Fixed4096 WB | 1010.8 | 88.2 | 2703.0 | 3916.9 | 2832.0 | 6195.4 | 18 | 2 |
| Redundancy40 WT | 2201.1 | 92.2 | 2596.7 | 5017.0 | 4616.0 | 8615.2 | 15 | 1 |
| Redundancy40 SEL | 1465.1 | 103.7 | 2733.9 | 4449.6 | 3215.9 | 6526.0 | 2 | 1 |
| Redundancy40 WB | 1380.3 | 82.5 | 2783.2 | 4372.7 | 3424.6 | 6624.3 | 18 | 1 |

各格首轮5条的pre均值321–893ms，后续36条938–2409ms；forward首轮110–205ms，后续74–103ms。后续输入/历史、并发状态和答案长度都变化，不能把差异仅归于轮次。全局最长pre来自Redundancy40 WT的d5 turn9：6,461.2ms，forward66.8ms、H=448、input=5077、output=406；另一个WT长尾c6 turn8为5,308.0ms pre、78.5ms forward。原始attempt/receipt与前三大pre记录都在JSON。

## 配对比较与相同IDs子集

下面每行是后者减前者的等会话均值；全部41轮保留。input相同和input+output相同按实际IDs逐元素比较，不只比较文本或hash。子集按其实际参与会话等权，分母见括号；**这些仍是观察性子集，arrival、缓存驻留和batch状态未控制**。

| 写策略比较 | 全量 Δpre / Δforward / Δdecode / ΔE2E ms | 同input ΔE2E（轮/会话） | 同input+output ΔE2E（轮/会话） |
|---|---|---|---|
| Rebuild SEL−WT | −762.5 / −3.0 / +57.9 / −703.8 | −116.4（10/5） | −609.1（5/3） |
| Rebuild WB−WT | −907.0 / −10.3 / +72.0 / −855.7 | −282.3（14/5） | −585.8（9/4） |
| Fixed4096 SEL−WT | −720.3 / −9.3 / +156.6 / −579.3 | −136.1（10/5） | −390.7（5/4） |
| Fixed4096 WB−WT | −688.2 / −16.6 / +52.0 / −668.6 | −167.7（10/5） | −314.8（5/4） |
| Redundancy40 SEL−WT | −736.0 / +11.5 / +137.2 / −567.4 | −63.0（10/5） | −555.4（5/3） |
| Redundancy40 WB−WT | −820.8 / −9.7 / +186.5 / −644.2 | −662.1（11/5） | −671.8（6/4） |

六个全量比较的forward变化仅−16.6至+11.5ms，pre下降688–907ms，同时decode增加52–186ms。相同IDs子集没有消除运行状态混杂，也不能据其时延比较声称正式质量相同。三家族WB−WT的“两边H=0”子集仍有pre下降342.5/222.2/465.2ms（23/22/22轮），所以不能把前置差异全部记到该请求的H2D。两边H>0的样本也不同：Fixed SEL−WT仅2轮，ΔE2E反而为+854.2ms；小样本反例完整保留。

| 家族比较 | 全量 Δpre / Δforward / Δdecode / ΔE2E ms | 同input ΔE2E（轮/会话） | 同input+output ΔE2E（轮/会话） |
|---|---|---|---|
| Fixed−Rebuild，WT | −91.1 / +5.3 / +7.4 / −72.5 | +33.1（14/5） | +112.0（11/3） |
| Red−Rebuild，WT | +411.0 / −7.3 / −46.8 / +359.1 | +76.3（8/5） | +50.8（4/3） |
| Fixed−Rebuild，SEL | −49.0 / −1.0 / +106.2 / +52.0 | +44.5（10/5） | +319.4（6/3） |
| Red−Rebuild，SEL | +437.5 / +7.1 / +32.4 / +495.4 | −88.3（8/5） | −75.1（4/3） |
| Fixed−Rebuild，WB | +127.7 / −1.1 / −12.5 / +114.6 | +72.6（9/5） | +324.6（5/3） |
| Red−Rebuild，WB | +497.2 / −6.7 / +67.7 / +570.5 | −125.7（8/5） | −210.6（4/3） |

家族全量与匹配子集有方向反转，进一步说明不能用小型相同答案子集代替完整P2路径，更不能由本报告直接冻结B6。所有12组比较还按两边H=0、两边H>0、仅左/仅右H>0交叉保存，空集不补零。

## 传输与下一观测边界

completed DMA只在整格关联，单位MiB。九格共634个独立transfer_id，全部有CUDA事件时长；没有逐请求传输归因。CUDA时长求和保存在JSON，**不添加到pre/forward或E2E**。

| 家族 | WT D2H / H2D | SEL D2H / H2D | WB D2H / H2D |
|---|---:|---:|---:|
| Rebuild | 8669.5 / 609.0 | 535.5 / 45.5 | 8004.5 / 570.5 |
| Fixed4096 | 8526.0 / 444.5 | 910.0 / 45.5 | 7892.5 / 724.5 |
| Redundancy40 | 7301.0 / 1823.5 | 1081.5 / 364.0 | 6755.0 / 2289.0 |

SEL传输量明显更少，但WB的D2H仍接近WT且pre也下降；仅看总字节无法解释等待。H>0常与更长pre共现，也可能同时标记容量竞争、上下文长度或先前轨迹差异，不能判定恢复操作本身耗掉了这些秒数。

最小下一步是在既有Req/终态观测上记录**实际尝试准入的门控、真正加入can_run_list的接受时刻、相关预算以及恢复未就绪区间**，保留同attempt/rid与服务端时钟。返回NO_TOKEN/OTHER不必然表示当前请求被拒绝，须区分“成功加入后不能再接下一条”的情况。随后才区分容量阻塞、传输就绪和调度顺序，并评估是否要改等待/联合决策模型。现forward曲线仍有价值，但本批证据不支持把更精细forward拟合列为解释写策略差异的首要动作；decode仍占完整服务成本约64%，输出长度和生成质量也不能被忽略。
