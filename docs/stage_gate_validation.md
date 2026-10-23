# 16K 等待模型的稳定期消融

在同一优化源码、阶段成本与误差 margin 下，将 `min_stable_turns` 从 2 改为 0，两次开发运行的等会话完成均值均下降：合并 **6516.37 → 5220.86 ms，下降 19.88%**。Gate0 仍比 Fixed_4096 慢 **15.16%**、比始终 REBUILD 慢 **10.74%**。结果支持继续检验较短稳定期，同时表明等待/成本预测仍需改进；没有建立质量相等、独立测试胜出或逐动作因果收益。

## 数据、配置与独立复核

原始数据为 `runs/remote-h200-gate/runs/controller-stage-gate-16384-v2`，主分析和独立复核分别运行：

```bash
.venv/bin/python runs/stage-gate-review/analyze.py
.venv/bin/python runs/stage-gate-review/independent_check.py
```

输出 `summary.json` 和 `independent-check.json`；后者不导入主分析器，直接读取 benchmark、manifest、events、audit 和 cold hook。两者主要均值及门槛归因一致。**8 组各 41 轮，328 success、0 failure、0 unfinished、0 timed-out session**；没有缺组或删掉失败组。每组有 41 个 native terminal 标签，成功轮 audit 完整，遥测未截断。八份 cold hook 均返回成功，GPU/host 唯一占用归零；池容量分别为 16384/279040，page=64。

Cloud dev 的 5 个会话，Q0、P2、c=4、seed42、输出上限 1024，保持同一模型指纹、数据/split、官方固定 top5 检索、context 配置、host/layout/write policy 及 cost artifact。引擎为固定 SGLang `3f36fc08331f64cd63c4fbbdf39cc402f38bb38b` 加现有研究补丁；未在这轮新增 admission 行为或细化观测接口。控制器冻结为 `ee480e41b37ab5f5bf559ca8cc4db51d326eb56a`，含 decode LRU 优化。按 manifest 规则重算的 controller 源码 SHA 为 `33e0c890d52813556112420e608e6e35e907b4586a782db1da9bd81c72cc5d29`，八份全部匹配。

Gate2/Gate0 的完整配置除各自 `state_dir` 和 `planner.min_stable_turns` 外一致：`delta_abs=0`、`delta_rel=.05`、`z=1`、fallback threshold=8192、online calibration=false，使用同一个 16K stage artifact。没有降低误差 margin 或把解析等待标成 calibrated。固定策略也使用相同优化实现和输出预算。

每次重复新启一个独占研究引擎，方法间仅 KV reset；r1 顺序为 Gate2→Gate0→Fixed_4096→Rebuild，r2 为 Gate0→Fixed_4096→Rebuild→Gate2。两次均显式 RAYON=4。八个 run ID 不同，`owned-server.json` 记录两个不同 PID。Cold reset 会重建协议 owner，因此八个 `engine_restart_id` 也不同；这不表示每方法都重启进程。该轮换不是完整随机化或全排列平衡，方法间 JIT/内核状态仍共享，不能称原生 B6 初态匹配。

## 时延与控制开销

等会话均值先在每次重复的每会话内平均 `complete_ms`，再对五会话等权、两重复等权。`complete_ms` 从 controller 的服务入口到提交完成；`turn_wall_ms` 还包含入口外适配、audit 等工作。表中的 TTFT 使用 `arrival_to_first_ms`，从 benchmark 到达到客户端首 chunk；保存的 `ttft_ms` 则从 controller 入口起算，两者不混用。P95 合并各方法的 82 轮，仅作描述统计。

| 方法 | r1 等会话完成均值 ms | r2 等会话完成均值 ms | 合并等会话均值 ms | 完成 P95 ms | 到达至首 chunk 均值 ms |
|---|---:|---:|---:|---:|---:|
| Gate2 | 6526.00 | 6506.73 | 6516.37 | 12110.42 | 3622.01 |
| Gate0 | 5246.68 | 5195.05 | 5220.86 | 9621.64 | 2346.00 |
| Fixed_4096 | 4305.07 | 4762.13 | 4533.60 | 7823.32 | 1876.82 |
| Rebuild | 4585.85 | 4843.28 | 4714.57 | 8214.66 | 1948.68 |

Gate0 相对 Gate2 的等会话均值在 r1/r2 分别下降 19.60%/20.16%。两重复合并后，五个会话都比 Gate2 快，但也都比两种固定策略慢；逐会话数值保留在 JSON，不能把所有相关轮次当作独立样本做显著性推断。

| 方法 | control 均值 / P95 ms | 提交前均值 ms | 响应后提交均值 ms |
|---|---:|---:|---:|
| Gate2 | 151.29 / 236.74 | 125.65 | 25.64 |
| Gate0 | 145.40 / 269.59 | 122.14 | 23.26 |
| Fixed_4096 | 109.66 / 210.03 | 86.72 | 22.94 |
| Rebuild | 106.19 / 209.58 | 83.64 | 22.55 |

Control 包括候选构建、报价/prepare 等 HTTP 等待、规划和持久化，不能叫纯 CPU 时间。Gate0 的 control 均值只比 Gate2 少 5.89 ms，P95 还更高；本轮总时延差不能归因于控制开销下降。所有方法均已使用 decode 缓存，不能把本次 Gate0/Gate2 对照当作该 CPU 优化的消融，也不能把与旧 `bc6c803` 运行的差异直接归因于缓存。

## 动作、实际工作与门槛

| 方法，每方法 82 轮 | APPEND / REBUILD | 输入 tokens | 输出 tokens | Prefill tokens | host 加载 tokens | 截断轮数 |
|---|---:|---:|---:|---:|---:|---:|
| Gate2 | 52 / 30 | 536296 | 48698 | 215272 | 221248 | 6 |
| Gate0 | 38 / 44 | 464802 | 46885 | 225698 | 147456 | 4 |
| Fixed_4096 | 2 / 80 | 383907 | 43135 | 270115 | 19136 | 4 |
| Rebuild | 0 / 82 | 385023 | 44765 | 277567 | 18816 | 1 |

Gate0 增加了重建，输入和 host 加载减少，Prefill 增加；固定方法进一步减少输入和 host 加载，但 Prefill 更多。省下 Prefill token 不会自动转成完整请求时延收益。输出长度和截断也随闭环变化，不能从这些计数推断回答质量相同。

每方法 10 个初始轮只有一个候选，剩余 72 次决策各有两臂。Gate2/Gate0 分别有 130/144、128/144 臂 eligible；两臂同时 eligible 均为 58/72。全部候选 calibrated=false，保留解析等待未经验校准的事实。两者各 14 次 fallback，均记录有阶段计时域不足；Gate0 有 2 次 fallback 最终选中的候选也不 eligible，这不等于执行失败。

逐会话重建已提交动作序列并复算 margin，得到：

| 门槛/选择原因，两次合并 | Gate2 | Gate0 |
|---|---:|---:|
| cost_saving，实际选择 REBUILD | 6 | 20 |
| hysteresis：预测重建收益非正 | 36 | 32 |
| hysteresis：正收益未超过原 margin | 6 | 6 |
| hysteresis：收益已过 margin，仅稳定期阻止 | 10 | 0 |
| uncalibrated_cost_fallback | 14 | 14 |

Gate2 两重复均在相同五个 task 被稳定期阻止：`1c041…` turn2、`c6c3…` turn6、`d5b1…` turn2/3/9；这 10 次不是 10 个独立案例。Gate0 的 cost_saving 从 r1 的 12 次变为 r2 的 8 次，闭环路径并不完全稳定。新增 14 次 cost_saving 也不能直接解释为“恢复了 10 次被阻止的真实收益”，因为前面的动作、回答、输入和原生压力已经改变。

关闭稳定期后仍有 38 次保持 APPEND，其中 32 次是模型本身预测 APPEND 更便宜，只有 6 次是正收益没过 margin。下一步主要问题已不只是稳定期；不应在这批结果上继续任意下调 z 或 margin，代替改进两臂成本差的可信度。

## 等待误差、状态漂移与 P2 限制

仅对已执行 eligible 候选比较 `server_ttft_ms − forward_to_first_token_ms`：Gate2 共 72 个标签，等待 MAE **1360.25 ms**、actual−pred 均值 **+968.81 ms**；Gate0 共 70 个，MAE **780.63 ms**、均值 **+691.44 ms**。前者是收到请求至 forward-entry 的联合区间，不是纯排队/CUDA 时间。不同策略执行了不同输入与动作，因此该误差差异不是模型升级后的独立精度比较。

两方法已执行预测的 forward MAE 为 26.59/30.79 ms，decode MAE 为 794.80/791.85 ms；输出长度平均低估 105.61/101.64 tokens。当前的误差仍主要落在等待和解码/长度，不支持把 forward-only 插值当成首要瓶颈。Gate2/Gate0 中实际等待超过 `upper_ms` 达 100 ms 的分别为 12/72、10/70；upper 继续只是风险情景，不是覆盖率保证。

两者各 72 个已执行 quote 中，实际 GPU hit 低于报价且 host load 更高的分别为 32、30 次，额外 host 加载总数 127040/92800 tokens。原生当前比例和 queued 前缀年龄仍未记录，无法把这些变化精确归因到某项 admission 规则。Locked 相对逻辑 allocated 的差额仍未超各请求页舍入总量，未见大量无法归因的锁释放证据。

Gate2/Gate0 的 task/query 均相同，但两重复中实际 input IDs 仅 **6/41、8/41** 相同，output IDs 各 **4/41** 相同，action 分别 **23/41、33/41** 相同。Gate0 与 Rebuild 同 input 仅 7/41、5/41。Gate2 自身两重复 41 个输入/输出/动作都相同；Gate0 自身只有 10/41 同 input、6/41 同 output、27/41 同 action。P2 历史传播和实时压力足以让相同参数产生不同后续输入。

因此这里报告“同声明配置下观测到的闭环结果”，不把同 task 的时延差当作候选动作的因果收益。同输入/输出子集是事后选择，数量很小，保留在 JSON 供核查，不拿它替代总体。**本轮没有语义 judge 或人工质量验证，不具备 quality eligibility**；独立的 GroUSE 元评测不是这些 Cloud 回答的质量证明。样本仍只有五会话、两次有序重复，不给统计显著性、吞吐提升或完整 B6 结论。

## 下一步的可执行改进

1. 将 Gate0 保留为下一轮开发对照，同时保留 Gate2 和固定策略。当前数据支持研究较短稳定期，但不自动更改默认策略；后续性能比较应把质量检查与输出预算并列保留。
2. 按 [原生 admission 合同诊断](admission_wait_model.md#b-v1-原生合同复核与下一轮设计只读开发诊断)，先窄观测动态 `new_token_ratio`、真实 running budget 所属批次、queued `prefix_len/host_hit_length` 及最近 match 时点，再分开“容量 gate reserve”与“预计服务长度”。B-v1 的全预算离线替换已使 MAE 更差，不能在缺状态时直接扩展为线上默认。
3. 复用小型受控 fixture 验证原生容量边界、页/续传 chunk 与前缀保护，再按冻结 session 做残余尾部/等待差值校准。未来实际长度只作离线 label，不进入当前预测；保留相同训练预算与简单先验对照。候选成本差还需要成对冻结状态验证，不能只拟合被当前策略选中的臂。
4. 本轮 32/38 次 Gate0 保持 APPEND 是预测收益非正，且 quote 到执行的 tier 变化仍常见。下一轮应检验“排队期间前缀是否仍驻留”和长输入解码代价是否改变 APPEND/REBUILD 的排序，再决定是否值得加入更细模型；避免继续堆阈值掩盖成本差误差。

本报告不修改任何原始结果、冻结成本 artifact、生产 planner 或运行配置。
