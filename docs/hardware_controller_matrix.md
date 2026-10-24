# Cloud dev 控制器硬件子矩阵：16 次运行诊断

> 历史只读报告。固定7B实验入口与配置已删除，不再运行本矩阵；当前工程仅使用显式Qwen3 scope。以下数值保留原批次范围，不能作为新模型证据。

2026-10-24，来源为 `runs/remote-h200-matrix/runs/hardware-controller-dev*/results.jsonl`、各 job manifest，以及从原始 benchmark/events 只读提取的 `runs/remote-h200-grounding/runs/matrix-compact-details.json`；派生数值保存在 `runs/remote-h200-matrix/independent-summary.json`，提取脚本为 `runs/remote-h200-matrix/extract_details.py`。这批真实运行证明矩阵、流式计时、实际 receipt 和缓存遥测能够贯通，但**尚不证明质量不变条件下的方法速度收益**。

## 范围与可比条件

11 个主配置和 5 个继承 Ours 已选配置的 matched 消融，各运行一次相同的 Cloud dev 5 个会话、41 轮，P2、并发 4、seed 42、固定 top-5、Q0 模板、greedy、输出上限 1024。全部 656 轮成功，无未完成轮次；每 job 的专用服务 cold reset 已验证成功，没有生成预热。模型、tokenizer/template fingerprint、控制器源码哈希和引擎缓存配置哈希分别只有一个值。每个方法使用自身先前生成的回答，因此后续实际输入和输出会分化；固定 query 序列不意味着同条件工作量。

真实遥测为 GPU 121,920 slots、host 279,040 slots、page size 64，HiCache `write_through / cache / kernel / layer_first`，TMA 关闭。配置中的 65,536/131,072 不是此次真实引擎池容量，不能据其宣称小池压力。采样峰值与积分来自 server monotonic 的 scheduler boundaries，不能称为分配器瞬时峰值。延迟从固定检索输入就绪开始，包含客户端控制与提交完成；检索耗时不在内。首 token 为客户端 SSE 首个 token chunk，服务端 TTFT 单列。

这里没有独立未补丁服务器的 B0/B1，也没有完整 B6。`frozen.json` 的 `b6=null`、质量门未应用，选出的 Ours 只是该 dev 子矩阵的延迟候选。所有 job 的质量均 `not_validated`、0/41 已评分；不把此 freeze 当作正式质量准入。

## 延迟与实际工作量

“等会话完成”与“等会话 TTFT”先求各会话内部平均，再对 5 个会话等权，避免 7/8/9 轮会话权重不同。Control P95 是 41 轮混合分布；单次小样本不作显著性判断。

| 配置 | 等会话完成 / s | 等会话 TTFT / ms | Control P95 / ms | 实际 Prefill tokens | 1024 截断 / 41 |
|---|---:|---:|---:|---:|---:|
| B2 每轮重建 | 3.001 | 168.3 | 134.8 | 138,884 | 2 |
| A_only | 3.513 | 170.9 | 239.1 | 84,517 | 3 |
| B3 阈值 4096 | 2.962 | 162.5 | 116.2 | 135,198 | 1 |
| B3 阈值 8192 | 3.138 | 164.3 | 144.8 | 109,702 | 2 |
| B3 周期 2 | 3.284 | 160.3 | 138.0 | 125,006 | 2 |
| B3 周期 4 | 3.275 | 175.8 | 176.3 | 106,744 | 4 |
| B4 ContextPilot 适配 | 3.353 | 164.5 | 217.1 | 88,008 | 2 |
| B5 warm 阈值 .25 | 3.387 | 192.5 | 231.9 | 84,714 | 1 |
| B5 warm 阈值 .75 | 3.207 | 161.6 | 164.3 | 101,873 | 3 |
| Ours .05 / 稳定 1 轮 | 3.244 | 212.8 | 269.0 | 109,028 | 0 |
| Ours .10 / 稳定 2 轮 | 3.312 | 204.4 | 202.3 | 109,111 | 0 |
| Ours − host cost | 3.210 | 209.3 | 202.7 | 109,734 | 1 |
| Ours − decode cost | 3.440 | 167.5 | 156.0 | 105,748 | 2 |
| Ours − control cost | 3.427 | 206.0 | 209.1 | 107,597 | 1 |
| Ours − hysteresis | 3.228 | 213.5 | 281.2 | 116,105 | 1 |
| A+B，无轮间 retire | 3.347 | 216.2 | 227.3 | 108,036 | 2 |

主要诊断如下：

- Ours 已选配置比 B3-4096 等会话平均慢约 0.281 s（9.5%），五个会话分别慢 0.098、0.313、0.006、0.684、0.306 s。当前结果不能宣称 Ours 性能更优，也不能仅凭此认定质量相同下 B3 更优。
- A_only 相对 B2 实际 Prefill 减少约 39%，但完成延迟增加；Ours 也少约 21.5% Prefill 而更慢。少计算的 Prefill 不自动等于更短的完整生成，须与输出长度和控制成本一起解释。此前[冻结两臂实验](stratified_decision_validation.md)已独立发现长度差主导；不能把其具体比例直接移植到这些 P2 分化轨迹。
- 原始输出量印证工作量不相同：B2 共 21,908 tokens、B3-4096 为 21,685、A_only 为 24,790、B4 为 23,746、Ours-c0 为 23,362；Ours 相对 B3-4096 多约 7.7%。无滞回为 23,252；无 Decode 成本为 24,640。这些是不同答案轨迹的真实工作量，不能按总 tokens 简单归一化后声称质量等价。
- Ours 的客户端首 token 较 B2/B3 高约 44–50 ms，并有较高控制成本。Control 包含 quote/prepare 与持久化等墙钟等待，不等于纯 Python 算法 CPU 时间。`−control cost` 只去掉模型中的成本项，仍执行真实控制工作。
- `−hysteresis` 同时解除绝对/相对/不确定性门槛与最短稳定轮数，不是只改变某一个阈值。它产生 16 次 `cost_saving` 重建，原 Ours 两配置均没有该分支；当前差异能定位“切换被门控”的现象，不能区分是哪一个门槛造成，也不能把 0.016 s 的单次平均差称为稳定收益。
- 截断为 0–4/41，Ours 的 0 截断只表明未达到上限，未验证覆盖、引用或正确性；没有理由把短答案自动视作合格答案。

## 当前负载未充分检验的机制

**Host 恢复尚未激活。** 16 个 job 的已完成 H2D 字节均为 0，656 轮 receipt 的 `actual_host_loaded_tokens` 也全部为 0；D2H 为约 6.1–9.1 GB，说明 write-through 已写出并不等于请求从 host 恢复。各方法平均采样 host 占用约 72K–110K slots，峰值 107K–159K，低于 279K host 容量。`−host cost` 不能据此证明该项没有价值。下一组应让活跃工作集超过真实 GPU 池，再让仍在 host 的会话回访；固定顺序/间隔并检查实际 `host_loaded_tokens` 与 H2D receipt 后才算触发成功。压力实验只需加入明确的回访模式，不必重写策略。

**退休不是物理释放。** 当前 retire 合同移除会话引用，ACK 的 `physical_free_tokens=0` 是正确语义；后续缓存淘汰才可能物理回收。16 个 job 共 280 个独立 retire 操作均 ACK、无 retry；Ours 为 15 个（10 次轮间重建、5 次关闭），无轮间 retire 为 5 个关闭操作，说明消融确实改变了引用生命周期。二者平均 GPU/host 占用仍接近（约 84K/85K），单次 41 轮不足以建立退休减少容量压力的因果证据。应增加会话更替、更多 epoch 和受压回访，保留完整 reference/retire/eviction 事件，分别看引用减少、可淘汰页、实际淘汰和吞吐，不把 ACK 当 freed slots。

**报价状态和成本支持域仍影响方法 B。** 原始 plan events 中 Ours 两配置均为：初始 5 轮；后续双臂校准 21/36（58.3%），全部 `hysteresis`；其余 15 轮 `uncalibrated_cost_fallback`，其中 10 次重建、5 次 APPEND。因此汇总里的 26 次 APPEND 并不都是已校准决策。真实 pressure bucket 取 1/2/3/4，配置并发 4 不是每一轮都满载。当前 compact 只有总 `calibrated` 标志，不能唯一确定每臂是 Prefill 还是 Decode 范围不支持；下一轮诊断应明确分项支持域。

**此轮无成本切换不能只归因于固定 2×control 门槛。** 用保存的两臂预测、固定 `paired_error_ms=0`、关闭在线残差更新及原公式重建 Ours-c0 的 21 次已校准 margin：预测节省范围 −143.6 到 703.4 ms，margin 433.1 到 3259.6 ms，全部 `saving≤margin`。两轮节省已超过固定 `delta_abs=433.09 ms`，仍被每臂时间误差之和挡住；即使不考虑最短稳定轮数，这一项已经足以解释没有切换。这里的重建使用 `build_candidates` 固定 APPEND/REBUILD 顺序，数值保存于派生 JSON。应继续比较有开发 OOF 依据的条件误差与候选长度预测，并把成对差值误差作为独立校准对象；直接缩小 `z` 不能替代验证。

**B5 的低阈值缺少动作对照。** `.25` 的 36 个后续轮全部 APPEND，A_only/B4 也全部 APPEND；这些策略的差异主要来自候选组织和执行轨迹。`.75` 有 20 次 warm gate 重建，才实际覆盖 warm 门的两支。可用不同回访间隔/竞争工作集形成 GPU-ready、host-only、cold 分层，沿用相同的长度/质量评估。

下一步优先级是：对本批各方法的答案完成离线质量与截断审核；基于已有 frozen pairs 检验候选长度与条件误差组合的 OOF 选择；针对实际未触发的 host 回访和退休压力增加小而明确的负载。随后再用固定参数重复硬件运行。最终 test 和原生 B0/B1 仍保留给完整质量门控验证。
