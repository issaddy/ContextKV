# 16K GPU 池的控制器开发矩阵：独立核验

本轮在真实 H200 上完成六方法各 41 轮，共 246/246 成功，没有未完成依赖轮次或超时会话。有限 GPU 容量确实触发了 host 恢复与 H2D DMA，但成本策略 B 没有一次按预测成本切换：36 个双候选轮次中 34 次回退、2 次滞回。首要限制不是总输入长度超出标定范围，而是当前 quote 把“空闲槽位不足”统一标成 `native_estimate_uncertain`，使常态容量压力下的成本决策被关闭。此处保留原实验与策略，不修改 calibrated 标记或放宽支持域。

原始目录为 `runs/remote-h200-capacity-matrix/runs/controller-capacity-16384-v1/gpu-16384/r1/`。独立重算脚本与 JSON 在 `runs/controller-capacity-review/{analyze.py,summary.json}`，未覆盖原始报告。按 task ID 核对了全部 audit 与 raw run 的问题、自身历史、当前检索证据、回答、output IDs，以及 audit input IDs 对 receipt input_hash；按 transfer ID 去重并独立重算占用积分。原报告内的远端绝对路径没有被当作本地文件路径执行。

## 条件与初态

- 单 GPU 0，NVIDIA H200；固定 Qwen2.5-7B-Instruct revision `a09a35458c702b33eeacc393d103063234e8bc28`；固定 patched SGLang base commit `3f36fc08331f64cd63c4fbbdf39cc402f38bb38b`。
- Cloud dev 同 5 会话/41 轮，P2、Q0、并发 4、seed 42、输出上限 1024、模型 context 32768；本轮只有一个重复。所有方法都使用自身生成历史。
- 实际 `/server_info` 与 scheduler pool 一致：GPU 16384 slots，host 279040 slots，page 64；host 参数 16 GB、write-through、layer-first、kernel、TMA=false。YAML 中保留的 mock capacity 默认值不能代替这些实际观测。
- 每方法前独立调用 cold reset，六次都核验 GPU/host unique tokens=0、active/queued=0。方法之间共享同一个已启动服务器的 JIT/kernel 状态；顺序为 Ours、Ours_no_retire、Fixed_4096、Fixed_8192、Rebuild、Rebuild_no_retire。父控制器启动环境记录 RAYON_NUM_THREADS=4，实际工作线程数未测量。
- 使用此 16K 池新采集并绑定的 lookup artifact。独立重算使用的文件 SHA 与六份 manifest 完全一致；engine fingerprint/cache_config_hash 匹配，四个 profile run IDs 均保留。这证明配置绑定一致，不单独证明跨硬件等价。运行上限明确覆盖 artifact 内旧输出 cap；实际 serving cap 为 1024。

native policy sweep 每个 policy 都新启动 stock server；此处是 patched server 内 KV-only reset，且后续 job 的 kernel 状态已热。因此不能把本表与 native sweep 的时间直接拼成 B0/B1/B6 排名；本轮也没有 B6 选择或质量准入。

## 生成、首 token 与控制时间

完成时间采用同一客户端时钟 `t_commit - turn.t_arrive`，P2 时间起点是固定检索输入准备好；TTFT 为 `t_first - t_arrive`。表中完成均值先求各会话平均，再对 5 会话等权；TTFT/control 是 41 轮的统计，二者权重口径不同。原始 turn-weighted 完成均值与逐会话值保留在 JSON，不能混用。

| 方法 | REBUILD / APPEND | 等会话完成均值 s | TTFT 均值 / P95 s | control 均值 / P95 ms | 输出 tokens | 截断轮数 |
|---|---:|---:|---:|---:|---:|---:|
| Ours | 23 / 18 | 5.325 | 2.612 / 5.251 | 147.96 / 274.79 | 22537 | 2 |
| Ours_no_retire | 23 / 18 | 5.313 | 2.601 / 5.245 | 132.53 / 257.98 | 22537 | 2 |
| Fixed_4096 | 40 / 1 | 4.461 | 1.858 / 5.173 | 92.38 / 187.33 | 21122 | 2 |
| Fixed_8192 | 24 / 17 | 5.302 | 2.650 / 4.713 | 96.06 / 204.89 | 22031 | 1 |
| Rebuild | 41 / 0 | 4.122 | 1.546 / 4.112 | 101.99 / 207.70 | 20690 | 0 |
| Rebuild_no_retire | 41 / 0 | 4.360 | 1.847 / 5.258 | 69.62 / 95.50 | 20481 | 1 |

成功生成与完整回答不同：246 个生成结果中有 8 个截断，分布在 4 个不同 task 上。所有方法质量状态仍是 `not_validated`，没有自动 judge 或人工分数。P95 只有 41 个 turn，单次结果不宜作强尾延迟论证。

Ours 与 Ours_no_retire 的 41/41 input_hash、完整 output IDs、答案与动作均完全相同，actual prefill 也相同；这是本轮最可解释的 C 对照。Ours 等会话均值慢 12.70 ms（约 0.24%），五会话差值分别约 8.71–15.09 ms，control 均值增加 15.43 ms。它没有展示时间收益；一次顺序运行也不足以把这点差异全部归因于 C。

Rebuild 与 Rebuild_no_retire 只有 6/41 输入、1/41 输出和答案相同。前者 4.122 s 对后者 4.360 s 的观测不能归因于退役机制；输出差异经 P2 自身历史继续改变后续提示。Ours 与 Fixed_8192 也仅 7 个输入、2 个输出相同，动作只在 38/41 轮一致，不是冻结输出的速度对照。Rebuild 的较低时间不能代替质量合格后的整体胜出结论。

## 34 次回退的逐因子核验

对两组 Ours 的 36 个双候选轮次、各 72 个候选，使用原 artifact 与记录的 quote/pressure bucket/输出先验重放 `_prefill`、全部实际预测范围内的 `_decode` 支持判断，并核验重算 calibrated 与原事件逐臂相同。没有用当前回答长度重新拟合或替换预测先验。

| 因子 | Ours | Ours_no_retire |
|---|---:|---:|
| 双候选回退轮次 | 34 / 36 | 34 / 36 |
| 发生 native_estimate_uncertain 的轮次 | 34 | 34 |
| free-slot 不足的候选数 | 63 / 72 | 62 / 72 |
| 在途 H2D 或 D2H 导致 uncertain 的候选数 | 0 | 0 |
| sparse-neighbor 距离超界的候选数 / 轮次 | 7 / 5 | 7 / 5 |
| 总输入 n 超出标定范围 | 0 | 0 |
| decode 支持范围不足 | 0 | 0 |
| 部分恢复或 host quote 不可用 | 0 | 0 |
| 两候选时间模型均支持、仅因 native uncertain 回退 | 29 轮 | 29 轮 |

当前 quote 的 uncertain 条件是 `inflight_h2d/d2h > 0` 或 `free_gpu_tokens < input_len - ready_gpu_prefix_tokens`。本轮触发的全是后者：它表示原生 scheduler 需要处理容量压力，不能等同于真实的在途 DMA 或不可确定恢复。现有 cost model 将这些不同来源统一降为 uncalibrated，导致 B 在最需要权衡缓存容量的场景中大量退回 8192 阈值。

其余 5 个回退轮次的 7 个候选还同时存在稀疏插值问题，最近归一化距离为 0.506–0.689，超过既定 0.5；它们仍都处于总 n=1024–12288 的采样包络内，不能误称长度越界。并发 bucket 分布为 c4:26、c3:4、c2:1、c1:5，采样主要位于 c1/c4；c2/c3 以及非规则 prefix/load 组合仍需要谨慎处理。这里不通过放宽 distance 消除失败标记。

仅有的两次双臂 calibrated 都在 Cloudant 附件会话：turn 2 的预测 REBUILD 节省约 177.3 ms，双臂误差之和约 4120.8 ms；turn 7 的预测节省为 -139.9 ms，误差之和约 3759.8 ms。两次都保留 APPEND，没有 `cost_saving`。时间模型支持与具备有用的动作区分能力是不同条件。

下一步应拆分过宽门控：在保持原生 admission、锁、真实恢复语义和原支持域的前提下，将可观测容量压力、在途 DMA、无法确定的恢复分别建模，比较压力条件误差及独立等待成本模型。应使用冻结的成对执行检查候选等待/恢复与决策 regret，不能只把 calibrated 改为 true，也不能把扩大采样本身当成算法收益。本轮没有修改这一门控或调参。

## 实际缓存工作、占用与 DMA

| 方法 | GPU 命中 tokens | host 恢复 tokens | actual prefill tokens | host 采样峰值 slots | GPU slot·s | host slot·s | H2D / D2H GiB |
|---|---:|---:|---:|---:|---:|---:|---:|
| Ours | 37504 | 87104 | 110575 | 132032 | 1092822 | 5606857 | 4.652 / 7.051 |
| Ours_no_retire | 41408 | 83200 | 110575 | 132032 | 1106626 | 5596239 | 4.443 / 7.051 |
| Fixed_4096 | 39936 | 15232 | 135040 | 154944 | 915410 | 5877989 | 0.813 / 8.275 |
| Fixed_8192 | 37504 | 83008 | 109012 | 129664 | 1111433 | 5489545 | 4.433 / 6.925 |
| Rebuild | 39168 | 12352 | 137510 | 156864 | 871080 | 6007460 | 0.660 / 8.377 |
| Rebuild_no_retire | 44672 | 7104 | 137496 | 156736 | 976868 | 6532971 | 0.379 / 8.371 |

六方法 GPU 采样峰值均为实际上限 16384，host 峰值都低于 279040（未观察到 host 池耗尽）。每轮实际 prefill 来自 receipt，不以 prompt 减 hit 推算。Ours 比 noC 多从 host 恢复 3904 tokens，GPU 命中相应少 3904，prefill 不变；较多 host 命中不是性能收益的证据。

占用使用全局唯一 slot，按 server monotonic_ns 的左端采样保持积分到最后一个样本，不与客户端时钟相减。六组各 435–473 个样本，`pool_sample_scope=scheduler_boundaries`，`events_truncated=false`；覆盖生成及会话关闭清理边界，是采样峰值/积分，不是 allocator 精确瞬时峰值。不同方法完成时长约 59.14–73.94 s，积分同时反映占用和暴露时间。Ours/noC GPU 积分低约 1.25%、host 积分高约 0.19%，host 峰值相同，没有显示明显的容量释放收益。

DMA 字节来自真实 completed transfer 事件，按 transfer ID 去重；六组无 pending transfer。全部事件有 `cuda_event` duration，字节与记录的 tokens×57344 一致，H2D totals 与本轮 receipt host 恢复总量一致。Ours/noC 分别 20 次 H2D、79 次 D2H；其 H2D duration 求和约 221.39/228.16 ms，D2H 约 162.51/164.34 ms。CUDA 区间可能重叠，求和不是请求关键路径等待，也不能用这些毫秒直接解释约 2.6 s 的端到端 TTFT。

退役 requested/acked/unique operation 逐项匹配：Ours 23/23（18 次回退重建后的旧 epoch，加 5 次关闭），noC 5/5（仅关闭）；Rebuild 41/41、Rebuild_no_retire 5/5。全部 ACK 的 immediate physical_free_tokens=0，全部本地 pending retire operations=0。这验证逻辑退役协议完成，但没有即时物理回收；关闭后 cache 中仍可保留无引用副本，不应把 ACK 解释为清空 KV。

## 进程结束与范围

生成、报告保存与逐会话清理均已成功。随后 owned server 最终返回 -9，这不等于正常 SIGTERM 退出，也不是生成阶段失败。已核本地 `startup-tail.txt`：03:05:14 收到 SIGTERM 开始 draining，03:05:15 detokenizer -15 触发 SIGQUIT 清理，03:05:20 报 no live schedulers 并调用包含 parent 的 kill_process_tree。不能仅凭 -9 推断 helper 等待 30 秒超时；此处记录上游信号清理行为，没有修改或宣称修复 shutdown。

本轮证实有限 GPU 下 host 恢复与 DMA 已激活，定位了抑制 B 的过宽压力门控，并给出输入/输出全等的 C 对照负面结果。它仍是 5 个开发会话的一次控制器诊断：没有最终 heldout、没有 B6、没有质量非劣结论。后续算法改进应另存新结果，保留此轮作为固定基准。
