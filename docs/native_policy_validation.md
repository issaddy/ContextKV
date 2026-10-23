# 原生 HiCache 写策略：32K 池开发诊断

2026-10-24。数据在 `runs/remote-h200-native-policies/runs/native-policy-sweep/first-pass/`，每组含 `native-manifest.json`、前后 `server-info`、`job.json` 与 `summary.json`。本节只记录已完成的 32K 三策略，16K 压力实验另行记录。

三策略均使用同一 NVIDIA H200 GPU UUID、固定 Qwen2.5-7B revision、BF16、TP1、kernel/layer_first 和 TMA 关闭；stock SGLang commit 为 `3f36fc08331f64cd63c4fbbdf39cc402f38bb38b`，源码干净且无研究补丁标识。每策略全新进程，无生成预热。服务报告实际 GPU 池均为 **32,768 slots**，启动日志记录 host **279,040 slots**；由 TP1 BF16 KV 几何得到 host payload **16,001,269,760 bytes**，不是把 16GB 启动声明直接当实测。

固定负载为 Q0、P2 Cloud dev 5 会话/41 轮、并发 4、1024 输出上限，策略依次 `write_through → write_through_selective → write_back`，各一次。三个 job 全部 41/41 成功、native retraction 均 0，123 轮缓存分层字段都可解释，无缺失 breakdown。**三组实际 host-loaded tokens 均为 0：这轮没有激活 host 恢复，不能据此说三策略等价。**

| 写策略 | 等会话完成均值 / s | 全轮客户端 TTFT 均值 / P95 ms | cohort 完成窗口 / s | GPU-hit tokens | Host-loaded tokens | 输出 tokens | 截断 / 41 |
|---|---:|---:|---:|---:|---:|---:|---:|
| write_through | 3.024 | 123.0 / 326.6 | 45.380 | 54,208 | 0 | 22,441 | 1 |
| write_through_selective | 2.901 | 120.5 / 302.4 | 47.200 | 52,416 | 0 | 21,634 | 0 |
| write_back | 2.975 | 152.2 / 535.7 | 46.420 | 54,144 | 0 | 21,961 | 2 |

完成均值先在会话内平均，再对 5 个会话等权；TTFT 为现有客户端 SSE 首 chunk 口径，P95 样本数只有 41。三组 TPOT 均值约 5.323/5.345/5.325 ms/token。GPU/host 数据直接来自 native `meta_info.cached_tokens_details`，不是研究 quote、推算命中或物理 DMA bytes；本地摘要没有 allocator 精确峰值，不补造占用/传输结论。

Selective 的等会话完成均值比 write-through 低约 4.0%，但输出总量也少约 3.6%，而 cohort 完成窗口反而更长。P2 使用各策略自身生成历史，实际输入总量也分别为 193,213 / 190,714 / 192,831 tokens；同一问题列表不代表相同生成工作量。只有一次运行、不同答案和排队轨迹，不能把均值差直接归因于写策略，更不能把均值与固定 cohort 吞吐混为同一个优化目标。

三个摘要的质量均 `not_validated`，0/41 已评分，没有质量非劣结论或 B6 选择。0–2 个截断仍需随答案一起审核。Native B1 close 只有原生 dispatch 语义，报告不把它当研究路径的 scheduler retirement ACK。

下一步按已准备的 16K 同负载实验检查容量压缩是否使真实 host load 非零；保留本轮零激活结果。若仍为零，再增加明确的长前缀回访/竞争工作集，使后续请求实际复用已从 GPU 淘汰但留在 host 的前缀，以 native host breakdown 验证触发，再比较三写策略。该扩展要保持固定生成预算、记录重试/失败和答案变化，并与质量审核、轮换策略顺序的重复运行配套；不需要为了制造 host 命中而把原生路径换成研究补丁。

执行器及容量口径见 [native_policy_sweep.md](native_policy_sweep.md)。早期已执行源码保留为 `runs/native-policy-sweep/run.py`；后续参数化入口为 `python -m bench.native_policy_sweep`，两者不混写。
