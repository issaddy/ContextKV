# 原生工作量观测的 H200 功能验收

本轮验证只读观测补丁能在真实 SGLang 上导出原生预算、最近匹配及时间字段，并保持 guarded generation / retire 的功能合同。它没有改变 admission、成本预测或规划策略，也没有评估等待误差、吞吐收益或答案质量。20 ms 轮询会扰动运行，因此不将 fixture 时延当作性能结果。

原始文件位于 `runs/remote-h200-observation/runs/`，包括原失败报告、GPU-v2 和 host 报告、prepare/runtime/server、原始 HTTP 409 与修复后客户端复查。独立分析入口为 `runs/native-observation-review/analyze.py`，输出 `summary.json`；输入文件 SHA 保存在结果中。分析器直接读取 wire 标量复算，不调用运行时或 probe 的校验函数。

```bash
python runs/native-observation-review/analyze.py \
  --input runs/remote-h200-observation/runs/native-observation-probe-gpu.json \
          runs/remote-h200-observation/runs/native-observation-probe-gpu-v2.json \
          runs/remote-h200-observation/runs/native-observation-probe-host.json \
  --evidence-root runs/remote-h200-observation/runs \
  --output runs/native-observation-review/summary.json
```

## 版本与受控负载

Prepare、服务启动及 runtime 记录的固定 SGLang revision 均为 `3f36fc08331f64cd63c4fbbdf39cc402f38bb38b`，补丁 SHA 为 `3aac1dfdf5860f70ae868059a9a9a1fa0ed94d48f45a6e9ab7f81b6d80c1ac4b`。原始源码 hash 检查、补丁应用及 13 个文件静态编译通过。GPU 使用 H200，driver 595.58.03、PyTorch 2.14.1+cu130、CUDA 编译目标 13.0；保存的启动/runtime 记录相符，不等同独立远程进程证明。

三份报告的模型指纹与 cache 配置相同：Qwen2.5-7B-Instruct 固定 revision、GPU 16384 / host 279040 tokens、page 64、Python FULL cache、write-through、kernel IO、layer-first、TMA 关闭。FCFS、priority 关闭、chunk 8192、mixed-chunk 关闭，TP/DP/PP 均为 1。同一服务进程 PID 37788 承载三份 fixture；各次 profiling cold reset 更新协议 restart ID，不能把三个 ID 当作三个新进程。

每份 fixture 复用 `SGLangProbe.prepare`：清理专用 profiler cache，准备三个互不共享首 token 的 1024-token 前缀，再提交三个输入 9216、输出预算 256 的 `ignore_eos=True` 请求。一次仅一个完整输入可驻留；先观察第一请求已开始生成，再提交后两个。GPU-ready / host-only 预热路径不同，随后都是空候选 `quote_batch([])`，不主动为等待请求匹配前缀。报告 `pool` 是 **prepare 前** 的快照，第二、三份的非零值不能误读为 cold reset 失败或 prepare 后的起始状态。

## 独立复算结果

| 报告 | 原记录状态 | 报价数 | 1 running + 2 waiting 报价 | waiting 已知 / 未知出现数 | waiting GPU / host 正命中出现数 | 完整生成并退役 |
|---|---|---:|---:|---:|---:|---:|
| 原 GPU | failed，保留 | 158 | 50 | 102 / 50 | 50 / 52 | 3 / 3 |
| GPU-v2 | passed | 158 | 50 | 102 / 50 | 50 / 52 | 3 / 3 |
| host-only | passed | 158 | 50 | 102 / 50 | 0 / 102 | 3 / 3 |

这些是重复轮询中的**成员出现次数**，不是独立请求数。GPU-ready 的后续已知匹配中也出现 host 命中，说明预热标签不保证后续设备驻留；不能用标签覆盖真实观察。所有已知匹配均为 1024 tokens，host-only 正命中全部来自 host；本轮没有采到“已匹配、确认零命中”的状态。`None` 始终表示没有捕获过匹配，不能当作 cold。

474 个报价逐个复算下式，最大绝对差为 0，所有已知匹配时间不晚于同份报价，顺序轮询的报价时间均非递减：

```text
running_budget = Σ min(max_new_tokens − generated_tokens, native_clip)
                   × (1 if ignore_eos else native_new_token_ratio)
```

每份都读取到 ratio 从 0.7 到 0.098、clip=4096；budget 列表各有 154 次成员出现，全部 `native_finished=False`、`ignore_eos=True`。剩余预算为原 GPU/GPU-v2 的 2–255 tokens、host 的 1–255 tokens。因此，本次真实运行验证了字段读取和 `ignore_eos` 分支的公式，**没有数值覆盖 ratio 缩放、clip 饱和、finished 成员留在 budget 列表、重入队列或 overlap 集合差异**。这些边界现由固定源码 CPU 合同测试覆盖，后续正常 EOS 工作负载可补真实观测。

| 报告 | waiting 最近匹配年龄范围 ms | waiting 年龄中位数 ms | running 最近匹配年龄范围 ms |
|---|---:|---:|---:|
| 原 GPU | 9.378–1389.687 | 689.288 | 192.361–1388.313 |
| GPU-v2 | 4.650–1388.205 | 686.274 | 190.129–1388.343 |
| host-only | 4.599–1389.818 | 686.947 | 194.038–1394.473 |

年龄由同 scheduler 进程的 `quote.observed_time_ns − last_native_match.observed_time_ns` 计算，不使用客户端时钟、wall clock 或 CUDA 时间。秒级旧记录实际存在；这验证了“最近一次 match 的保存副本”语义，不能保证报价时仍驻留，更不能据此承诺未来 admission。

## 原失败与 HTTP 错误修复

首次 GPU fixture 已完成三个 256-token 生成、完整 receipt、注册引用及退役，随后因 probe 最初要求错误代码为 `EPOCH_RETIRED` 而失败。原生 `/generate` 实际返回 HTTP 409，其顶层 body 为 `object=error, message="EPOCH_RETIRED: EPOCH_RETIRED", code=409`；此前非流客户端只查嵌套 `error/detail`，把消息丢成 `HTTP_ERROR: {}`。独立 raw POST 的权威状态为 `REJECTED, terminal=true, inflight=false`。这是 fixture 预期与错误信息传播问题，**不是退休 guard 放行**；原 failed 报告没有改分或覆盖。

GPU-v2 与 host fixture 改为同时检查异常及权威 terminal 拒绝后通过；这两次仍运行旧客户端，所以原报告继续保留 `{}`。随后 `f0e845b` 的最小客户端修复保留顶层原生 message 和 HTTP 状态，不从 message 猜协议错误代码；已有嵌套控制错误保持原样。真实无生成复查得到 `HTTP_ERROR: HTTP 409: EPOCH_RETIRED: EPOCH_RETIRED`，同 host fixture restart 下状态为 `REJECTED, terminal=true`。先前拿旧 GPU restart 做检查得到 `STALE_RESTART`，是后续 profiler cold reset 后预期的隔离行为，复查记录明确保留了这一点。

## 下一步范围

本轮确认这些观测可以采集，尚未证明把它们用于预测会更准确。下一步先从普通 `ignore_eos=False` 工作负载的 `quote_observed` 事件复算动态比例分支，并按原始成员集合核查剩余预算、匹配年龄和已知/未知状态。分析器接受该 JSONL；并发响应可能按不同于快照时间的次序落盘，故仅报告文件内顺序反转，不将它误判为服务器时钟故障。未知或未采到的状态属于覆盖不足。

随后导入的 `p3-user-topic-v1` 两组各 41 轮、405 条事件均没有 `quote_observed`（固定 `normal_rag` 路径不报价）。`runs/native-observation-review/p3-eos-summary.json` 记录每组 quote_count=0、budget_formula_coverage=not_observed；这批普通 EOS 生成没有补上比例分支的运行时证据，也不是补丁失败。

之后才单独比较 service 长度与 native gate reservation 分离、已知 queued 前缀及其陈旧情景；保持当前模型为对照。还需实际 pre-forward 标签、同配置负载和独立决策评估，不能从这次功能通过推导等待精度、APPEND/REBUILD 收益、一般调度正确性或质量准入。
