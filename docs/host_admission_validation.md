# C：host 备份准入的首轮真实机制验证

2026-10-24，在 H200 GPU0 完成 `off`、`fixed` 两个新进程的隔离对照。两臂各生成两次，完整结果、原生 receipt、池占用、准入事件、传输完成 ACK 和退役检查全部通过。固定规则确实在原生 chunked prefill 中生效，后续请求也能补齐被跳过的备份；即时回访则揭示了额外重算的代价。**本轮未证明整体性能或质量收益，也未执行 conditioned 压力策略。**

## 固定输入与可复现记录

项目源码 `e95fdfdcff3a9713689430855bc8465f58e7b709`，283 个已跟踪文件在容器中逐个核对 Git tree；SGLang `3f36fc08331f64cd63c4fbbdf39cc402f38bb38b`，补丁 SHA256 `0ed0295274411b9fe60415e7f48604937d84d3fc894eabd9e143b0560de7488f`。Qwen2.5-7B-Instruct 固定 `a09a35458c702b33eeacc393d103063234e8bc28`，BF16/TP1/page64。GPU 池实测16,384 tokens，1 GB host 池实测17,472；write_through、kernel/layer_first、TMA关闭、chunk8192，预算8192 tokens。

输入是同一个合法非特殊 token 重复12,288次；固定输出1 token、ignore_eos开启。四次实际输出均为 `[11]`，不是问答质量任务。每臂新进程、cold reset，首次生成后 settle；再用已有 profiler 回收 GPU 副本，观察 host 前缀并执行同 epoch 的新请求。该人为设备回收不计为 host 分配压力。

原始17文件保存在 `runs/remote-h200-c-host-fixture/runs/`；导出归档 SHA256 `ca409c7c46c7afe6ede18578091eb1b178852c48b6c3eebb44f062576c6b0440`。可执行 driver 随归档保留为 `c-host-admission-review/run_fixture.py`，默认只输出计划，`--execute` 才运行。独立重算入口为 `runs/c-host-admission-review/analyze_fixture.py`，结果 `fixture-summary.json`；不读取成本模型、数据集或 qrels。

## 实际结果

| 观测 | off | fixed |
|---|---:|---:|
| 首次 host 可恢复前缀 | 12,288 tokens | 8,192 tokens |
| 首次完成 D2H | 704,643,072 bytes（672 MiB） | 469,762,048 bytes（448 MiB） |
| 回访实际 host 恢复 | 12,224 tokens | 8,192 tokens |
| 回访实际 Prefill | 64 tokens | 4,096 tokens |
| 回访 server elapsed | 29.71 ms | 109.02 ms |
| 两次请求累计完成 D2H | 704,643,072 bytes | 704,643,072 bytes |
| 回访后 host 可恢复前缀 | 12,288 tokens | 12,288 tokens |
| 退役立即释放 GPU/host slots | 0 / 0 | 0 / 0 |

fixed 的原生事件依次为：首次 Req 允许128页（此前已准入0），拒绝64页（此前已准入128）；新 Req 允许64页（此前已准入0）。实际累计准入192页，没有给共享节点永久设置禁止备份标记。off 没有进入准入计数。两边都观测到 D2H pending，之后存在真实完成 ACK；不是仅用计划页数推算完成传输。

退役前设备副本已由 pressure 清至0，host仍占12,288 tokens；因此本轮直接验证的是有host驻留时退役不立即释放host。GPU列0是该时点观测，不能单独证明有GPU驻留时的释放行为。

quote 使用 `输入+sentinel` 检查完整12,288-token缓存；回访原输入的原生匹配会保留末 token，并按page64对齐，所以 off receipt 的实际恢复为12,224、重算64。这两个数对应不同请求，不能把quote前缀直接写成回访load量。

首次 server elapsed 为off528.70、fixed420.13ms，只有各一次且服务刚启动，不能据此给出加速比。回访时fixed比off多重算4,032 tokens，并补写此前省略的4,096 tokens：累计D2H完全相同。该负例说明固定过滤可能只是延后开销；减少首次写入本身不足以支持系统收益。

## 后续验证

下一步检验[conditioned 机制](host_admission_design.md)：真实分配淘汰、高占用与其他业务工作集同时成立才约束预算；比较不同业务、同一业务多个epoch及大host无压力情形。小host不能容纳全部前缀时如实保留原生分配结果，不以缺失备份冒充收益。机制通过后，继续比较完整会话回访的恢复、重算、完成时长和质量，决定是否采用或调整该扩展。
