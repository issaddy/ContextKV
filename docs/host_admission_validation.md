# C：host 备份准入的真实机制与压力对照

2026-10-24，在 H200 GPU0 先完成 `off`、`fixed` 两个新进程的隔离对照，两臂各生成两次。固定规则确实在原生 chunked prefill 中生效，后续请求能补齐被跳过的备份；即时回访则揭示了额外重算的代价。随后完成五种压力/身份设置、共15次生成，条件策略及同业务去重的行为均通过核验，但小池各策略实际完成D2H相同。**这些结果未证明整体性能或质量收益。**

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

下面的追加实验已检验[conditioned 机制](host_admission_design.md)的压力与身份条件；接续需要比较中等容量及实际会话回访，才可判断恢复、重算、完成时长和质量的净效果。

## 五种压力与业务身份设置：15次真实生成

项目源码更新为 `58a0a089afbea875bd2ba793362a49113f9a5bd5`，290个tracked文件在容器核对；引擎补丁仍为相同0ed，模型、page64、GPU16K、chunk8192、write_through等相同。每case新进程并cold reset，三个不同token分支的输入分别8192、8192、12288，仍输出1 token/ignore_eos。五case的实际输入相同、输出均依次为`[11]`、`[320]`、`[323]`。没有回访请求，没有成本模型或质量判分。

首case为conditioned不同业务小池；其后独立运行off、fixed、同一业务三个epoch和大池负例。小池用size0/ratio0.5，实测8256 tokens；大池size2GB实测34880。五case全部完成、无合同或压力覆盖缺口，15份receipt及输出绑定、逐action谓词、完成D2H、退役均独立核对。37文件归档SHA为`a25267fec8f8a9cfa805ecf56d4d73be30f4d986f79428c61b908d7157cc670e`，原始数据在`runs/remote-h200-c-conditioned/runs/`；分析入口`runs/c-host-admission-review/analyze_conditioned.py`、结果`conditioned-summary.json`。

| 设置 | 实际host池 | 业务/epoch数 | 计划允许新tokens | 实际准入新tokens | 计划跳过 | 实际完成D2H bytes |
|---|---:|---:|---:|---:|---:|---:|
| conditioned，不同业务 | 8256 | 3 / 3 | 24576 | 24576 | 4096 | 1409286144 |
| off，不同业务 | 8256 | 未观测 | 未观测 | 未观测 | 0 | 1409286144 |
| fixed，不同业务 | 8256 | 3 / 3 | 24576 | 24576 | 4096 | 1409286144 |
| conditioned，同一业务 | 8256 | 1 / 3 | 28672 | 24576 | 0 | 1409286144 |
| conditioned，不同业务大池 | 34880 | 3 / 3 | 28672 | 28672 | 0 | 1644167168 |

off关闭了新增准入观察器，所以其raw零值不能当作“实际准入/淘汰为零”；上表相应项留未观测，实际D2H仍由独立原生完成ACK计量。请求虽然由fixture组织为三个业务，但off未提供业务身份，不能把它写成引擎测得的业务数。

不同业务小池的第三请求开始备份时，host占用8192/8256，近期分配淘汰8192 tokens；三个业务的平均prompt为9557.33 tokens，其他业务工作集估计19114.67>8256。第一批128页允许后，近期淘汰增至16384；第二批64页在累计已准入128时被拒绝。三个压力分量均来自当时原生状态，没有用profiler伪造host淘汰。

同一业务三个epoch具有相同小池高占用/淘汰，但业务数为1、其他业务工作集为0，两个第三请求action都被允许。大池则占用和工作集条件均不成立，所有28672 tokens真正准入且分配淘汰为0。两项分别证明业务去重与无压力负例，不以写出更多或更少本身给策略排名。

**小池没有实际流量收益。** 同业务策略允许最后4096 tokens，原生分配却没有成功提交它；成功准入计数保持24576，恰与其他小池case的完成D2H一致。策略跳过与原生容量失败在这个设置下产生相同的物理写出结果。小池不能容纳完整12288-token分支，不能据此将计划跳过4096写成实测节省4096。下一步应在能容纳单个完整分支、但不足以容纳多业务工作集的中等容量点，比较被保留的其他业务前缀及真实回访，而不是继续缩池制造更多skip。

本轮退役前五case均仍有12288个GPU tokens；host分别为8192或28672。全部epoch退休后两池占用保持不变，策略业务/epoch活动数退出窗口。这补充了首轮“退役时GPU已先清空”的覆盖限制，仍不等于以后原生淘汰不会释放这些槽。
