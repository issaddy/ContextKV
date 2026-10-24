# 16K 工作量等待模型：开发阶段实跑

> 归档记录：保留当时的设计、结果与复用依据；其中“下一步”和执行计划不代表当前任务。当前工作仅以[实现清单](../implementation_plan.md)为准。

本轮验证了新的只读工作量观测、阶段成本与解析等待模型能在真实容量压力下参与选择，**没有观察到整体时延优于现有固定策略**。四方法、两次重复均完成 41 轮，共 **328 成功、0 失败、0 unfinished**；不因结果不利而删去某组或改换口径。

原始导入目录为 `runs/remote-h200-stage/runs/controller-stage-16384-v1`。复核入口与逐轮结果：

```bash
.venv/bin/python runs/stage-workload-review/analyze.py
```

输出 `runs/stage-workload-review/summary.json` 保存原文件 SHA、逐轮预测和服务端阶段标签、每会话结果、每重复方法配对、失败/缺失统计及门槛重建。它不访问引擎、不重新拟合模型。统计另经独立代理从原始 benchmark/events 抽查一致。

## 固定范围与来源

Cloud dev 的 5 个会话、41 轮，P2、c=4、seed=42、Q0、1024-token 输出上限；GPU 池 16384、host 池 279040、page=64，HiCache `cache/write_through/layer_first/kernel/TMA=0`，RAYON=4。8 份 manifest 的模型指纹、引擎/cache 配置、输入与 split、context、运行环境、控制器源码完全一致。每个 job 的 cold hook 均成功，GPU/host 唯一占用均从 0 开始；每组保留 41 个 native terminal 标签，遥测未截断，成功轮的 audit 完整。

实跑源码是 **`bc6c803`**。按 manifest 的“排序 controller/*.py 文件名加源码内容”规则，从该 Git 树重算 SHA 为 `131a4aa75adc6e718cc0238e4562b6e02ef5ff5ba80527731dfa60aeceb42a7d`，与全部 manifest 精确相符。**本轮不包含后续 decode LRU 缓存优化**。

Workload 使用 stage artifact `3d455c64…4ebdaf3`，Strict 使用同 16K 配置绑定的原始 total-TTFT lookup `f19bdb17…6c00dc`。Workload 采用 `native_workload` 可选模型；Strict 保留容量不足时不参与成本选择的原规则。两者均 `delta_abs=0, delta_rel=.05, z=1, min_stable_turns=2`，fallback 阈值 8192。Fixed_4096 按候选长度阈值选择，Rebuild 每轮重建。固定方法虽继承配置中的 artifact 路径，但没有两臂成本预测事件。

每次重复共用一个引擎进程，方法之间只做 KV reset：r1 为 Workload→Strict→Fixed_4096→Rebuild，r2 为 Strict→Fixed_4096→Rebuild→Workload。这是一次轮换顺序的开发实验，未实现每方法相同 JIT/内核初态、随机化多次重复或完整原生 B6 对照。

## 时延与实际执行工作

“等会话完成均值”先在每会话内平均 `complete_ms`，再对 5 个会话等权；该字段为服务处理到提交完成。TTFT 为 benchmark 到达至客户端首 chunk。合并 P95 是 82 轮的描述统计，不能把这些相关轮次当作 82 个独立样本。

| 方法 | r1 等会话完成均值 ms | r2 等会话完成均值 ms | 两轮等会话均值 ms | 合并完成 P95 ms | 合并 TTFT 均值 ms |
|---|---:|---:|---:|---:|---:|
| Workload | 6103.27 | 6023.99 | 6063.63 | 10207.37 | 3257.38 |
| Strict | 5842.56 | 5332.19 | 5587.38 | 8574.01 | 2780.26 |
| Fixed_4096 | 4678.54 | 3982.87 | 4330.70 | 8312.71 | 1636.03 |
| Rebuild | 4490.54 | 4480.43 | 4485.49 | 7756.28 | 1762.84 |

Workload 的等会话均值比 Strict 高 8.52%、比 Rebuild 高 35.18%；两轮合并后五个会话均比这两个方法慢。该结果不支持把“选择覆盖提高”当作性能改善。合并 control 均值分别为 Workload 262.98、Strict 150.50、Fixed_4096 97.90、Rebuild 103.41 ms，CPU 控制成本也需继续优化。

| 方法，每方法 82 轮 | APPEND / REBUILD | 输入 tokens 总计 | 输出 tokens 总计 | 实际 Prefill tokens | 实际 host 加载 tokens | 截断轮数 |
|---|---:|---:|---:|---:|---:|---:|
| Workload | 51 / 31 | 522198 | 47086 | 212822 | 209856 | 2 |
| Strict | 34 / 48 | 468906 | 46714 | 224298 | 171328 | 4 |
| Fixed_4096 | 2 / 80 | 378277 | 43331 | 269989 | 17472 | 4 |
| Rebuild | 0 / 82 | 382693 | 43863 | 277029 | 21120 | 1 |

Workload 减少了实际 Prefill 工作，但输入更长、host 加载更多，完成时间没有因此下降。328 轮的实际 Prefill 均等于 input−device-hit−host-load，没有观察到额外执行的 extend token；这不是所有 retraction 次数为零的证明。截断与输出长度不同也不能当作语义质量相同的证据。

## 选择资格、回退与稳定期

每方法 10 个初始轮只有一个候选；两种成本方法各有 72 个两臂决策、144 个候选预测。

- Workload：121/144 个候选 `selection_eligible=true`，52/72 个决策两臂均可选择，fallback 为 20/72；r1/r2 分别 9/11 次。其 144 个 `calibrated` 均为 false，正确保留“解析等待尚未经验校准”的区别。
- Strict：19/144 个候选 calibrated，仅 5/72 个决策两臂均可选择，fallback 为 67/72；r1/r2 为 33/34 次。
- Workload 的 120 个 `capacity_pressure` 候选中，99 个可选择，55 个被实际选中；旧统一 free-slot 门槛已经在新模式中被实质绕开。DMA 不确定性与无支持计时没有被伪装成 calibrated。
- Workload 的 20 次 fallback 中，18 次仅有阶段计时支持域不足，1 次同时有计时域和等待计时不足，1 次仅等待计时不足。这里的 fallback 是明确记录的策略路径，不是请求失败。

Workload 最终 3 次 `cost_saving`、49 次 `hysteresis`。用已成功提交的动作序列还原 `turns_since_rebuild`，49 次保持 APPEND 可分为：33 次预测 APPEND 本身更便宜；7 次重建预测收益为正但未超过误差 margin；**9 次收益已超过 margin，却被两轮稳定期阻止**。最后一组预测收益为 0.90–2.68 秒，margin 为 0.46–0.70 秒。它们是有具体任务和前置状态的消融目标，但 REBUILD 臂未执行，不能把这些预测收益称为实测可得收益。

## 等待误差与状态漂移

只能对已执行动作验证标签，不能据此计算两臂真实 regret。72 个带预测的成功轮中，69 个已执行候选 eligible：真实 `server_ttft − forward_to_first_token` 与预测等待相比，MAE **991.51 ms**、有符号偏差 actual−pred **+778.22 ms**。这是收到请求至 forward-entry 的联合区间，不是纯 queue 或 CUDA 恢复时间。

| 已执行预测分组 | 个数 | 等待 MAE ms | actual−pred 均值 ms |
|---|---:|---:|---:|
| FIFO waiting + logical release proxy | 47 | 1164.07 | +1031.29 |
| 上述条件且虚拟 decode headroom 超订阅 | 5 | 1611.10 | +1611.10 |
| 无 waiting、需要 logical release proxy | 5 | 1157.79 | −537.51 |
| idle capacity available | 13 | 0.225 | +0.225 |

表内包含全部可取得标签的对应原因，资格口径与 69 个 eligible 子集不同。13 个 idle 预测没有秒级错误；其中 12 个快照没有 running/waiting，另 1 个有 3 running、0 waiting。不能把本次主要误差归因于 idle 插值。69 个 eligible 标签中，22 个高于 `upper_ms`；其中 13 个只是 idle 项的亚毫秒误差，超过 upper 达 1 ms 的为 9 个、达 100 ms 的为 8 个。`upper_ms` 是风险情景，不是覆盖率保证。

Workload 的 72 个已执行预测中，forward 阶段 MAE 为 **25.18 ms**，decode 总时间 MAE 为 **797.29 ms**，当前请求输出长度平均低估 **90.38 tokens**。这些分项提示应优先研究排队和长度误差；不能把其它排队请求的未来输出当成线上输入。Strict 的预测是 total TTFT，不能直接拿来与 forward-only 标签比较。

三个最大 eligible 等待低估均观察到 1 running、2 waiting：r1 `c6c3b02ca32795af64c903dd76700517<::>7` 为预测 2645.06、实际 6006.67 ms；r2 `d5b1e735a040853ed361a3dfde1b8ef0<::>3` 为 2646.57 / 5979.45 ms；r1 同会话 turn 4 为 1505.70 / 4734.90 ms。独立原始事件复核显示相关排队请求的实际输出存在长于公共 536-token 先验的情况；该观察只用于离线解释。

Workload 的 72 个已执行报价中，29 个实际 GPU hit 小于 quote ready 前缀，同时实际 host load 大于当时报价，总额外 host 加载为 119936 tokens。Strict 对应 34/72。报价是瞬时观察，并未预留前缀；虚拟排队期间的 tier 变化需要纳入后续风险建模。

物理归因也做了单独检查：Workload 59/72 个快照的 locked=`capacity−free−evictable` 大于运行请求逻辑 allocated 总和，最大差 161 tokens、比率 1.01481；Strict 为 61/72、123 tokens、1.01770。两者都没有“无 running allocation 但存在 locked”，且 locked 均未超过各请求按 64-token 页分别向上取整后的总量，waiting allocation 全为零。差额与页舍入相容；本次没有证据表明大量无法归因的锁被虚拟释放，也不能据此证明逐请求真实释放归属。

## 配对与质量限制

所有方法每重复的 task/query 集合相同，但 P2 回答进入后续历史。Workload 与 Strict 的同输入 token IDs 仅 r1 **8/41**、r2 **7/41**，同输出为 3/41、2/41；与 Rebuild 同输入均 5/41，同输出为 2/41、3/41。Workload 自身两重复也只有 7/41 同输入。方法改变了动作、历史和可见材料，按同 task 的时延差不是候选动作的因果收益；事后同输入/输出子集同样有选择偏差。详细计数及子集保留在 JSON，未用小子集替代总体结果。

这仍是 5 个会话、两次有序重复的开发观察；不报告显著性或独立测试胜出，不声称质量相等。当前未做本轮语义 judge/人工质量验证，截断和最终 token 一致率仅是可审计事实。

## 已完成的 CPU 改进与下一步

本轮运行后，decode lookup 增加实例内 8192 项有界缓存，键包含 context、concurrency 与可变距离阈值。同一台 Linux 机器的 72-arm CPU 快照重放，以 before→after 和 after→before 两种顺序测得：before 墙时 **5.900/5.978 s**，after **1.516/1.519 s**；process CPU 对应 **5.882/5.958** 与 **1.510/1.513 s**。所有 CostEstimate 字段精确一致，平均墙时下降 74.45%，每候选约 82.49→21.08 ms。记录在 `runs/stage-workload-review/linux-cache-equivalence.json`。这是本机 CPU 重放结果，不是已经重跑的完整 GPU 闭环收益；后续对照应两边一起使用优化后的代码。

下一轮已有明确可执行项：

1. 用现有 `planner.min_stable_turns=0` 做匹配消融，检验那 9 次被稳定期阻止的选择及其它轮次的连锁影响，保留原 margin，不预设收益。
2. 分开“原生 admission 的声明预算预留”与“预计服务时间的输出长度”。独立离线检查中，仅把候选 reserve 改为已知 1024、其它先验不变，仅改变 69 个 eligible 预测中的 7 个，MAE 991.507→991.332 ms，几乎不变；这项语义修正不能被宣称已解决主因。
3. 继续校准排队请求的条件残余尾部，并检查报价至提交期间的状态变化、虚拟 admission 后的 GPU→host 风险及服务速率变化。使用已冻结前置特征，未来输出仅作离线标签；同预算比较、逐会话统计与质量评估继续保留。

本阶段不修改这些运行的冻结 artifact 或规则。覆盖提高、计算优化和等待预测准确性分别记录，下一轮以实测效果决定继续采用、修正或撤回具体算法改动。
