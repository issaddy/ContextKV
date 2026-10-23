# 原生工作量快照的等待估计

`controller/admission.py` 提供可选的 `analytical_native_workload` 解析启发式。它估计候选开始 forward 前的等待；候选自身的 forward 和 decode 另计。`supported` 只表示快照可用、所需时间回调在各自支持域内，**不表示异质队列预测已经经验校准**。模块无持久状态，不改变原生调度。

输入只包括候选长度、当前 GPU resident 数、预测输出均值/P90、报价时运行/等待请求的逻辑长度、开发集输出长度分布，以及独立 forward/decode 时间回调。运行请求已生成 `g` 个 token 时，对开发集 `L > g` 的样本计算 `L - g` 的均值和经验 P90，并受该请求的剩余声明预算约束。`ignore_eos` 请求直接使用剩余声明预算。超出开发样本尾部或没有样本时也使用该预算，但 reason 明确记录；不会读取当前请求未来实际输出。

两个情景分别执行相同过程：

1. 以 `free_gpu + evictable_gpu` 为当前容量，先减去运行请求的预测剩余解码预留。负值表示该情景已有超订阅，并在 reason 标记。
2. 当前物理 locked 数为 `capacity - free - evictable`。按各运行请求的 `kv_allocated_tokens` 比例分配这个单一总数，作为完成后容量可用的代理；不会将多个请求共享的逻辑前缀相加冒充物理容量。虚拟完成时归还该份额及先前扣除的解码预留。
3. 将当前 waiting 列表作为 FIFO 预测顺序。它没有逐请求缓存报价，因此 queued/retracted 请求按 `input + generated` 的完整 cold 输入及剩余输出预留估计。这是保守输入假设；不是原生最终执行顺序保证。
4. 容量不足时推进到下一个虚拟完成事件，同时完成的请求一起更新容量和并发数。每个虚拟请求按开始时的并发数调用实测时间回调，服务期内固定该速率。此实现没有逐 token 模拟连续批次的速率变化。
5. 候选需求为 `input - resident + predicted_output + min(resident, evictable_gpu)`，最后一项避免同时把候选自己的 resident 前缀算作可回收空间。待该需求可容纳，即得到开始 forward 前的预测等待，再加由成本模型传入的该候选 idle 准备项。

forward 回调只定价 forward-entry 至首 token 的墙时；不使用含等待的 total TTFT。Queued 请求（包括重入队列的请求）及尚未生成首 token 的 running 请求，由 forward 产生下一个 token，随后只计剩余输出数减一的 decode。已经生成 token 的 running 请求，其全部预测残余都按 decode 定价。当前容量和所有预留已足够时，无需等待任何完成事件，也不要求无关运行请求的时间模型有支持。

Idle 准备项按候选 `(P,U,L)`——可复用前缀、新计算后缀、host 加载 token 数——在同配置的 `c=1` 实测 pre-forward 单元间插值。同一形状的不同输出预算按观测数合并，保留 host 恢复和准备开销随形状变化的证据；它不是全局常数，也不是纯恢复或纯排队时间。该准备项和 forward 的支持域由上层分别检查。不能把 `c>1` 突发请求的整体 pre-forward 查表值叠加到这里，再重复计算当前虚拟等待。

`upper_ms = max(mean_scenario, p90_scenario)` 是两种输入假设中的较大值，差额可作启发式风险量，**不是置信区间、概率上界或 Conformal 保证**。这些情景没有模拟原生优先级、具体共享节点锁、页对齐、待完成 DMA、cache victim 选择、retraction 或速率随并发变化的过程；当前原生快照本身也不是资源预留。缺少快照/容量、必须等待但所需时间无支持、保守单请求需求超容量，或无法归因的 locked 容量无法被观察到的完成事件解释时，返回 unsupported 原因。DMA/partial restore 的选择资格由上层成本模型单独处理。

合同测试覆盖共享前缀的物理归一化、候选 resident 双计、运行 future headroom、条件残余与尾部回退、queued first-token 记账、同刻/零时完成、即时可接纳路径、旧/缺失摘要与无时间支持。这些 CPU 测试验证算法记账，不代替真实等待预测残差、决策 regret 或性能实验。

首次真实运行还暴露了解码插值的 CPU 开销。用该运行的 36 次真实报价、72 个候选长度及已保存 prior 在本地重放，`_decode` 被调用 286819 次；主要时间花在对同一 context/concurrency 重复插值。成本模型现为固定标定表的这个纯函数保留最多 8192 项 LRU 派生值，插值距离设置也纳入缓存键，模型实例之间不共享，不存请求文本或 token 内容，也不增加内容哈希。72 臂的所有 `CostEstimate` 字段逐项精确一致，冷缓存本地模型计算墙时从 1.924 s 到 0.485 s；阈值键修正后的独立重放也完全一致。证据为 `runs/stage-workload-review/cache-equivalence.json`、`cache-equivalence-v2.json` 及 profile；这是本地计算优化，不能外推为端到端比例收益。`controller-stage-16384-v1` 使用优化前的固定代码，后续 GPU 对照须同时更新各策略的实现。

## 原理来源与复用边界

本模块复用下列公开工作的建模思想，并依照固定 SGLang 的可读状态实现小型估计器；没有直接移植这些系统的 scheduler。

| 原始工作 | 本项目采用的原则 | 尚未实现的上游组件 |
|---|---|---|
| [TIE，arXiv:2604.00499v2，§3–4](https://arxiv.org/html/2604.00499v2#S3) | 输出长度有尾部风险，均值之外应保留分布信息，并尊重 `max_tokens` 截断。本模块用经验条件残余均值/P90表达两种情景。 | TIE 的 log-t 参数分布、DeBERTa 参数预测器、CVaR 膨胀分数及优先级调度。我们的 P90 情景不是 TIE 分数，`L>g` 的经验残余是本项目的具体实现选择。 |
| [WAIT / Nested WAIT，arXiv:2504.11320v4，§2、§5](https://arxiv.org/html/2504.11320v4#S5) | 解码进度不断增长 KV 需求；已到达阶段揭示了继续生成的信息，未来工作与内存预留应一起考虑。本模块只用已生成长度更新残余，并配对扣除/归还预留。 | Fluid equilibrium 求解、按 decode segment 的 threshold admission、Nested WAIT 的边界队列和安全缓冲。其稳定性与避免淘汰定理依赖论文假设，不能转用于本模块。 |
| [Tail-Aware Scheduling / UniBoost，arXiv:2606.18431v1，§3](https://arxiv.org/html/2606.18431v1#S3) | 已知 prompt 大小、已完成 decode 工作和 KV 压力是无需未来标签的有效信号；单一长度点预测不足以表达内存耦合和尾部行为。 | UniBoost 的 phase-aware priority、MemGuard 的几何量化/最短运行滞回、γ 自适应及 cache-aware preemption。当前模块不重排或抢占原生请求，也不继承该工作的尾时延保证。 |
| [Sarathi-Serve，OSDI ’24 官方论文与代码入口](https://www.usenix.org/conference/osdi24/presentation/agrawal) | Prefill 与 Decode 的资源特性和同批干扰不同，需要分阶段定价，而不是把总 TTFT 当作内在 Prefill 成本。 | Chunked prefill、stall-free scheduling 与均匀批次构造。当前 forward/decode 回调复用本机实测，仍保留原生 SGLang 调度。 |

这些来源提供可继续扩展的方向：先用真实 pre-forward 残差检验“固定服务期速率”和 FIFO 近似，必要时按批次成员变化重估剩余服务；在冻结开发/测试边界后，比较经验残余、尾部截断分布和轻量候选条件长度模型；对新的内存等待误差单独校准单侧风险，再做等预算决策实验。若进一步评估 UniBoost/Nested WAIT/Sarathi 类调度，应单列原生 scheduler 实验和基线，避免把控制器选择 APPEND/REBUILD 的收益与引擎调度变化混在一起。

## B-v1 原生合同复核与下一轮设计（只读开发诊断）

本节不修改当前模型、冻结 artifact 或运行中的 B gate 消融。诊断使用 `controller-stage-16384-v1` 两次重复的 72 个 Workload 报价、144 臂预测及已执行臂标签；早期 CPU profile 文件只包含其中 r1 的 36 个报价、72 臂。入口为 `runs/stage-workload-review/admission_contract_probe.py`，结果保存为同目录 `admission-contract-probe.json`。重放基线与保存的 72 个等待预测精确一致，敏感性只比较其中原已 eligible 的 69 个已执行动作。

### 已确认的实现差异

原生源码固定为 `3f36fc08331f64cd63c4fbbdf39cc402f38bb38b`。本轮重新从官方固定 raw URL 读取 `schedule_policy.py`、`prefill_budget.py`，SHA 分别为 `28f57c7a…5aec68`、`b1d64e64…88e07e`，与工程 source manifest 及本地副本逐字一致；本地副本没有有效 Git HEAD，不能仅凭目录名声称版本正确。其它小型源码保存在 `runs/admission-source-audit/`，均从同一固定 revision 取得。

- **容量预留不是服务长度。** 原生新候选以剩余 `max_new_tokens`（上限 `CLIP_MAX_NEW_TOKENS`，默认 4096）加未驻留输入和每请求页开销检查容量；运行请求的声明剩余预算再乘动态 `new_token_ratio`，`ignore_eos` 则乘 1。当前解析模型用经验条件残余同时决定服务时间和容量，两者可能偏向相反方向。此次所有 176 个 running/waiting 摘要出现次数的上限均为 1024，`ignore_eos=false`；不能将它们当作 176 个独立请求。[原生 PrefillAdder](https://github.com/sgl-project/sglang/blob/3f36fc08331f64cd63c4fbbdf39cc402f38bb38b/python/sglang/srt/managers/schedule_policy.py#L757-L765)、[新候选容量检查](https://github.com/sgl-project/sglang/blob/3f36fc08331f64cd63c4fbbdf39cc402f38bb38b/python/sglang/srt/managers/schedule_policy.py#L1360-L1397)
- **动态比例当前没有被观测。** `scheduler.new_token_ratio_tracker.current` 会随正常 decode 下降、retraction 后更新；默认初始 0.7，最小为 `0.7×0.14=0.098`。它不是输出长度分布参数，也不能根据静态 `schedule_conservativeness=1` 推断每个报价的当前值。[原生 tracker](https://github.com/sgl-project/sglang/blob/3f36fc08331f64cd63c4fbbdf39cc402f38bb38b/python/sglang/srt/managers/scheduler_components/new_token_ratio_tracker.py)
- **原生容量有两个账本。** `PrefillBudget` 同时记录 total 与 current offset；新请求先以未舍入总需求作严格 `< remaining_total` 检查，提交预留时将本次 extend 向页取整，再加一页和生成 headroom。当前模型只有一个虚拟 free。这里 page=64、chunk=8192；chunk 未完成时，本轮提交预算不包含最终生成 headroom，不能简单把整段 prompt+输出当作所有调度阶段的即时占用。[原生预算](https://github.com/sgl-project/sglang/blob/3f36fc08331f64cd63c4fbbdf39cc402f38bb38b/python/sglang/srt/mem_cache/prefill_budget.py#L67-L168)
- **FIFO 在本次配置有源码支持。** 实际 runtime 为 `fcfs`、priority=false、max-running=8、mixed-chunk=false；FCFS 不重排，容量不足时停止这一轮队列遍历。当前不存在“因为用了优先级调度而 FIFO 必错”的证据。原生仍优先处理续传 chunk，并可能因 LoRA、pooling mode、L3 prefetch 跳过请求；这些能力不能泛化为此轮已经发生。当前报价的 running 集合还含 overlap 的 last batch/chunk，而 PrefillAdder 的运行 headroom 取传入的 `running_batch.reqs`，集合也不能默认为完全相同。[排序入口](https://github.com/sgl-project/sglang/blob/3f36fc08331f64cd63c4fbbdf39cc402f38bb38b/python/sglang/srt/managers/schedule_policy.py#L211-L278)
- **树持有的前缀不由单个请求直接释放。** 原生 release 只直接释放 `cache_protected_len` 之后归请求所有的区间，再解锁路径；共享节点只有最后一个设备锁释放时才从 protected 转为 evictable，且这不等于立刻归还 allocator。此次 running 摘要共 99 次，全部有 protected 前缀，其占逻辑 allocated 的平均比例为 96.81%。这要求保留“locked 归一化代理”的限定，不能把 protected 前缀全扣掉（最后解锁可能使其可回收），也不能逐请求承诺全部释放。此前 locked 与 allocated 差额仅与页舍入相容，未证明存在大量共享锁误释放。[KV release](https://github.com/sgl-project/sglang/blob/3f36fc08331f64cd63c4fbbdf39cc402f38bb38b/python/sglang/srt/mem_cache/common.py#L395-L440)、[FULL 锁计数](https://github.com/sgl-project/sglang/blob/3f36fc08331f64cd63c4fbbdf39cc402f38bb38b/python/sglang/srt/mem_cache/unified_cache/components/full.py#L282-L377)

### 全预算替换并未改善等待 MAE

以下情景只在诊断进程中替换容量公式，服务长度先验、forward/decode 回调、FIFO、冷 queued 前缀和释放代理全部保持原值；没有读取当前答案来预测。四行使用同一批 69 个原 eligible、且各情景均 supported 的已执行动作。后两行使用默认比例作敏感性场景，**不是还原出的真实动态比例**。

| 离线情景 | 相对基线均值变化个数 | 等待 MAE ms | actual−pred 均值 ms |
|---|---:|---:|---:|
| 当前模型 | 0 | 991.51 | +778.22 |
| candidate/queued 改用 1024 预算及 64-token 页开销 | 18 | 1083.46 | +379.96 |
| 再将 running 预留改为剩余预算×0.098 | 19 | 1106.72 | +515.22 |
| 再将 running 预留改为剩余预算×0.7 | 19 | 1070.57 | +367.07 |

统一预算减少了平均低估，但扩大了一些过度等待，整体 MAE 更差；它是需要纠正的语义混用，尚不是已验证的性能改进。既有仅改 candidate reserve=1024 的检查也仅使 MAE 从 991.507 到 991.332 ms，不能据此宣称解决主因。

77 次 waiting 摘要均尚未生成 token；其中 19 次完整输入超过 8192，但其真实缓存命中未在摘要中保存，因此不能断言实际都做了两次 chunk。所有候选报价的未缓存后缀均不超过 8192。通过输入长度及“报价时刻位于同机 submit/response 区间”匹配其它请求，99 次 running、77 次 waiting 都各有一个可能匹配；这仍不是 request identity 证明，重复出现也不是独立样本。仅作离线标签诊断，running 的条件残余均值为 445.88、实际 431.48 tokens，MAE 171.89；waiting 先验 536、实际均值 573.88，MAE 185.42。结果支持继续研究状态条件和尾部，不能简单把公共均值调高后声称改进。

### 下一轮最小观测与实现顺序

先等待冻结 B gate 结果，按是否仍有净诊断价值决定是否扩接口。建议分两步，先精确观测，再单独启用 reservation/service 分离；保持原模型为对照，不同时更换长度先验、margin 或 cache 策略。

1. **窄观测合同。** batch 可直接读 `new_token_ratio_tracker.current`（O(1)）；启动信息记录实际 clip、page、policy、chunk 和 mixed-chunk 设置。若要精确复算运行 headroom，区分 `running_batch.reqs` 与现有 overlap 汇总，或在原生创建 PrefillAdder 时记录其已算好的 running total offset；前者为 O(active)，后者是既有标量读取。不能把上一轮 offset 当作当前值，须附原生 pass/观测时点。
2. **等待前缀的现成字段及年龄。** `match_kv_cache` 已写入 `req.prefix_len`（设备命中）、`req.host_hit_length`、`req.num_matched_prefix_tokens`；只复制这些标量是 O(waiting)，无需额外树遍历、同步 CUDA、锁前缀或重新 hash。它们是最近一次原生 match 的结果，未走到队头的请求可能尚未初始化，旧结果也可能被淘汰。需要同时记录“是否匹配/最近匹配 pass”；不能把默认零当作确证 cold，也不能在 quote 中调用 `init_next_round_input` 获得新值，因为后者会改变缓存/请求状态。[原生字段写入](https://github.com/sgl-project/sglang/blob/3f36fc08331f64cd63c4fbbdf39cc402f38bb38b/python/sglang/srt/mem_cache/common.py#L220-L267)
3. **小型独立记账改动。** 给虚拟 job 分开保存 service 剩余均值/P90 与 native gate reserve；前者仍只决定预测完成时间，后者使用已观察比例、声明预算及页规则，完成时扣除/归还同一份 gate reserve。已知 queued 前缀分开定价设备驻留、host 恢复和新计算；缺失/陈旧观察继续用明确的冷前缀情景。先限于本轮 FCFS、单 FULL 池、不混合 prefill/decode 的配置，不假装复刻全部原生 scheduler。
4. **必要验证。** CPU 合同测试覆盖 ratio 变化不改变 service 长度、ignore_eos 使用完整剩余预算、clip 在减去 generated 之后、页取整与严格容量边界、queued 首 token/后续 decode、共享锁最后释放、chunk 未完成时 headroom 口径，以及缺失/陈旧前缀保持未知。原生纯预算函数可用固定源码 test double 做差分，避免只用复制公式验证自身。
5. **小规模可控采样。** 在同 16K/host 配置做“1 running+2 waiting+候选”fixture，分别覆盖 cold、GPU 共享前缀、host 前缀和较长 waiting；只预热物理上能同时容纳的前缀。先用 `ignore_eos` 的已知短/长预算验证容量记账，再用 EOS 请求验证服务长度情景。仅在 quote、原生 admission verdict、forward-entry、terminal 采样 scalar/pool 信息，保存同一请求引用作离线关联，不逐 token 记录或做内容哈希。对照 gate 是否正确、等待残差及状态漂移，再跑同代码、同质量预算的闭环方法对照；不从已执行臂误差直接推出 APPEND/REBUILD 因果收益。

目前没有新增生产 patch，也不选择上表中某个默认 ratio 作为线上“最佳值”。下一轮应先用原生真实状态消除不可辨识项，再决定是否值得加入更细的服务速率/尾部模型。
