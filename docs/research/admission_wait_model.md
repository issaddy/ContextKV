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

## 原理来源与复用边界

本模块复用下列公开工作的建模思想，并依照固定 SGLang 的可读状态实现小型估计器；没有直接移植这些系统的 scheduler。

| 原始工作 | 本项目采用的原则 | 尚未实现的上游组件 |
|---|---|---|
| [TIE，arXiv:2604.00499v2，§3–4](https://arxiv.org/html/2604.00499v2#S3) | 输出长度有尾部风险，均值之外应保留分布信息，并尊重 `max_tokens` 截断。本模块用经验条件残余均值/P90表达两种情景。 | TIE 的 log-t 参数分布、DeBERTa 参数预测器、CVaR 膨胀分数及优先级调度。我们的 P90 情景不是 TIE 分数，`L>g` 的经验残余是本项目的具体实现选择。 |
| [WAIT / Nested WAIT，arXiv:2504.11320v4，§2、§5](https://arxiv.org/html/2504.11320v4#S5) | 解码进度不断增长 KV 需求；已到达阶段揭示了继续生成的信息，未来工作与内存预留应一起考虑。本模块只用已生成长度更新残余，并配对扣除/归还预留。 | Fluid equilibrium 求解、按 decode segment 的 threshold admission、Nested WAIT 的边界队列和安全缓冲。其稳定性与避免淘汰定理依赖论文假设，不能转用于本模块。 |
| [Tail-Aware Scheduling / UniBoost，arXiv:2606.18431v1，§3](https://arxiv.org/html/2606.18431v1#S3) | 已知 prompt 大小、已完成 decode 工作和 KV 压力是无需未来标签的有效信号；单一长度点预测不足以表达内存耦合和尾部行为。 | UniBoost 的 phase-aware priority、MemGuard 的几何量化/最短运行滞回、γ 自适应及 cache-aware preemption。当前模块不重排或抢占原生请求，也不继承该工作的尾时延保证。 |
| [Sarathi-Serve，OSDI ’24 官方论文与代码入口](https://www.usenix.org/conference/osdi24/presentation/agrawal) | Prefill 与 Decode 的资源特性和同批干扰不同，需要分阶段定价，而不是把总 TTFT 当作内在 Prefill 成本。 | Chunked prefill、stall-free scheduling 与均匀批次构造。当前 forward/decode 回调复用本机实测，仍保留原生 SGLang 调度。 |

## 已实现的只读观测扩展

研究补丁提供以下只读字段；当前交付成本配置使用 `native_budget`，较旧的 `strict` / `native_workload` 读取路径仍保留。旧 wire 缺字段时，schema 将其解码为 None；不会填零冒充已知状态。`/cache/quote` 与 metrics 共用同一个只读 workload snapshot，HTTP client 沿用已有一次共享 dataclass 解码，没有增加内容 hash 或请求缓存。

| 位置 / 字段 | 真实来源与语义 |
|---|---|
| batch `native_new_token_ratio` | 报价时 `scheduler.new_token_ratio_tracker.current` 的实际值，不以启动默认值代替动态值 |
| batch `native_max_new_tokens_clip` | 固定原生模块的 `CLIP_MAX_NEW_TOKENS` 实际值，包含启动环境覆盖 |
| batch `native_running_budget_requests` | 当前 `scheduler.running_batch.reqs` 的原顺序成员摘要，包含原列表内尚未清理的 finished/retracted 成员；不是旧字段的 overlap/last-batch/chunk 联合集合 |
| batch `native_running_budget_tokens` | 在上述当前成员与实际 ratio 上直接调用原生 `PrefillAdder._get_running_request_total_token_offset` 求和。保留原生“先减 generated，再 clip”、ignore_eos 与未过滤成员的行为；是当前 running headroom 的可复算快照，不是已执行 admission verdict 或完整 total/current offset |
| request `request_id` | 直接复制已有原生 `req.rid`，关联服务集合与预算集合及终态 receipt；不创建新 UUID、hash 或按长度猜成员。旧 wire 为 None；它不是业务 session ID |
| request `native_finished` | 同次读取 `req.finished()`；旧引擎未知为 None，不能用这个标志擅自过滤预算列表 |
| request `last_native_match` | 最近一次被研究 observer 捕获的原生 `match_kv_cache` 完成结果；未捕获为 None，不从 req 默认 prefix=0 推断确证 cold |

`last_native_match` 保存五个标量：`observed_time_ns`、本次参与匹配的 `input_tokens`、`device_prefix_tokens`、`host_hit_tokens`、`matched_prefix_tokens`。Hook 在原生函数已经完成 match 和请求字段更新之后，只复制结果与 `time.monotonic_ns()`；不修改原生返回值，不再调用 match、树遍历、锁、DMA 或 GPU 同步。每请求只覆盖一份记录，不积累历史。未启用研究 runtime 的 cache 不记录。

`last_native_match.observed_time_ns` 与 batch `observed_time_ns` 均为**同一 scheduler 进程的 monotonic 纳秒**，可直接相减得到此次匹配观测的年龄；不得与 wall clock、跨机器客户端时钟或 CUDA event 相减。年龄不是缓存仍驻留的证明：匹配后可能被淘汰、加载或重入队列，单次报价也不预留前缀。Saved match 是匹配时的字段副本；之后 req.prefix_len/host_hit_length 的原生变化不会把旧记录悄悄刷新。已有候选纯 peek 的行为保持不变，新增等待请求观测不主动触发任何匹配。

Running 预算和旧服务工作量摘要使用不同成员集合，所以只在少量长度标量上有 O(active) 的必要重复；没有重复 token 数组、文本或物理 slot。Observer 每次原生 match 的额外工作为 O(1)，workload 导出为 O(active+waiting)，不会扫描缓存树。报价时重算的 running headroom 使用当前成员；下一次实际 prefill pass 可能先合并、过滤或改变批次，因此不能把这个标量叫未来 admission 的精确承诺，也不包含 mixed-decode、已准入 prefill 和页开销等完整预算项。

本地验证直接复用固定原生 offset、ratio tracker、match 函数源码片段，检查 finished/retracted/overlap 成员差异、ratio 衰减、remaining-then-clip、ignore_eos、原生返回与字段不变、未知匹配/过时匹配/只读副本、旧 wire、嵌套 dataclass HTTP 合同，以及旧等待估计在加入观测后逐值不变。完整补丁在干净固定源码上应用并静态编译；该结果不代替 GPU/IPC 验收。


## 当前成本配置：native_budget 预留与服务分账

`cost_model.admission_estimator: native_budget` 是当前交付成本配置的显式选项，必须配合同配置的 `forward_first` stage artifact。它复用同一成本模型，不改变原生调度。新结果的 `waiting_estimate_kind=analytical_native_budget`，`calibrated` 仍为 false；`selection_eligible` 只表示当前启用的预测合同和实测阶段支持域满足，不表示新等待预测已经校准。保存/加载成本 artifact 会保留选项，旧 wire 与旧 artifact 仍可由原模式读取。新模式缺少 request identity、原生比例/预算、容量或 artifact 中的实际 page_size 时明确返回 unsupported，不按请求长度匹配成员。

实现分开维护三类量：

1. **total 容量账本。** 初始值为 `free + evictable − native_running_budget_tokens`。各服务请求完成时，只归还相同原生 `request_id` 对应的预算份额；其公式为 `min(max_new_tokens − generated_tokens, clip) × ratio`，ignore_eos 使用 1。预算列表内不属于服务集合的 finished/retracted 成员继续占着该份额，并记录 `unmatched_native_budget_held`，不为了预测可接纳而擅自删去。Overlap 服务成员若不在预算列表，不能凭空归还一份原生 offset。
2. **current 容量账本。** 初始值为 `free + evictable`，不扣 running 输出 headroom。Queued 虚拟完整 prefill 的 current 需求为页取整后的完整输入加一页，total 另加完整 clipped 输出预算。候选 gate 用未驻留输入、声明 `max_new_tokens`（受实际 clip 限制）与一页开销，并保留原生严格 `< total` 及页分配 `<= min(current,total)` 边界。对重入 waiting 的请求，gate 使用剩余预算，但完整 prefill commit 使用原生完整预算，两者没有混同。
3. **服务时间。** Running 的经验条件残余均值/P90、ignore_eos 余量及尾部回退仅决定完成时间，不再决定容量预留。Queued 每次首先定价准备与 forward，产生一个 token，随后才计剩余 decode；其 callback 分别接收 GPU/host 前缀，使用对应形状的 c1 准备项及实测 forward，不重叠叠加含队列的 TTFT。

物理容量释放仍是显式代理：将单一 `capacity − free − evictable` 按当前服务请求的逻辑 allocated 长度归一化分配，完成后转为可用。它不是树的逐节点解锁模拟；无法识别最后一个共享锁。候选的已命中 GPU 前缀在初始及虚拟完成后都要保护，避免将同一前缀同时算作可回收空间与免费复用。完成后保护量使用“原 evictable 加累计虚拟释放”的上限代理，而不是宣称已知哪个请求释放了候选路径。无可解释的完成事件、仍被未知预算占据的空间或超出冷需求容量时，预测可返回 unsupported。

Waiting 容量始终按完整 cold 输入估计；**最近一次 match 不抵扣容量**。服务均值情景可以使用 `last_native_match`，条件只是其输入形状等于当前 `input+generated` 且观测不晚于 quote；reason 明确写 `last_match_service_scenario_not_residency`。这不是新鲜度判定，也不承诺前缀仍驻留。另一情景用 cold forward 和经验 P90 服务长度，输出 `upper=max(mean,cold_p90)`；不是概率上界。缺失、未来时钟或输入形状不符的 saved match 使用明确 cold 情景。没有加入匹配调用、缓存树扫描、每轮 hash、额外 GPU 同步或持久预测状态。

这个模型复用原生 FULL 页预算规则，但仍是 **FIFO、冻结当前 ratio、完整请求预留的有限事件近似**。真实引擎会逐 pass 过滤/合并批次，未完成 chunk 暂不加入最终输出 headroom，后续 ratio 会变化，cache match/淘汰和速率也会变化；本模式没有复刻这些过程。候选受当前缓存布局影响，virtual queued admission 后布局也未被精确重建。因此适用范围限于现有 FCFS、FULL、单 GPU、不混合 chunk 的配置。`capacity_risk` 保留原 P90 物理需求诊断语义，不是新 native gate 的判定结果；旧/新方法比较时不能将这两个量混称。

本地验证包含直接执行固定原生 `PrefillBudget` 的差分测试，覆盖页大小、严格等号、原生初始预算和候选边界；另外验证 request ID 跨成员顺序关联、未匹配 finished 预算保留、ratio/ignore_eos 与服务残余分离、重入队列两种 headroom、旧 host 匹配仅影响服务情景、共享前缀不重复释放、同时完成、即时可容纳不要求无关时间支持，以及 config/artifact/schema/HTTP 往返。新观测只在 `workload_snapshot` 复制一次已有 rid，固定原生源码补丁可应用并编译。这些证据仅证明实现合同，尚不证明真实等待误差或性能改善。
