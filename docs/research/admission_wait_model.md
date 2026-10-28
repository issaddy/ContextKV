# 原生工作量快照的等待估计

`controller/admission.py` 提供 `analytical_native_workload` 等待估计器，在原生状态快照上预测候选开始 forward 的时间；候选自身的 forward 和 decode 另计。`supported` 表示快照可用且时间回调处于测量支持域内。估计器为无状态计算，实际调度由SGLang执行，异质队列误差尚待实测。

输入包括候选长度、当前 GPU resident 数、预测输出均值/P90、报价时运行/等待请求的逻辑长度、开发集输出长度分布，以及独立 forward/decode 时间回调。运行请求已生成 `g` 个 token 时，对开发集 `L > g` 的样本计算 `L - g` 的均值和经验 P90，并受该请求的剩余声明预算约束。`ignore_eos`、超出开发样本尾部或没有样本时使用剩余声明预算，并在reason记录原因。

两个情景分别执行相同过程：

1. 以 `free_gpu + evictable_gpu` 为当前容量，先减去运行请求的预测剩余解码预留。负值表示该情景已有超订阅，并在 reason 标记。
2. 当前物理 locked 数为 `capacity - free - evictable`。按各运行请求的 `kv_allocated_tokens` 比例分配这个单一总数，作为完成后容量可用的代理，共享前缀计入同一物理总量。虚拟完成时归还该份额及先前扣除的解码预留。
3. 按当前 waiting 列表顺序作 FIFO 预测；queued/retracted 请求按 `input + generated` 的完整 cold 输入及剩余输出预留估计。实际执行顺序由原生调度器决定。
4. 容量不足时推进到下一个虚拟完成事件，同时完成的请求一起更新容量和并发数。每个虚拟请求按开始时的并发数调用实测时间回调，服务期内固定该速率。
5. 候选需求为 `input - resident + predicted_output + min(resident, evictable_gpu)`，最后一项避免同时把候选自己的 resident 前缀算作可回收空间。待该需求可容纳，即得到开始 forward 前的预测等待，再加由成本模型传入的该候选 idle 准备项。

forward 回调只定价 forward-entry 至首 token 的墙时；不使用含等待的 total TTFT。Queued 请求（包括重入队列的请求）及尚未生成首 token 的 running 请求，由 forward 产生下一个 token，随后只计剩余输出数减一的 decode。已经生成 token 的 running 请求，其全部预测残余都按 decode 定价。当前容量和所有预留已足够时，无需等待任何完成事件，也不要求无关运行请求的时间模型有支持。

Idle 准备项按候选 `(P,U,L)`——可复用前缀、新计算后缀、host 加载 token 数——在同配置的 `c=1` 实测 pre-forward 单元间插值，表示单请求准备与host恢复的联合成本。同一形状的不同输出预算按观测数合并。该准备项和 forward 的支持域由上层分别检查，在线队列等待由虚拟完成事件单独计算。

`upper_ms = max(mean_scenario, p90_scenario)` 取两种情景的较大值，差额表示启发式风险量。两者是输入假设下的点预测，尚无统计覆盖率。该近似固定请求速率并以总量分摊共享锁；原生优先级、页对齐、DMA、淘汰和retraction由引擎执行。缺少快照/容量、所需时间超支持域、单请求需求超容量，或locked容量无法由已知完成事件解释时，返回unsupported原因。DMA/partial restore的选择资格由上层成本模型处理。

CPU测试覆盖共享前缀的物理归一化、候选resident计量、运行future headroom、条件残余与尾部回退、queued first-token记账、同刻/零时完成、即时可接纳路径、旧/缺失摘要与无时间支持。真实等待残差、决策regret和系统收益通过引擎负载评价。

## 原理来源与复用边界

本模块结合下列建模思想与固定SGLang的可读状态实现等待估计。

| 原始工作 | 建模依据 | 本项目实现 |
|---|---|---|
| [TIE，arXiv:2604.00499v2，§3–4](https://arxiv.org/html/2604.00499v2#S3) | 输出长度分布及其尾部影响服务时间，`max_tokens`限定可服务长度 | 对`L>g`的经验样本计算残余均值/P90，构造两种服务情景 |
| [WAIT / Nested WAIT，arXiv:2504.11320v4，§2、§5](https://arxiv.org/html/2504.11320v4#S5) | 解码进度同时影响剩余工作与KV需求 | 按已生成长度更新服务残余，配对扣除和归还容量预留 |
| [Tail-Aware Scheduling / UniBoost，arXiv:2606.18431v1，§3](https://arxiv.org/html/2606.18431v1#S3) | prompt大小、已完成decode工作与KV压力共同影响尾部行为 | 使用报价时可见的长度、进度和容量，分别计算等待情景 |
| [Sarathi-Serve，OSDI ’24 官方论文与代码入口](https://www.usenix.org/conference/osdi24/presentation/agrawal) | Prefill与Decode的资源特性和同批干扰不同 | 分阶段调用本机实测时间模型，保留SGLang原生调度 |

## 已实现的只读观测扩展

研究补丁提供以下只读字段；当前交付成本配置使用 `native_budget`，较旧的 `strict` / `native_workload` 读取路径仍保留。旧 wire 缺字段时，schema 将其解码为 None，表示未知状态。`/cache/quote` 与 metrics 共用同一个只读 workload snapshot，HTTP client 复用一次共享 dataclass 解码。

| 位置 / 字段 | 真实来源与语义 |
|---|---|
| batch `native_new_token_ratio` | 报价时 `scheduler.new_token_ratio_tracker.current` 的实际值 |
| batch `native_max_new_tokens_clip` | 固定原生模块的 `CLIP_MAX_NEW_TOKENS` 实际值，包含启动环境覆盖 |
| batch `native_running_budget_requests` | 当前 `scheduler.running_batch.reqs` 的原顺序成员摘要，包含原列表内尚未清理的 finished/retracted 成员；不是旧字段的 overlap/last-batch/chunk 联合集合 |
| batch `native_running_budget_tokens` | 在上述当前成员与实际 ratio 上直接调用原生 `PrefillAdder._get_running_request_total_token_offset` 求和。保留原生“先减 generated，再 clip”、ignore_eos 与未过滤成员的行为；是当前 running headroom 的可复算快照，不是已执行 admission verdict 或完整 total/current offset |
| request `request_id` | 原生 `req.rid`，用于关联服务集合、预算集合及终态receipt；旧wire缺失时为None |
| request `native_finished` | 同次读取 `req.finished()`；旧引擎未知为None，预算列表保留原生成员 |
| request `last_native_match` | 最近一次被研究observer捕获的原生`match_kv_cache`完成结果；未捕获为None |

`last_native_match`保存五个标量：`observed_time_ns`、本次参与匹配的`input_tokens`、`device_prefix_tokens`、`host_hit_tokens`、`matched_prefix_tokens`。启用研究runtime后，Hook在原生match和请求字段更新完成时复制结果及`time.monotonic_ns()`，每请求保留最近一份记录。

`last_native_match.observed_time_ns`与batch `observed_time_ns`使用同一scheduler进程的monotonic纳秒，相减得到观测年龄。Saved match保留匹配时的字段副本；后续淘汰、加载或重入队列可能改变实际驻留。候选报价以peek读取当前状态，实际准入时再由原生路径匹配。

Running预算和服务工作量摘要使用不同成员集合，各自读取所需长度标量。Observer每次原生match的额外工作为O(1)，workload导出为O(active+waiting)。报价时的running headroom描述当前成员；实际prefill pass还会更新批次，并计入mixed-decode、已准入prefill及页开销。

本地验证复用固定原生offset、ratio tracker和match源码片段，检查成员差异、ratio衰减、remaining-then-clip、ignore_eos、原生返回值、匹配记录、旧wire与HTTP数据往返，并确认新增观测后旧等待估计逐值不变。完整补丁已在干净固定源码上应用并静态编译。


## 当前成本配置：native_budget 预留与服务分账

当前交付配置使用`cost_model.admission_estimator: native_budget`，配合同配置的`forward_first` stage artifact。结果记录`waiting_estimate_kind=analytical_native_budget`、`calibrated=false`；`selection_eligible`表示所需观测和阶段支持域满足。artifact保存该选项，旧wire与旧artifact由原模式读取。缺少request identity、原生比例/预算、容量或实际page_size时返回unsupported。

实现分开维护三类量：

1. **total 容量账本。** 初始值为 `free + evictable − native_running_budget_tokens`。各服务请求完成时，只归还相同原生 `request_id` 对应的预算份额；其公式为 `min(max_new_tokens − generated_tokens, clip) × ratio`，ignore_eos 使用 1。预算列表内不属于服务集合的 finished/retracted 成员继续占着该份额，并记录 `unmatched_native_budget_held`。不在预算列表中的 Overlap 服务成员完成时，归还的原生预算份额为零。
2. **current 容量账本。** 初始值为 `free + evictable`，不扣 running 输出 headroom。Queued 虚拟完整 prefill 的 current 需求为页取整后的完整输入加一页，total 另加完整 clipped 输出预算。候选 gate 用未驻留输入、声明 `max_new_tokens`（受实际 clip 限制）与一页开销，并保留原生严格 `< total` 及页分配 `<= min(current,total)` 边界。对重入 waiting 的请求，gate 使用剩余预算，但完整 prefill commit 使用原生完整预算，两者没有混同。
3. **服务时间。** Running 的经验条件残余均值/P90、ignore_eos 余量及尾部回退仅决定完成时间，不再决定容量预留。Queued 每次首先定价准备与 forward，产生一个 token，随后才计剩余 decode；其 callback 分别接收 GPU/host 前缀，使用对应形状的 c1 准备项及实测 forward，不重叠叠加含队列的 TTFT。

物理容量释放按总量近似：将`capacity − free − evictable`按服务请求的逻辑allocated长度归一化分配，完成后转为可用。该分摊缺少逐节点共享锁信息，因此在初始和虚拟完成后都保留候选已命中GPU前缀的保护量；完成后以“原evictable加累计虚拟释放”为上限估计保护量。无可解释的完成事件、空间仍被未知预算占据或超出冷需求容量时，返回unsupported。

Waiting 容量始终按完整 cold 输入估计；**最近一次 match 不抵扣容量**。服务均值情景可以使用 `last_native_match`，条件是其输入形状等于当前 `input+generated` 且观测不晚于 quote；reason 记录为 `last_match_service_scenario_not_residency`。该记录用于构造服务时间情景，实际驻留以原生当前状态为准。另一情景用 cold forward 和经验 P90 服务长度，输出两种情景的较大值 `upper=max(mean,cold_p90)`。缺失、未来时钟或输入形状不符的 saved match 使用 cold 情景。计算复用现有快照。

模型复用原生FULL页预算规则，采用FIFO、固定当前ratio、完整请求预留的有限事件近似，适用于现有FCFS、FULL、单GPU、不混合chunk配置。实际引擎逐pass更新批次、chunk进度、ratio及缓存布局，这些动态变化构成预测误差来源。`capacity_risk`表示P90物理需求风险，native gate单独表示预算准入结果。

本地差分测试直接执行固定原生`PrefillBudget`，覆盖页大小、严格等号、初始预算与候选边界。其他测试覆盖request ID关联、未匹配预算保留、容量与服务分账、重入队列、历史match、共享前缀、同时完成、即时可容纳路径及config/artifact/schema/HTTP往返。真实等待误差和系统性能仍待负载验证。
