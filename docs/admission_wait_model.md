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

这些来源提供可继续扩展的方向：先用真实 pre-forward 残差检验“固定服务期速率”和 FIFO 近似，必要时按批次成员变化重估剩余服务；在冻结开发/测试边界后，比较经验残余、尾部截断分布和轻量候选条件长度模型；对新的内存等待误差单独校准单侧风险，再做等预算决策实验。若进一步评估 UniBoost/Nested WAIT/Sarathi 类调度，应单列原生 scheduler 实验和基线，避免把控制器选择 APPEND/REBUILD 的收益与引擎调度变化混在一起。
