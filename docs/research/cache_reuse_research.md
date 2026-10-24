# 有限双层缓存的三条算法复用路线

核查日期：2026-10-24。本文补充[已有相关工作](related_work.md)和[贡献方向](../research_contributions.md)，只筛选 RLT、Continuum、KVFlow 三项；不重复 EfficientAgent、ContextPilot、SmoothAgent 等既有分析。本文没有迁移代码、运行新实验或依据正在生成的质量分数选参。论文结果、公开实现与本项目拟议适配分别说明，阈值或通用公式本身不作为贡献。

## 问题与接续目标

当前[conditioned host 准入](host_admission_design.md)按压力限制单 Req 的累计成功备份页。它尚未比较“保留哪条路径、保留多久、何时恢复”。[中容量回访](../archive/host_admission_validation.md)已经观察到保住另一业务前缀，同时推迟本业务尾段备份而增加其后重算。这是替换损失与重访收益的取舍，不能仅以 D2H 减少评价。

适配的共同目标是降低多会话完整响应成本，同时满足实际 GPU/host 容量与质量约束。可以研究“预期避免的重建、恢复和等待成本，减去新增传输及替换损失”，但这只是本项目的优化方向，不是下述论文已证明的双层最优性。资源按唯一物理页计量，祖先链、原生锁、在途 DMA 和退役语义仍限制合法动作；相同 EvidenceKey 不自动代表相同因果前缀。

| 方法 | 增加的能力 | 最小可复用部分 | 当前主要信息缺口 |
|---|---|---|---|
| RLT / KVRouting，ICLR 2026 | 不预测未来，用阶段标记和随机化抵抗 LRU 的访问顺序偏置 | 树叶 marking/淘汰算法及状态化对照 | 本项目页、节点和双层不同成本不满足原竞争分析的全部假设 |
| Continuum，2025–2026 | 按重访收益与占用机会成本决定保留时长 | TTL 有限候选求解原理；pin/unpin 生命周期参考 | 真实返回间隔、排队代价；作者公开代码未包含论文估计器 |
| KVFlow，2025；同仓后续 ScaleSim | 将已知调用需求传播到共享前缀，提前恢复 | 共享节点最小调用距离、恢复状态感知 | 自由问答没有预先已知的下一用户问题或完整调用图 |

## RLT：不依赖未来预测的树叶淘汰对照

**论文。** [ICLR 官方论文 §3–4、算法1](https://proceedings.iclr.cc/paper_files/paper/2026/file/8a1d8759bfaeaad3946798e65543c07c-Paper-Conference.pdf)将缓存对象建模为前缀树上的 token。RLT 标记阶段内访问过的 token，容量阶段结束后重置；缺页时从未标记的叶 token 中均匀随机选择。它的对数竞争界针对 token miss 模型，批处理分析另要求容量足以容纳在执行请求等条件。它不是按预期重算毫秒加权的双层策略；同论文的多 worker 学习路由 LBGR 也不应被当作本项目单 GPU 的现成功能。

**公开实现。** 作者仓固定 `fc3b2a16645273bd1d2e7e158fc4f4aeb94ac046`。入口为 [`radix_cache_lrt.py`](https://github.com/fzwark/KVRouting/blob/fc3b2a16645273bd1d2e7e158fc4f4aeb94ac046/sglang-0.4.6/python/sglang/srt/mem_cache/radix_cache_lrt.py) 的阶段标记和 `evict`，以及 [`cy_evict.pyx`](https://github.com/fzwark/KVRouting/blob/fc3b2a16645273bd1d2e7e158fc4f4aeb94ac046/sglang-0.4.6/python/sglang/srt/mem_cache/cy_evict.pyx) 的叶集合收集、逐 token 选择和尾部释放。它基于 SGLang 0.4.6，不是当前 UnifiedRadixCache 插件。每次收集会遍历树，随机删除循环的工作量随释放 token 数增长；需要实测控制开销，不能只比较 miss。

许可证据是该 SGLang 子目录的 [Apache-2.0 LICENSE](https://github.com/fzwark/KVRouting/blob/fc3b2a16645273bd1d2e7e158fc4f4aeb94ac046/sglang-0.4.6/LICENSE)。根目录未发现独立 LICENSE；本次不把根目录脚本或全部依赖一并认定为同一许可。若复制子目录代码，应保留许可、版权与修改说明。

**ContextKV 适配与反证。** 可先在 CPU 状态化回放中把原生可淘汰叶集合交给 LRU/RLT，作为不训练预测器的强对照；保留页对齐、祖先关系和锁资格。将 token 改为64-token页或整节点后不能直接引用原竞争比。RLT 可能丢掉高频、高重算成本的热门路径，随机化也不是已验证的平均时延优势。固定扫描、周期回访、热门共享前缀三种轨迹及总容量，预声明多个随机种子，报告 miss、重算代价与 CPU 开销；若热门负载恶化或额外开销抵消收益，应保留该负例。随机种子不是新增独立会话。

## Continuum：用机会成本选择保留期限

**论文。** [v7 §4–5](https://arxiv.org/html/2511.02230v7)在工具间歇期间保留 GPU KV，并联合程序级调度。其简化 TTL 目标为：

\[
\tau^*=\arg\max_{\tau}\;F_f(\tau)\,[\mathcal T\eta+T_{\mathrm{rebuild/reload}}]-\tau.
\]

`F_f` 是已完成工具调用间隔的经验 CDF，`T` 为历史等待估计，`eta` 刻画程序已完成进度与剩余工作的关系。有限候选取历史间隔及零。这个目标将重建、再排队的节省与占用机会成本放在同一时间单位，避免仅因缓存热就无限保留。饱和等待队列和程序进度假设属于模型条件，不能无条件套到自由问答。

**公开实现缺口。** 作者仓固定 `316a58794a6ff86b216e579b74fd56ed0c5a911f`，[许可 Apache-2.0](https://github.com/Hanchenli/vllm-continuum/blob/316a58794a6ff86b216e579b74fd56ed0c5a911f/LICENSE)。[README](https://github.com/Hanchenli/vllm-continuum/blob/316a58794a6ff86b216e579b74fd56ed0c5a911f/README.md)明确公开版本没有论文估计部分；[`ToolCallEstimator.set_up_pin`](https://github.com/Hanchenli/vllm-continuum/blob/316a58794a6ff86b216e579b74fd56ed0c5a911f/vllm/v1/core/estimate_with_func.py#L150)实际比较历史均值与固定阈值，然后返回零或该阈值。不能宣称调用这个类即复现了上式。

[`scheduler.py`](https://github.com/Hanchenli/vllm-continuum/blob/316a58794a6ff86b216e579b74fd56ed0c5a911f/vllm/v1/core/sched/scheduler.py#L237)提供 pin/unpin、到期检查和同程序等待状态的参考。但它依赖 vLLM 请求/块生命周期，不能直接代替 SGLang 软引用或业务提交后退役。重写短 TTL 求解器属于**论文公式适配**；生命周期代码迁移属于另一项工程工作。

**ContextKV 适配与反证。** 优先单独评估经验 TTL 求解器，不先引入新的强 pin。只使用已发生的记录；未返回或观察窗口尚未结束的会话也要保留其未覆盖/截尾状态，不能只统计成功回访而高估返回概率。不读取 MTRAG 后续问题、总轮数或当前尚未发生的完成长度。现 P2 紧接下一轮的客户端间隔不代表真实用户思考时长。固定轨迹分别使用短间隔、长尾间隔与低压力，比较原生、固定保留和经验 TTL；既报告被保留会话的节省，也报告其他会话被阻塞的等待。若间隔漂移或稀疏样本导致净成本增加，则否定当前估计假设，而非只调大 TTL。初次排序经验间隔需要 `O(n log n)`，给定有序候选后求最大值可线性扫描；这是估计目标内的最优候选，不是全系统最优保证，无需引入通用优化框架。

## KVFlow：将已知需求传播到共享前缀

**论文。** [KVFlow v1 §3](https://arxiv.org/html/2507.07400v1)用 Agent Step Graph 估计调用距离；AND/OR 依赖分别用 max/min 聚合。固定提示词对应节点向祖先传播需求，共享节点取最小距离；动态后缀优先淘汰。它还提前从 host 恢复即将使用的前缀，并让调度器区别已就绪与仍在传输的请求。该方法依赖工作流提供的调用信息，不等价于已经解决未知未来用户问题的预测。

**公开实现。** 固定 `7ef897eaedc16626a281ccfe0d453fae80251e8f`，[许可 Apache-2.0](https://github.com/PanZaifeng/KVFlow/blob/7ef897eaedc16626a281ccfe0d453fae80251e8f/LICENSE)。[`AgentManager.get_node_hold_priority`](https://github.com/PanZaifeng/KVFlow/blob/7ef897eaedc16626a281ccfe0d453fae80251e8f/python/sglang/srt/managers/agent_manager.py#L55)聚合节点相关 agent 的最小距离并按更新版本复用；[`hiradix_cache.py`](https://github.com/PanZaifeng/KVFlow/blob/7ef897eaedc16626a281ccfe0d453fae80251e8f/python/sglang/srt/mem_cache/hiradix_cache.py#L376)含祖先保护、空间检查、恢复提交和状态处理。当前作者仓还包括后续 ScaleSim/LoRA 扩展，见[固定 README](https://github.com/PanZaifeng/KVFlow/blob/7ef897eaedc16626a281ccfe0d453fae80251e8f/README.md)；这里定位可读模块，未把整个仓当作原论文逐字 artifact 或当前引擎的可直接替换件。

**ContextKV 适配与反证。** A+C 的最小因果信号是：当前 query 已到达、最终证据与布局已确定，从实际 token 前缀得知接下来需要哪段已驻留 KV。可复用共享需求的 min 聚合和恢复状态，不将 top50 文档候选视为50条可预取 KV，也不对尚未选中的 APPEND/REBUILD 同时预取后隐藏浪费。需求变动的成本随受影响节点/关联业务数增长；预取还消耗 GPU 槽和 PCIe 带宽，重叠是否有效必须实测。

固定最终输入、容量、提交顺序及 native 恢复路径，比较 reactive restore 与仅使用已确定当前需求的提前恢复。分别测共享前缀、低共享和传输竞争；保存无效预取、被挤出页、H2D 与端到端成本。若提前恢复仅把等待前移，或抢占其他请求使完整响应更差，则不能据恢复命中提升宣称成功。未知下一轮维持 unknown；不把数据集未来轮次包装成工作流元数据。论文“动态后缀优先淘汰”也不能直接迁移：本项目完整 QA 后缀可能在下一轮精确复用。

## 有限验证顺序与共同边界

1. **先解释资源取舍。** 固定实际 token 轨迹和池容量做状态化回放，比较原生与 RLT，显式计算保留/替换双方的重算和传输。不能沿用原运行的命中数给新策略重评分，因为策略已经改变后续缓存状态。
2. **再检验时间价值。** 用相同轨迹、预声明间隔验证 TTL，低压力和长尾都是必要负例；有真实排队才研究 pin/调度联合效应，不把 TTL 单独变化称完整 Continuum。
3. **最后接已知 A 需求。** 先保持相同材料与输入验证恢复时机，再允许独立的证据布局/选择实验。后者需完整质量与控制开销评价；缓存命中不能成为保留无关材料的理由。

三条路线不做全笛卡尔组合。统一模型、tokenizer、输出预算、页、写策略和实际容量；复用已有 quote/receipt/DMA/pool/retire 观测，计入全部失败、传输、重算、其他会话等待和新增 CPU 成本。硬件标定必须匹配新的 cache 配置。随机重复、工具间隔重放和同一开发会话变体分别报告，不能扩大独立样本数。保真缓存机制与可能改变回答的 A 分支分开验收，最终联合效果仍须闭环与冻结质量证据。

另核了 [Marconi](https://github.com/ruipeterpan/marconi/tree/08016617b1524e6bf6ac29b680641cc945bda7f0)、[AsymCache](https://arxiv.org/html/2606.02964v1)和[在线 compaction](https://arxiv.org/html/2608.00902v1)。Marconi 的价值/内存思想可参考，但其 hybrid recurrent 状态场景与[CC BY-NC 4.0 artifact 许可](https://github.com/ruipeterpan/marconi/blob/08016617b1524e6bf6ac29b680641cc945bda7f0/LICENSE.md)限制直接迁移。AsymCache 的非连续 attention kernel、仅 GPU 管理范围，以及 compaction 改变注意力和后续行为的性质，使它们不是本轮小模块复用首选；本次未核到后两项作者公开代码。它们仍可作为后续独立研究分支，而非被推断为无效方法。
