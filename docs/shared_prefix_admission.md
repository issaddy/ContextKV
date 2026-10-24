# 共享 GPU 前缀的即时准入估计

当前实现修正了一个可复现的容量误判：候选匹配的 GPU 前缀若已由运行请求硬锁，不能再次从可淘汰预算扣除。新增只读交集观测，仅供可选 `native_budget` 模型判断空等待队列下的即时容量；默认 `native_workload` 与原生调度不变。目前已完成源码级反例和 CPU 回归，真实 GPU 验证另行记录，不宣布整体加速。

## 问题来自缺失的共享关系

对固定 FULL/MHA 路径，令 F 为空闲 tokens、E 为可淘汰 tokens、D 为候选连续 GPU 命中、O 为原生运行请求输出预算、I 为临时 pin 导致的新增保护量。原生首 gate 包括：

\[
n-D+\min(\mathrm{max\_new},\mathrm{clip})+\mathrm{page}<F+E-O-I.
\]

分页分配另有 `ceil_page(n−D) ≤ min(current−I,total−I)` 检查，不能只看这一个不等式。`min(D,E)` 是 I 的静态上界；当 D 已被另一请求锁定，而 E 来自无关分支时，它会重复扣减。I 需要匹配路径与硬锁状态的交集，不能由两个全局长度恢复。业务 session 软引用也不等于原生 FULL 硬锁。

[固定原始 SGLang 源码反例](../runs/joint-cost-design/native-pin/shared-prefix-result.json)执行了真实 `Full.acquire_component_lock/release_component_lock`、`PrefillBudget` 和 `PrefillAdder` 函数，存储/分配设施用 CPU 小对象替代，未执行 GPU forward。旧 runtime/schema/admission 绑定到项目 `4e689eb`，重放记录及源码身份另存，避免用修正后代码冒充旧结果。

两个等长 X/Y 分支各4096 tokens；候选匹配 X，输入12288，输出预算512，page64。peer 输入6144、已生成1、预算剩511。两状态的旧 quote/workload 完全相同：F6080、E4096、D4096，唯一差别是 peer 锁 X 或 Y。

| peer 锁定 | 实际 I | gate 请求量 | pin 后剩余预算 | 原生接纳 | 旧模型 |
| --- | ---: | ---: | ---: | --- | --- |
| X，共享 | 0 | 8768 | 9665 | 接纳 | 误判需要等完成 |
| Y，无关 | 4096 | 8768 | 5569 | 拒绝 | 需要等完成 |

共享臂返回的 `OTHER` 是接纳后 chunk 预算耗尽，`can_run_list` 已包含候选；不能按返回枚举把它记为拒绝。8192输入的较小候选在两状态都接纳，作为非触发控制。测试给运行请求人为正服务时长，仅用于判定是否需完成事件，不将该数值称为硬件时延预测。

此前[264次受控调用](peer_interference_validation.md)的大APPEND拒绝样本中，旧代理恰好等于实际 I；本反例不推翻那些记录，也不把它们改称旧模型已经发生误判的证据。

## 实现范围

`CacheQuote.evictable_gpu_prefix_tokens` 记录**连续 ready GPU 匹配中，FULL `lock_ref==0` 的 matched tokens**。沿既有 `peek_full_prefix` 遍历累加，无第二次树遍历、内容哈希、拆节点或加锁。partial-node 只计匹配且按 page 对齐的片段；原生 split 会继承该片段的 lock 状态。host/pending 洞之后的 GPU 尾段不计入此字段。旧 wire 缺字段时为 `None`，与实测0区分。

只有以下范围使用该值：

- `native_budget`，候选为无 host load、无在途传输歧义的连续 GPU 匹配；其模型 resident 与该连续命中相同。
- 原生 waiting 列表为空；在任何虚拟完成/等待请求接纳之前检查即时 fits。
- 其它已有时序支持、容量、分页、配置和选择资格要求仍适用。这个容量观测不是已标定的等待模型。

有队列时，即使无需先等完成，虚拟接纳也可能改变锁与驻留，因此继续使用原模型。当前不把 I 沿每个未来 RID 精确推进：运行请求逻辑长度分摊的释放量仍只是代理，不能把 `min(D,I+proxy_release)` 称为真实共享释放的保守界。未立即 fit 的分支、host/mixed/pending 与旧引擎继续原路径；这项修正没有解决其全部等待误差。

报价只是观测，不预留资源。真实调用须记录 quote 至 candidate 接收/首 gate 的时间与状态变化；即使 quote-time 公式正确，也不保证未来 gate 相同。新增观测不改变实际原生 admission、锁、allocator、ACK 或生命周期协议。

## 验证与待检验效果

CPU 合同覆盖旧 wire/真实0、硬锁重数与软引用、部分节点、不跨 host 洞计数、quote 无副作用、真实原生 gate 的共享/非共享差别、host/pending 接线禁用、队列与完成后保持旧代理。部署前完成固定补丁正反应用和源码检查。

下一步使用相同16K池的等长X/Y预热、一个运行peer及一个候选，对同一实际 quote 重放旧/新即时分类并核对原生 gate，保留全部warmup、失败和状态漂移。它首先验证机制，不以固定输出合成输入宣称自然问答或planner收益。完整P2/P3仍需检验这种共享状态出现的频率、实际选择变化、控制开销及端到端收益；若只在人工构造中出现，贡献边界必须保留。
