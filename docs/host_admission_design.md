# C：业务会话与版本感知的 host 备份准入

2026-10-24。可选主流程已实现，默认关闭；固定源码应用、CPU 原生方法/action 顺序、HTTP 和恢复合同通过，尚未进行本扩展的 GPU 验收。当前补丁 SHA256 为 `0ed0295274411b9fe60415e7f48604937d84d3fc894eabd9e143b0560de7488f`。既有 epoch 退役结果不能替代本次新准入的验证。固定来源与获取摘要保存在 `runs/c-host-admission-review/sources/sources.json`。

## 复用依据与实际接点

EfficientAgent 固定 `b21b12a07a3ccf473c82acf00820b1aa2073044b`、Apache-2.0，其小型准入谓词结合压力与本请求新增完整 chunk 数。压力由其他活跃工作集的容量估计、较新的占用及窗口淘汰情况共同构成。本项目复用条件写入思想，不移植 vLLM/LMCache 接口、TP8 字节常量或背景线程。[固定作者实现](https://github.com/KunmingSHAO/efficientagent_release/blob/b21b12a07a3ccf473c82acf00820b1aa2073044b/efficientagent/kvtier/connector.py)、[许可与适配说明](third_party/EFFICIENTAGENT_NOTICE.md)。

固定 SGLang `3f36fc08331f64cd63c4fbbdf39cc402f38bb38b` 的 FULL 备份链为 `checkpoint_into_tree(req) → insert → _apply_cache_actions → BackupKV → _execute_and_commit_kv_backup → _execute_kv_backup → cache_controller.write`。`FullComponent.build_hicache_transfers(BACKUP_HOST)` 返回 `None`，主 KV 写入不经过额外 PoolTransfer。判断位于整条 BackupKV action 的 host 分配、commit、新锁、pending 和 DMA 之前；分裂节点、释放设备槽及其他 action 原顺序保留。[原生缓存实现](https://github.com/sgl-project/sglang/blob/3f36fc08331f64cd63c4fbbdf39cc402f38bb38b/python/sglang/srt/mem_cache/unified_radix_cache.py)、[FULL 组件](https://github.com/sgl-project/sglang/blob/3f36fc08331f64cd63c4fbbdf39cc402f38bb38b/python/sglang/srt/mem_cache/unified_cache/components/full.py)。

只过滤请求触发的 write_through/write_through_selective。write_back 淘汰路径保留原生；selective 仍先遵守原生命中次数门槛，研究策略不主动制造 BackupKV。决定以原生构造的完整祖先链为单位，避免备份缺口；共享 node 没有永久 skip 标记，后续新请求仍可补齐。研究 runtime 已限定 Python、dense FULL、TP/DP/PP=1、cache 模式，无 storage/linker/buffer/推测解码，不能外推到那些分支。

## 最终机制：每个 Req 的累计成功准入页预算

起始阈值 `8192 tokens` 对应 EfficientAgent 的 `8×1024`，在 page64 下为 128 页。这是初始适配，未证明最优。若只检查单个 action，native chunk8192 的常见路径难以超过阈值。因此最终实现是 **incremental request backup budget**，不同于上游一次整请求预先跳过，也不同于旧设计的每 action 独立额度。

对当前原生 Req，令 `B` 为此前已成功准入的 FULL 备份页，`N` 为本 action 的实际新页，`K=ceil(8192/page_size)`：

```text
pressure = fixed
        OR (other_business_working_set > actual_host_capacity
            AND actual_host_used >= 0.95 * actual_host_capacity
            AND allocation_evicted_host_tokens_in_last_60s > 0)
skip = enabled AND pressure AND N > 0 AND B + N > K
```

`conditioned` 使用公式的后三项，`fixed` 始终施加额度，`off` 不进入过滤。工作集为 `(最近60秒活跃业务数−1)×最近最多256个请求的平均prompt_tokens`；活动和 prompt 样本在请求**第一次原生 match** 记录，同 Req 再匹配/重排队不重复采样。这是逻辑工作集启发式，不是共享前缀去重后的物理占用证明。`host_hit_length` 只代表本次 host 恢复长度，未拿它冒充已备份前缀。

`N` 由本来要执行的 `build_backup_spec` 的 FULL device tensor 长度读取；每个 spec 构建一次并复用，不新增匹配、树遍历、GPU 同步或内容 hash。既有 host 副本的 native spec 长度为零。允许 action 后，只有每个节点的 host 分配、`commit_backup` 和 `_track_write_through_node` 成功，才将该节点的新页计入 Req 的 `B`；这表示 native 已提交备份，DMA 可能仍未完成。

- page64/chunk8192：首批128页可写；同 Req 的下一批4096 token/64页，在压力成立时整 action 跳过。
- 分配失败不扣页。祖先成功、后代分配失败只记成功祖先，并保留原生失败返回0及 pending 路径。
- 已备份 spec 为0时不重复扣。无压力时可以超过预算；先前已经准入的页仍被记录，以便后来压力出现时使用。
- 另一个 Req 有自己的额度，可沿原生祖先链补齐此前跳过的路径；若已经写出的页后来又被淘汰并实际重新备份，这是新的写入，会再次计量。
- 不取消已发 DMA，不在 ACK 前把页数称为“已完成传输”，也不保证跳过能带来净收益。

## 身份、退休和旧协议

`prepare` 可选传入既有业务 session ID，registry 保存业务→engine epoch 关系；同 epoch 不能重绑另一业务。正常执行和恢复共用 ACK 核验，sid/restart/operation/正 generation/通告能力下的 business ID 匹配后才保存 generation。恢复仍使用持久化 epoch 的 restart ID，不能以新引擎身份覆盖旧尝试。

首次 checkpoint 的 BackupKV 发生在 `register_session_ref` 之前。这里通过 `req.session_id` 查 prepare registry，因此不依赖尚未建立的树上会话引用。原生 `open_radix_session` 已创建 generation；完成后才注册 leaf 引用。ContextKV guard 禁止 `session_params`，因此原生 StreamingSession 的私有 KV 快捷 checkpoint 不会接管这些请求。[会话引用源码](https://github.com/sgl-project/sglang/blob/3f36fc08331f64cd63c4fbbdf39cc402f38bb38b/python/sglang/srt/mem_cache/unified_cache/session_ref_tracker.py)、[StreamingSession 源码](https://github.com/sgl-project/sglang/blob/3f36fc08331f64cd63c4fbbdf39cc402f38bb38b/python/sglang/srt/session/streaming_session.py)。

多 epoch 按业务去重，epoch 数另报。引擎 COMMITTED 早于 SQLite 提交，不用它推断业务事务；只有原退役协议完成才从未来工作集和相关 prompt 样本移除。RETIRING 且仍有 inflight 时保留活动；实际 host 槽继续计入占用，直到原生释放。

默认 `CONTEXTKV_HOST_ADMISSION=off` 时，旧 cache_config 字典/hash、info 和未传业务字段的 prepare wire 保持等价；metrics 可新增完整策略说明。仅启用时通告 `business_session_identity_supported`，并将模式/阈值纳入 cache identity。旧 wire、原生请求或 profiler 未提供业务身份时走原生写入，不猜测归属。控制器不会向未通告能力的旧引擎发送新字段。

## 反例审查与可观测边界

固定 scheduler 的 `stash_chunked_request → checkpoint_kv_cache → checkpoint_into_tree` 保留同一个 Req；insert 的主 action 和 finally drain 都携带该 Req，prompt 边界第二次 insert 也携带。这使跨 chunk 累计可执行，不只存在于孤立策略函数。native tree 的 inserted_len 避免同 Req 重复增加旧节点 hit_count，`build_backup_spec` 再去重已有备份。[chunk stash 源码](https://github.com/sgl-project/sglang/blob/3f36fc08331f64cd63c4fbbdf39cc402f38bb38b/python/sglang/srt/managers/scheduler.py)、[原生备份 action/spec](https://github.com/sgl-project/sglang/blob/3f36fc08331f64cd63c4fbbdf39cc402f38bb38b/python/sglang/srt/mem_cache/unified_cache/unified_tree_core.py)。

本次反查未发现上述范围的新阻断。CPU 测试覆盖128+64页、完整/部分分配失败、零页重入、新 Req 补齐、首次业务身份、多 epoch/退休、默认旧 hash/wire、write_back/未知身份放行、其他 action 顺序和错误 ACK 恢复。CPU fake tensor 仅验证真实提取方法的控制流，不能证明 DMA、host allocator、IPC 或真实压力行为。

`host_backup_admission` 保存实际 action 新页、Req/业务/epoch、压力分量、允许/跳过原因。metrics 分开报告 planned allowed、skipped 与 native-admitted 页；completed DMA bytes/duration 仍只来自原生 ACK。跳过 action 可能再次出现，累计 skipped tokens 不是唯一槽数或实测节省字节。淘汰窗口目前只观察 FULL backup 分配不足触发的 `evict_host`，不把 profiler 人为 pressure 的回收伪装成分配压力。事件使用服务端 monotonic；GPU kernel时间仅在原生ACK有CUDA事件时可用。

## GPU 第一阶段：off/fixed 通路 fixture

可执行入口 `runs/c-host-admission-review/run_fixture.py` 默认只输出计划；`--execute` 才启动本机独占服务。只占 GPU0/单端口，串行 off→fixed，每臂新进程，复用已有 server 启停与固定源码检查；finally 只停止自己创建的进程组。已有服务需操作者事先停止并在之后恢复，脚本不接管它。它没有成本 artifact、planner、数据集/qrels 或质量判分。

固定模型/源码，GPU16384、page64、chunked-prefill8192、write_through、host size1GB、kernel/layer_first/TMA0。用合法非特殊 token ID 构造相同12288-token输入，输出仅1 token并ignore_eos，顺序如下：

1. 专用 profiling cold reset，重新读取 restart；核实际 GPU/host 池、配置和业务能力。host 需足够容纳完整12288，否则此fixture失败，不改标签。
2. prepare一个业务 epoch；首次生成期间读取quote/metrics原始快照，保存完整result/receipt；记录目标Req的allow128与skip64事件。off没有准入事件，预期走原生完整备份。
3. settle后保存真实completed D2H、unique pool、quote；经已有 GPU pressure 回收设备副本，保留会话软引用。再次quote `输入+sentinel`：fixed预期host可恢复8192，off预期12288。
4. 同epoch另一个Req重新生成原输入，检查receipt恢复/重算，settle和GPU回收后host前缀应补齐12288；这验证共享node未永久禁止备份。
5. 完整结果后retire，检查ACK、终态以及retire本身不释放unique slots。无新的生成来验证过期guard；已有guard回归独立保留。

阈值触发必须由实际events和prefix验证，不能仅凭配置宣称命中。pending状态短暂，若采样未命中，报告coverage不足；不能把未观测到pending当零延迟或失败。允许→pending→completed路径由CPU顺序合同、原始quote及ACK联合审核；不要求一个action对应一个DMA ACK，原生可能合并。失败保留原始快照/结果并非零退出，不跳过失败臂或重试生成。两臂输出并非质量任务，ignore_eos fixture也不等价真实EOS工作负载。

## 第二阶段计划：conditioned 和容量

固定源码 `hicache_size` 为整数字段、十进制GB。当前模型每token KV为57344 bytes、page64；1GB源码预期17472 host tokens。size=0比例模式在GPU16384、ratio0.5下源码预期8256 host tokens，运行时必须读取pool，当前未在本扩展实测。[host pool 分配](https://github.com/sgl-project/sglang/blob/3f36fc08331f64cd63c4fbbdf39cc402f38bb38b/python/sglang/srt/mem_cache/pool_host/base.py)、[参数](https://github.com/sgl-project/sglang/blob/3f36fc08331f64cd63c4fbbdf39cc402f38bb38b/python/sglang/srt/arg_groups/fields/memory.py)。

conditioned不在首个driver执行。后续可在约8256 host池顺序生成不同业务的8192、8192-token路径，使原生备份分配实际淘汰并留下高占用；在60秒窗口内提交第三业务12288-token请求。必须先核近期eviction>0、占用≥.95、其他业务工作集>host，再检查该Req首128页允许/后64页跳过；如整节点淘汰或调度导致条件未成立，保留未覆盖结果，不人为填计数。再比较同一业务多epoch、退休旧epoch后工作集退出而物理占用不变，以及无压力允许超预算。

小host/启用策略改变cache identity，旧大host成本artifact不适用。首轮只有固定动作机制验收；之后实际完成时长、host恢复、额外重算与完成DMA一起报告，降低D2H不能独立证明改善。完整闭环和质量验证另做，不以实现完成代替效果验证。
