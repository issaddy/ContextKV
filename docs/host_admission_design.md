# C：业务会话与版本感知的 host 备份准入设计

2026-10-24。此处是固定源码审查和可执行实现设计，尚未接入引擎、采集新 host 压力数据或证明收益。现有 epoch 退役机制继续使用；本扩展针对备份后很快被淘汰、未来未能恢复的流量。固定源码与获取摘要保存在 `runs/c-host-admission-review/sources/sources.json`。

## 复用依据与实际接点

EfficientAgent 固定 `b21b12a07a3ccf473c82acf00820b1aa2073044b`、Apache-2.0，其小型准入谓词结合压力与本请求新增完整 chunk 数。压力由其他活跃工作集的容量估计、较新的占用及窗口淘汰情况共同构成。这里复用条件写入的机制，按 SGLang 的实际 page 与池容量计量；原系统的 vLLM/LMCache 接口、TP8 字节常量和 1024-token chunk 不直接移植。[固定作者实现](https://github.com/KunmingSHAO/efficientagent_release/blob/b21b12a07a3ccf473c82acf00820b1aa2073044b/efficientagent/kvtier/connector.py)。

固定 SGLang FULL 备份链为 `checkpoint_into_tree(req) → insert → _apply_cache_actions → BackupKV → _execute_and_commit_kv_backup → _execute_kv_backup → cache_controller.write`。`FullComponent.build_hicache_transfers(BACKUP_HOST)` 返回 `None`，不生成额外 PoolTransfer，修改那里不能控制实际主 KV 写入。准入判断应位于执行整条 `BackupKV` action 之前；分裂节点、释放设备槽及其他 action 仍按原顺序执行。[原生缓存实现](https://github.com/sgl-project/sglang/blob/3f36fc08331f64cd63c4fbbdf39cc402f38bb38b/python/sglang/srt/mem_cache/unified_radix_cache.py)、[FULL 组件](https://github.com/sgl-project/sglang/blob/3f36fc08331f64cd63c4fbbdf39cc402f38bb38b/python/sglang/srt/mem_cache/unified_cache/components/full.py)。

先限定请求触发的 write_through/write_through_selective 备份。write_back 淘汰路径没有当前请求，保留为原生独立对照。决定应针对本次 action 的完整备份祖先链，避免留下不连通的 host 前缀；以后允许备份的请求仍可由原生构造器补齐祖先。共享节点不持久标记为“禁止备份”。已启动 DMA 的分配、锁、完成事件和 ACK 继续由原引擎处理。

## 本项目需要增加的语义

1. **业务会话与 engine epoch 分开。** 当前 prepare 只有 engine session/generation；多个 REBUILD 不能算成多个活跃业务任务。应透传已有业务 session ID，并在原 registry 中关联 generation，复用已有身份而非增加 UUID/hash。
2. **退休与占用分开。** 引擎的 COMMITTED 表示生成完成，早于控制器 SQLite 提交；不能把它当作业务提交事件。只有原退役协议确认后的版本退出未来工作集估计，尚存在的 host 槽仍计入实际压力。
3. **新增备份量按真实 action 计。** `host_hit_length` 只表示本次缺失设备副本而需 host 恢复的长度，不能当作全部已备份 host 前缀。首版可使用原生 `backup_spec` 中真实待写页数，形成与 EfficientAgent 匹配但不同的动作准入策略；已有备份节点的原生去重继续复用。若要严格使用首次匹配的完整 host 覆盖，需要在已有遍历中观测，不额外建立缓存树。
4. **判据与收益可审计。** 每次记录请求/业务版本、实际待备份页数、占用与淘汰窗口、允许/跳过原因。计数从实际 action 和完成事件获得，不能将“计划跳过”计成已经节省的 DMA 字节。

## 最小验证与容量

CPU 合同先覆盖同一业务多个 epoch、退休但仍占槽、共享祖先链、以后允许补齐、无研究身份请求、无压力时原行为、跳过备份但保留其他 action、重复 checkpoint 以及生成失败。配合固定源码差分检查 action 顺序和原生锁/ACK 路径。默认关闭，并为来源/算法适配保留许可归属。

真实机制对照先固定 APPEND/REBUILD 轨迹，比较原生写策略、只看引擎 session 的工作集准入、业务 session/epoch 去重的准入。实际完成时长、host 恢复、额外重算与完成 DMA 一起报告；降低 D2H 不能独立证明改善。随后才进入自身答案历史的完整闭环与质量检查。

固定源码 `hicache_size` 为整数字段、单位十进制 GB。当前模型每 token KV 为 57344 bytes，page64；显式 1 GB 会向页对齐得到 17472 host tokens。若需更小容量，可使用 size=0 的比例模式，在 16384 GPU tokens 下设置 ratio=0.5，源码计算为 8256 host tokens。两者仅为源码预期，运行时必须读取真实 pool；当前尚未实测这两个点。16 GB 下的旧成本 artifact 不因此适用于新 host 配置，机制对照先使用固定动作。[host pool 分配](https://github.com/sgl-project/sglang/blob/3f36fc08331f64cd63c4fbbdf39cc402f38bb38b/python/sglang/srt/mem_cache/pool_host/base.py)、[HiCache 参数](https://github.com/sgl-project/sglang/blob/3f36fc08331f64cd63c4fbbdf39cc402f38bb38b/python/sglang/srt/arg_groups/fields/memory.py)。

该扩展的贡献候选是业务语义、版本存活与实际两层工作集共同约束备份的作用效果；原生缓存写回、阈值判断和会话 ID 本身均属于复用基础。与[其余 A/C 扩展](ac_extension_priorities.md)并行推进，不以方法设计完成代替效果验证。
