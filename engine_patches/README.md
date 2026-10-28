# SGLang ContextKV 协议补丁

基准为 SGLang `3f36fc08331f64cd63c4fbbdf39cc402f38bb38b`。`0001-contextkv-protocol.patch` 是基于该提交真实源码生成的可应用补丁，覆盖 15 个源文件。`source_manifest.json` 保存审查文件 SHA256；`contextkv_runtime.py` 是补丁新增模块的同内容镜像，供 CPU 合同测试。准备脚本校验二者一致。

已在 macOS 完成固定源码独立下载、原文件 hash 验证、`git apply --check`、应用和 Python 语法编译。协议与树视图有 CPU 单元测试。`verification_report.json` 只记录源码验证，不汇总 GPU 实测；当前 H200/Qwen3 联合运行及效果见[数据与评价](../docs/data_and_evaluation.md)。

## 实现范围

- `/context/engine/info`、`/cache/quote`、`/context/epoch/prepare`、`/context/epoch/retire`、`/context/epoch/status` 全部经 tokenizer communicator 进入拥有缓存的 scheduler；HTTP 线程不读取可变树。
- 批量报价使用同一个观察周期，对完整最终 token IDs 共用原生 `Req._compute_max_prefix_len` 与页对齐。native radix key 使用 `array('q')`，与真实树一致。
- `UnifiedTreeCore.peek_full_prefix` 只遍历匹配路径，返回连续 GPU ready、GPU/host recoverable、需新加载的 host-only、H2D pending、D2H pending。节点中间命中不 split；不触碰 LRU、引用、分配器、传输队列或 CUDA fence。
- H2D 的 `contextkv_h2d_pending_id` 是新增观测字段，对所有原生写策略在提交加载时标记，节点 split 继承、原生加载 ACK 后清除。它不参与原生 admission、锁、淘汰决策。D2H 使用已有 `write_through_pending_id`；已发布地址但未完成传输的副本不计 ready。
- NATIVE 加载预测参考原生 `load_back_threshold`；旧 `native_estimate_uncertain` 保留 pending 或瞬时 free GPU slot 不足的原语义，同时 `uncertainty_reasons` 区分 `inflight_transfer` 与 `capacity_pressure`。后者表示可能需要原生淘汰或等待，不表示缓存信息不可读取。报价不是预留，实际匹配、加载、排队与预算决定仍由原生路径执行。
- 活动 epoch 的 prepare 对同一 operation 幂等，不重开 native generation；生成在原生 `ensure_session_generation` 前同时验证 restart、registry 状态和当前 tracker generation。保留期内的关闭 ID 继续由原生关闭记录阻止重开。
- retire 先阻止新请求，等待已接纳请求终态，再撤销 native 软引用并 ACK。软引用退役的 free 计数恒为零；缓存物理占用不会被假定已回收。原生已接纳请求在 RETIRING 期间可能先完成注册、再立即被退役撤销，因此 receipt 的 `session_ref_registered` 表示本次曾成功经过注册路径，不是 ACK 后仍持有引用的证明。
- 终态 receipt 通过原生 customized-info 输出，包含真实 tracker 注册结果、cache inserted length 和 native 分层命中。实际 prefill 工作来自每次成功 worker forward 的 `extend_lens` 累计，覆盖 chunk/retraction 重算并排除 mixed batch 的 decode 行。

当前运行合同是显式 Qwen3 LocalModelScope、单 GPU、TP/DP/PP=1、Python TreeCore、dense FULL、单条累计 SSE 流式或非流式请求、无动态 LoRA、默认 cache key。LoRA可先离线导出为独立dense checkpoint，再用新scope走同一合同。旧Qwen2.5-7B下载/启动/runtime fallback已删除，不再支持该模型的新运行；历史源码/GPU记录只作已有机制证据，不能作为新模型校准。HiCache 的 cache 模式受支持，buffer-only、L3、外部 linker、SWA/Mamba、speculative、PD/PDMux 不在此补丁范围。HTTP client 必须先验证研究能力，绝不把 guard 字段直接提交给原生未补丁服务器。

## 准备与启动

仅验证源码，不安装 GPU 依赖：

```bash
python scripts/prepare_sglang.py --source-only \
  --checkout vendor/sglang-source-check \
  --report runs/sglang-source-verification.json
```

Linux/CUDA 机器准备完整固定 checkout，并在独立环境安装该 checkout 的依赖：

```bash
python scripts/prepare_sglang.py --checkout vendor/sglang
SGLANG_BUILD_RUST_EXTS=none python -m pip install -e vendor/sglang/python
```

Python TreeCore 运行合同可使用固定 [`python/setup.py`](https://github.com/sgl-project/sglang/blob/3f36fc08331f64cd63c4fbbdf39cc402f38bb38b/python/setup.py) 支持的 `SGLANG_BUILD_RUST_EXTS=none` 跳过 Rust 扩展编译，无需为本课题构建 Rust TreeCore。

固定源码的 `python/pyproject.toml` 实际声明 `torch==2.14.1`、`transformers==5.19.0`、`tokenizers==0.23.2`；文件原 SHA256 `8a176848d601510b3557ece2cf9e37d3fd449f5eca2d469cf34d7078b103bc0c` 纳入 `source_manifest.json`；依赖的来源是同一固定提交的 `python/pyproject.toml`。已删旧模型专用 `model_manifest.json`，模型配置/库存身份直接复用 `configs/model_scopes/` 的显式本地记录，不维护第二套清单。这是一份明确依赖来源的安装入口，未在当前 macOS 环境验证整套 CUDA 构建。正式运行应冻结 Linux 环境包清单、驱动和编译器版本。

使用用户已经提供的 Qwen3 权重，不下载任何生成模型。先声明一个已有 scope；例如8B配置指向 `/media/ssd1/qwen3-8b`，32B配置指向 `/media/ssd1/qwen3-32b`：

```bash
export CONTEXTKV_MODEL_SCOPE_PATH=configs/model_scopes/qwen3-8b.local.json
scripts/bootstrap_sglang.sh
source .venv-sglang/bin/activate
export CONTEXTKV_FINGERPRINT_JSON="$(cat runs/fingerprint.json)"
# 资源/兼容路径显式选择；例值不是最佳参数或显存可用保证。
CONTEXTKV_HICACHE_GB=8 CONTEXTKV_HICACHE_LAYOUT=layer_first \
SGLANG_HICACHE_TMA_TRANSFER=0 scripts/launch_sglang.sh \
  --host 127.0.0.1 --port 30000 --max-total-tokens 16384 --mem-fraction-static .8
```

`bootstrap_sglang.sh`仅安装固定SGLang/项目依赖，然后通过共享`ModelConfig/create_model_template`验证已存在的scope快照、tokenizer和模板，保存指纹。它不调用snapshot_download，不创建旧模型目录；缺少scope或本地文件会明确失败。源码准备下载SGLang及依赖，模型使用已有本地文件。

`launch_sglang.sh`必须设置`CONTEXTKV_MODEL_SCOPE_PATH`；仅有`CONTEXTKV_MODEL_PATH`不能启动。scope拥有实际model/tokenizer路径，HF loading revision为None，不接受尾部`--revision`或另一个model/tokenizer路径。可选`CONTEXTKV_MODEL_PATH`仅作为同scope路径一致性检查，不能作为fallback。scope按模型元数据标识本地快照，验证范围由`weight_content_verified`记录。

schema v1 `local_snapshot_metadata`保持原有config/分片库存身份和全部旧指纹。离线`python -m training.export`新增schema v2 `local_checkpoint_artifact`，将完成的LoRA产物合并到新的独立BF16 dense目录，按每个weight文件的`file/bytes/sha256`建立`local-checkpoint:`身份；同大小不同内容可区分，目录位置和来源注释不进入身份。发布后应保持目录不可变；控制器`model.scope_path`和引擎`CONTEXTKV_MODEL_SCOPE_PATH`指向同一新scope，重新生成fingerprint并使用独立state/engine，不继承基座KV和成本标定。stock/native基线复用该身份接口，无需动态LoRA缓存路径。完整命令见[训练与导出](../docs/research/grounded_training.md)。

v2的权重摘要仅在导出发布边界全量计算一次。启动时核对config、实际文件名/大小与index；单文件`model.safetensors`无需index，只读取header长度及JSON offsets核对tensor字节数，不读或哈希tensor body。`weight_content_verified=false`明确当前内容未被启动时全量复验，发布摘要不能监控后续同尺寸文件改写。PEFT `safe_merge=True`只检查有限值，BF16舍入和实际回答效果另行验证。

通过 `CONTEXTKV_ENABLE_HICACHE=0` 可运行GPU-only功能检查；`CONTEXTKV_HICACHE_GB`/`CONTEXTKV_PAGE_SIZE`设置容量与页大小，`CONTEXTKV_WRITE_POLICY`复用原生三写策略。scheduler的`get_runtime`首次初始化必须读取scope，验证实际模型设置、config/库存、tokenizer/template与fingerprint；缺scope返回`MODEL_SCOPE_REQUIRED`，不存在旧revision/config隐式支持。初始化后复用实例，不逐请求扫描/重新解析。纯CPU `ContextKVRuntime`协议容器使用无模型fixture测试缓存与事务机制。

当前scope仅支持已明确的dense Qwen3、BF16、无SWA/RoPE扩展；32B的head_dim使用显式128。缓存、事务、只读观测、DMA和原生admission逻辑保持，未为退役旧模型改动这些通用机制。

## 观测与恢复合同

`/context/metrics` 经相同 scheduler 控制面读取原生 allocator 的 `capacity − free`，返回 GPU/host 唯一物理占用和当前 running/waiting 请求数。它不扫描树，不按 session 引用重复计数。事件环最多保留 4096 条记录，`after_seq` 增量拉取，并通过 `events_truncated/dropped_before_seq` 显式暴露丢失区间。

报价还返回 `gpu_capacity_tokens` 和原生 `full_evictable_size()` 的 `evictable_gpu_tokens`；两者都是 O(1) 计数。GPU 可淘汰量是 FULL device 未硬锁 token 总数，包含可能被当前候选命中的共享前缀。不能把它与候选 ready prefix 重复计算成 admission 保证。原生 [PrefillBudget](https://github.com/sgl-project/sglang/blob/3f36fc08331f64cd63c4fbbdf39cc402f38bb38b/python/sglang/srt/mem_cache/prefill_budget.py#L98-L113) 使用 free + evictable；[PrefillAdder](https://github.com/sgl-project/sglang/blob/3f36fc08331f64cd63c4fbbdf39cc402f38bb38b/python/sglang/srt/managers/schedule_policy.py#L1350-L1407) 再保护候选前缀并考虑已运行请求的剩余输出预算、页开销、chunk 和其他约束。这个补丁不模拟或修改它。Host 没有对应的精确 O(1) token 总计；`evictable_host_tokens=null` 明确未知，既不报假零，也不为报价扫描所有 host 节点。

报价与 metrics 的 `native_running_requests` / `native_waiting_requests` 只传出每个请求的 `input_tokens`、`generated_tokens`、`max_new_tokens`、`ignore_eos`、`kv_allocated_tokens`、`kv_committed_tokens` 和 `cache_protected_tokens`，没有文本或 token IDs；新增可选 `request_id` 直接复制既有原生 `req.rid`，供服务集合、预算集合和终态 receipt 精确关联，不创建新标识或内容 hash。输出预算取 native 已按上下文和池容量裁剪后的 `sampling_params.max_new_tokens`，已生成数只取 scheduler 当前已处理的 CPU 状态；不读取未来输出。[ReqKvInfo](https://github.com/sgl-project/sglang/blob/3f36fc08331f64cd63c4fbbdf39cc402f38bb38b/python/sglang/srt/managers/schedule_batch.py#L974-L987) 的三种 KV 数字是**逻辑序列边界**：树拥有 `[0, cache_protected_tokens)`，请求自有尾部位于其后；共享前缀、页边界和完成后的缓存插入都使其不能直接作为唯一物理占用或即将释放 slot 数。Overlap 时 committed 状态还可能落后已执行 forward。

运行集合复用 native `collect_inflight_reqs()` 对 running/last/overlap batch 去重，纳入独立 chunked request，排除已结束请求；已回到 waiting 的请求只计 waiting。Running 摘要无顺序语义；waiting 保留当前列表顺序，但原生优先级、staging 和 admission 可改变实际执行顺序。读取成本为 O(active + waiting)，每个请求只读标量和数组长度，不遍历 token 内容、不维护新缓存、不处理 ACK、不改变锁和 LRU。没有 scheduler 时摘要为 `null`，已知空集合为 `[]`。这些观测可用于区分容量压力和传输未完成，尚不构成经过 GPU 验证的新等待预测模型。

- `request_terminal`：receipt 与原生完成时序。
- `transfer`：在原生 HiCache ACK 已完成后复用 `num_bytes`、`start_event.elapsed_time(finish_event)`，输出 H2D/D2H 字节与 CUDA event 时间。计时不支持时为 `null`，不能将 CPU enqueue 时间替代它。该观察点不增加 CUDA 同步，不改变传输调度。
- `retire_ref`：软引用退役，free 为零。
- `pool_snapshot`：仅 `CONTEXTKV_SAMPLE_POOL=1` 开启；每个 scheduler 边界用 O(1) allocator 计数采样，只有占用变化时写事件。用于计算 scheduler 边界的**采样峰值**与占用积分。

生成结果中的 `server_ttft_ms` 为 scheduler 收到请求至原生首 token 完成，`server_elapsed_ms` 为收到至终态，`decode_ms` 为首 token 至终态，`tpot_ms` 为该段除以后续输出 token 数。`forward_to_first_token_ms` 单独表示原生 forward-entry 至首 token 完成。全部来自同一服务端 monotonic 时钟；原生 forward-entry 在部分 cache 准备之后，不能拿它精确拆出排队与恢复。

`elapsed_ms` 是客户端提交至完整 HTTP 响应。默认累计 SSE 客户端通过首次非空 output_ids 和最后新增 token 的帧观察 `ttft_ms/last_token_ms`，重复累计帧不更新时刻。客户端 TPOT 由首末 chunk 间隔除以 N−1 计算，报告明确标为 chunk 观察；服务端 `tpot_ms` 仍是另一时间域的原生指标。可显式 `SGLangEngineClient(..., stream=False)` 使用非流式，此时两个客户端 token 时刻未知；专用 profiler 保持非流式。`queue_ms/prefill_ms/restore_prefill_ms` 保持未知；当前记录提供联合阶段墙时。成本标定使用 `phase=first_token, timing_scope=server_request` 联合阶段，在同一并发压力桶中比较，避免重复加排队或 H2D。

完整最终文本和 token IDs 经过原生 detokenizer，终态响应之前通过一次 IPC 发布回 scheduler。guarded SSE 统一为累计输出，兼容原生 incremental streaming 开关。最后新增 token 先以不带 receipt/finish 标志的帧发送，再执行发布 IPC，最后发送重复累计 token 的完整终态 receipt 帧，避免将发布延迟算入最后 token 到达时刻。客户端只有收到终态 receipt 和 `[DONE]` 才返回结果；断帧、部分断连、缺少终态均走未知结果恢复，不能提交部分答案。`status` 在默认 900 秒 TTL、64 MiB 总结果预算、1024 个 attempt receipt 的范围内返回原样结果；通过 `CONTEXTKV_RESULT_TTL_SECONDS` 和 `CONTEXTKV_RESULT_MAX_BYTES` 调整。超限/过期、终态后发布前的进程崩溃可能只剩 receipt，此时按恢复协议处理；UNKNOWN 表示执行结果未知。

研究层的退役 epoch 记录和终态操作回执各复用 `receipt_limit` 上限，默认1024；活动及仍在退役中的记录保留到完成。原生[会话追踪器](https://github.com/sgl-project/sglang/blob/3f36fc08331f64cd63c4fbbdf39cc402f38bb38b/python/sglang/srt/mem_cache/unified_cache/session_ref_tracker.py)另保留最近8192个关闭 ID。两处记录都过期后，显式 prepare 可以获得新的原生 generation，旧 generation 的请求仍被拒绝。迟到的退役重试在确认原生引用已不存在、或当前 incarnation 更新时可确认旧引用已解除，不能释放新引用，也不证明旧生成已完成。`control_metadata` 指标分别报告活动和保留的终态记录数量。

原生 `/flush_cache` 会重建 generation 表并从头计数。`get_runtime` 以该表的对象身份检测重置，先更换研究 restart 标识和控制状态，再处理请求；不扫描缓存树或权重。专用 profiler 冷重置复用同一替换逻辑。这样旧请求即使遇到相同数值的 generation，也会被旧 restart 标识拒绝。

## 可执行的真机验收与标定入口

工程验收先通过 CPU 合同和源码检查，再在已授权的 H200 环境运行以下入口。协议冒烟与小规模标定的通过，不代替正式基线、质量和容量实验。

先在正常研究服务器验证协议，不 flush 共享缓存：

```bash
python -m engine_patches.smoke --config configs/qwen3_8b_scope.example.yaml \
  --report runs/sglang-smoke.json
```

该入口验证指纹与固定提交、prepare 幂等、真实生成 receipt/服务端时序、原样 status 结果恢复、重复只读报价、退役 ACK 及旧 epoch 拒绝。`smoke` 是功能验收，不是性能或质量结论。

成本标定需要另起**专用**引擎实例（端口和相应 config 的 `base_url` 一致）：

```bash
CONTEXTKV_ENABLE_PROFILING=1 CONTEXTKV_SAMPLE_POOL=1 \
  scripts/launch_sglang.sh --host 127.0.0.1 --port 30001
python -m profiling.sglang_probe --config configs/qwen3_8b_scope.example.yaml --base-url http://127.0.0.1:30001 \
  --output runs/calibration.jsonl --cases-file runs/selected-cost-cases.json \
  --warmup 1 --repetitions 3
```

配置中的 `model.scope_path` 与服务端 `CONTEXTKV_MODEL_SCOPE_PATH`必须指向同一显式Qwen3身份；客户端本地tokenizer与引擎必须一致。不要通过`--model-path`绕过scope或重用旧模型的成本产物。已有成本范围直接复用；只有实际缺口才选择必要测点。

`/context/profiling/prepare` 默认禁用，并仅在空闲专用实例接受操作。cold 调用原生 cache flush 并更新 restart identity；gpu_ready 由普通原生生成暖前缀并 settle 写入；host_only 在暖前缀后经原生 write ACK、软引用退役和 eviction 构造。每轮测量前后均在专用实例 settle 写 ACK，前游标排除暖缓存传输，后游标包含本轮生成写回；它不进入 server request 阶段耗时。客户端随后用只读报价核实**实际**前缀和页边界，容量/写策略不满足时停止，不能用标签伪称“冷/暖/host”。GPU-only 服务须省略 host_only。write_through_selective 若未保存 host 前缀也会显式失败，不能误记为 host 测量。

每个 cache/shape/concurrency case 默认先运行 `--warmup 1`：执行相同 prepare/probe，但只在 telemetry 标记 `warmup=true`，不写入 Measurement 训练行。随后正式 repetitions 标记 `warmup=false`；manifest 记录 warmup 数量。这样避免首执行 JIT/初始化开销直接污染成本表，仍保留全部暖机证据。

输出包括原始 Measurement JSONL、同目录 `.telemetry.jsonl`（逐请求 receipt/timing 与 pool/DMA 事件）和 `.manifest.json`（指纹、配置、实际 pool、seed、case、完成状态）。专用合成标定请求设置 `ignore_eos=True` 以覆盖指定输出长度；所有记录仍采用实际输出 token 数，业务生成默认行为不变。DMA 缺少原生计时的事件保留在 telemetry，但不写入可标定样本。

## 运行配置与原生观测

启动脚本默认 `kernel/layer_first/TMA=0`，与当前 H200/Qwen3 真机运行一致。显式环境变量仍可覆盖。早期 `page_first` 的 staged D2H 路径出现过进程 segfault，未隔离为单一上游/驱动原因，因此不再作为本项目默认值；本项目没有修改 CUDA kernel。

报价复用实际 native ratio/clip、当前 running_batch 成员和原生 running offset；最近 match 只复制标量结果，不重复匹配。读取成本为 O(active+waiting)，缺失观测保留 None。它们已进入当前 `native_budget` 选择，字段及近似边界见[等待模型](../docs/research/admission_wait_model.md)。

源码核验以当前 `source_manifest.json` 与补丁为准，真实运行以各批次源码清单为准。旧补丁的阶段纪要不再混排为当前状态。

## C：受控生命周期归因验收

```bash
python -m engine_patches.lifecycle_probe --config configs/qwen3_8b_scope.example.yaml \
  --base-url http://127.0.0.1:30001 --tiers gpu,host \
  --shared-tokens 128 --branch-tokens 128 --report runs/lifecycle-paired.json
```

该工具仅用于显式启用 profiling 的空闲独立实例，会 cold reset。每个 tier 做 `keep_old` 与 `retire_old` 两个配对 trial：固定 token 序列建立共享前缀与两个独有分支，验证新 epoch 完整结果；再触达旧分支使它比新分支更新，防止“普通 LRU 本来就淘汰旧分支”掩盖引用退役作用。只有 treatment 模拟业务新结果提交后退役旧 epoch；随后施加同量原生 eviction。它不模拟 SQLite 事务，提交边界明确标为 `simulated_after_verified_new_complete_result`。

新增 `profile_prepare(mode="pressure", tier="gpu"|"host", tokens=N)` 仅在 profiler 开关与空闲检查通过后接受，调用 native `evict(EvictParams(...))` 或 `evict_host(N)`，**不释放任何 session 引用**。host trial 先通过相同 pressure 清除 GPU 副本并 settle 原生写入，继续保留两个 epoch 的引用，再单独施压 host。原生算法可能按整节点回收超过请求量，结果保留实际 native eviction counters 和唯一 pool 的前后差额。

输出保存旧/新/共享路径的前后报价、两层 unique slot、retire ACK、真实原生 eviction 数量和 epoch 状态。只有新分支保留改善、旧分支保留降低、共享前缀存活、retire 本身未改变物理占用等证据一致，才标 `effect_observed`；否则报告 `no_effect_observed`，不预设收益。无法构造 host 副本（例如 selective 未准入）会明确报 fixture 不满足。GPU-only 服务指定 `--tiers gpu`。这是可归因的机制验收，不能外推一般时延、吞吐或回答质量收益。

## 可选 Host 备份准入

`CONTEXTKV_HOST_ADMISSION=off|fixed|conditioned` 默认off。启用时将业务session映射到epoch，过滤请求触发的合法 BackupKV action；在原生分配、锁和DMA之前按压力与每请求累计新页额度判定。write_back淘汰路径和原生去重/补齐保持。具体机制与EfficientAgent归属见[Host准入说明](../docs/research/host_admission_design.md)。当前共同对照使用off，不能将可选准入的局部结果算作本轮收益。

## 可选准入诊断

同时启用 `CONTEXTKV_ENABLE_PROFILING=1`、`CONTEXTKV_ADMISSION_DIAGNOSTICS=1` 时，terminal事件保存原生 receive/首次forward/first-token/completion及有限预算摘要。真实加入 `can_run_list` 决定接受；`NO_TOKEN`也可能是加入后预算耗尽。正常服务默认关闭，诊断不更改原生调度和receipt。

## 固定复用来源

本项目复用已有原生能力，没有另造 DMA 计时器、allocator 或 kernel：

| 固定来源 | 复用内容 |
|---|---|
| [req_time_stats.py](https://github.com/sgl-project/sglang/blob/3f36fc08331f64cd63c4fbbdf39cc402f38bb38b/python/sglang/srt/observability/req_time_stats.py) | 原生 perf_counter 请求阶段时钟 |
| [batch_result_processor.py](https://github.com/sgl-project/sglang/blob/3f36fc08331f64cd63c4fbbdf39cc402f38bb38b/python/sglang/srt/managers/scheduler_components/batch_result_processor.py) | native copy completion 之后的 prefill 完成与终态观察点 |
| [cache_controller.py](https://github.com/sgl-project/sglang/blob/3f36fc08331f64cd63c4fbbdf39cc402f38bb38b/python/sglang/srt/managers/cache_controller.py) · [l2_transfer.py](https://github.com/sgl-project/sglang/blob/3f36fc08331f64cd63c4fbbdf39cc402f38bb38b/python/sglang/srt/mem_cache/l2_transfer.py) | HiCacheAck 的实际 bytes 与已启用 CUDA event pair；无计时支持时明确未知 |
| [unified_radix_cache.py](https://github.com/sgl-project/sglang/blob/3f36fc08331f64cd63c4fbbdf39cc402f38bb38b/python/sglang/srt/mem_cache/unified_radix_cache.py) | 原生完成 ACK、写回、淘汰与 session 引用路径 |
| [allocator/base.py](https://github.com/sgl-project/sglang/blob/3f36fc08331f64cd63c4fbbdf39cc402f38bb38b/python/sglang/srt/mem_cache/allocator/base.py) · [memory_pool_host.py](https://github.com/sgl-project/sglang/blob/3f36fc08331f64cd63c4fbbdf39cc402f38bb38b/python/sglang/srt/mem_cache/memory_pool_host.py) | 唯一物理 pool capacity/free 计数 |

这些审查源码的固定 hash 均纳入 `source_manifest.json`。prepare 脚本检查新增模块镜像、guard 顺序、native timing/ACK seams 与完整结果先发布后 HTTP 返回的路径。
