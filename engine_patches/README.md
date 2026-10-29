# SGLang ContextKV 协议补丁

基准为 SGLang `3f36fc08331f64cd63c4fbbdf39cc402f38bb38b`。`0001-contextkv-protocol.patch` 是基于该提交真实源码生成的可应用补丁，覆盖 15 个源文件。`source_manifest.json` 保存审查文件 SHA256；`contextkv_runtime.py` 是补丁新增模块的同内容镜像，供 CPU 合同测试。准备脚本校验二者一致。

已在 macOS 完成固定源码独立下载、原文件 hash 验证、`git apply --check`、应用和 Python 语法编译。协议与树视图有 CPU 单元测试。`verification_report.json` 只记录源码验证，不汇总 GPU 实测；当前 H200/Qwen3 联合运行及效果见[数据与评价](../docs/data_and_evaluation.md)。

## 实现范围

- `/context/engine/info`、`/cache/quote`、`/context/epoch/prepare`、`/context/epoch/retire`、`/context/epoch/status` 全部经 tokenizer communicator 进入拥有缓存的 scheduler；HTTP 线程不读取可变树。
- 批量报价使用同一个观察周期，对完整最终 token IDs 共用原生 `Req._compute_max_prefix_len` 与页对齐。native radix key 使用 `array('q')`，与真实树一致。
- `UnifiedTreeCore.peek_full_prefix` 只遍历匹配路径，返回连续 GPU ready、GPU/host recoverable、需新加载的 host-only、H2D pending、D2H pending。节点中间命中不 split；不触碰 LRU、引用、分配器、传输队列或 CUDA fence。
- H2D 的 `contextkv_h2d_pending_id` 是新增观测字段，对所有原生写策略在提交加载时标记，节点 split 继承、原生加载 ACK 后清除。它不参与原生 admission、锁、淘汰决策。D2H 使用已有 `write_through_pending_id`；已发布地址但未完成传输的副本不计 ready。
- NATIVE加载预测参考原生`load_back_threshold`；`native_estimate_uncertain`表示pending或瞬时free GPU slot不足，`uncertainty_reasons`进一步区分`inflight_transfer`与`capacity_pressure`。报价提供状态快照，实际匹配、加载、排队和预算决定由原生路径执行。
- 活动 epoch 的 prepare 对同一 operation 幂等，不重开 native generation；生成在原生 `ensure_session_generation` 前同时验证 restart、registry 状态和当前 tracker generation。保留期内的关闭 ID 继续由原生关闭记录阻止重开。
- retire先停止接收新请求，等待已接纳请求终态，再撤销native软引用并ACK，free计数为零。RETIRING期间的请求可以先完成注册，再由退役流程撤销；receipt的`session_ref_registered`记录本次注册事件。
- 终态 receipt 通过原生 customized-info 输出，包含真实 tracker 注册结果、cache inserted length 和 native 分层命中。实际 prefill 工作来自每次成功 worker forward 的 `extend_lens` 累计，覆盖 chunk/retraction 重算并排除 mixed batch 的 decode 行。

当前补丁支持Qwen3 LocalModelScope、单GPU、TP/DP/PP=1、Python TreeCore、dense FULL、默认cache key和HiCache cache模式，提供累计SSE及非流式请求。LoRA通过离线导出独立dense checkpoint后加载。动态LoRA、buffer-only、L3、外部linker、SWA/Mamba、speculative及PD/PDMux尚未适配。HTTP client先检查协议能力，再提交guard字段。

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

默认使用已有的Qwen3-32B本地快照，先声明对应scope：

```bash
export CONTEXTKV_MODEL_SCOPE_PATH=configs/model_scopes/qwen3-32b.local.json
scripts/bootstrap_sglang.sh
source .venv-sglang/bin/activate
export CONTEXTKV_FINGERPRINT_JSON="$(cat runs/fingerprint.json)"
# H200资源参数示例，按实际设备配置容量。
CONTEXTKV_HICACHE_GB=16 CONTEXTKV_HICACHE_LAYOUT=layer_first \
SGLANG_HICACHE_TMA_TRANSFER=0 scripts/launch_sglang.sh \
  --host 127.0.0.1 --port 30000 --max-total-tokens 32768 --mem-fraction-static .8
```

`bootstrap_sglang.sh`仅安装固定SGLang/项目依赖，然后通过共享`ModelConfig/create_model_template`验证已存在的scope快照、tokenizer和模板，保存指纹。它不调用snapshot_download，不创建旧模型目录；缺少scope或本地文件会明确失败。源码准备下载SGLang及依赖，模型使用已有本地文件。

`launch_sglang.sh`从`CONTEXTKV_MODEL_SCOPE_PATH`读取model/tokenizer路径，HF loading revision为None；冲突的`--revision`或model/tokenizer参数会报错。可选`CONTEXTKV_MODEL_PATH`用于核对scope路径。scope按模型元数据标识本地快照，验证范围由`weight_content_verified`记录。



`CONTEXTKV_ENABLE_HICACHE=0`启用GPU-only；`CONTEXTKV_HICACHE_GB`/`CONTEXTKV_PAGE_SIZE`设置容量与页大小，`CONTEXTKV_WRITE_POLICY`选择原生写策略。scheduler的`get_runtime`在首次初始化时读取scope并核对模型配置、文件库存、tokenizer/template和fingerprint，缺scope返回`MODEL_SCOPE_REQUIRED`，之后复用实例。CPU `ContextKVRuntime`使用无模型fixture测试缓存与事务机制。

当前scope支持dense Qwen3、BF16和原始RoPE配置；32B的head_dim使用配置值128。

## 观测与恢复合同

`/context/metrics` 经相同 scheduler 控制面读取原生 allocator 的 `capacity − free`，返回 GPU/host 唯一物理占用和当前 running/waiting 请求数。它不扫描树，不按 session 引用重复计数。事件环最多保留 4096 条记录，`after_seq` 增量拉取，并通过 `events_truncated/dropped_before_seq` 显式暴露丢失区间。

报价返回O(1)计数`gpu_capacity_tokens`及原生`full_evictable_size()`对应的`evictable_gpu_tokens`。后者包含FULL device未硬锁token及可能被候选命中的共享前缀。原生[PrefillBudget](https://github.com/sgl-project/sglang/blob/3f36fc08331f64cd63c4fbbdf39cc402f38bb38b/python/sglang/srt/mem_cache/prefill_budget.py#L98-L113)使用free + evictable，[PrefillAdder](https://github.com/sgl-project/sglang/blob/3f36fc08331f64cd63c4fbbdf39cc402f38bb38b/python/sglang/srt/managers/schedule_policy.py#L1350-L1407)保护候选前缀，再计入运行请求输出预算、页开销与chunk等需求。Host可淘汰总量缺少对应的O(1)观测，`evictable_host_tokens`返回null。

报价与 metrics 的 `native_running_requests` / `native_waiting_requests` 只传出每个请求的 `input_tokens`、`generated_tokens`、`max_new_tokens`、`ignore_eos`、`kv_allocated_tokens`、`kv_committed_tokens` 和 `cache_protected_tokens`，没有文本或 token IDs；新增可选 `request_id` 直接复制既有原生 `req.rid`，供服务集合、预算集合和终态 receipt 精确关联，不创建新标识或内容 hash。输出预算取 native 已按上下文和池容量裁剪后的 `sampling_params.max_new_tokens`，已生成数只取 scheduler 当前已处理的 CPU 状态；不读取未来输出。[ReqKvInfo](https://github.com/sgl-project/sglang/blob/3f36fc08331f64cd63c4fbbdf39cc402f38bb38b/python/sglang/srt/managers/schedule_batch.py#L974-L987) 的三种 KV 数字是**逻辑序列边界**：树拥有 `[0, cache_protected_tokens)`，请求自有尾部位于其后；共享前缀、页边界和完成后的缓存插入都使其不能直接作为唯一物理占用或即将释放 slot 数。Overlap 时 committed 状态还可能落后已执行 forward。

运行集合通过native `collect_inflight_reqs()`对running/last/overlap batch去重，纳入独立chunked request并排除已结束请求；回到waiting的请求归入waiting。Running摘要表示集合，waiting保留列表顺序；实际执行顺序由原生调度决定。读取成本为O(active + waiting)，每请求读取标量和数组长度。缺少scheduler时返回null，已知空集合返回`[]`。等待估计及验证状态见[等待模型](../docs/guides/stage_cost_model.md)。

- `request_terminal`：receipt 与原生完成时序。
- `transfer`：原生HiCache ACK完成后读取`num_bytes`和`start_event.elapsed_time(finish_event)`，记录H2D/D2H字节与CUDA event时间；计时不可用时为null。
- `retire_ref`：软引用退役，free 为零。
- `pool_snapshot`：仅 `CONTEXTKV_SAMPLE_POOL=1` 开启；每个 scheduler 边界用 O(1) allocator 计数采样，只有占用变化时写事件。用于计算 scheduler 边界的**采样峰值**与占用积分。

`server_ttft_ms`为scheduler收到请求至首token完成，`server_elapsed_ms`为收到至终态，`decode_ms`为首token至终态，`tpot_ms`为该段除以后续token数。`forward_to_first_token_ms`从原生forward-entry开始计时；此前已完成部分cache准备，因此前置区间合并记录排队与恢复。全部使用同一服务端monotonic时钟。

`elapsed_ms` 是客户端提交至完整 HTTP 响应。默认累计 SSE 客户端通过首次非空 output_ids 和最后新增 token 的帧观察 `ttft_ms/last_token_ms`，重复累计帧不更新时刻。客户端 TPOT 由首末 chunk 间隔除以 N−1 计算，报告明确标为 chunk 观察；服务端 `tpot_ms` 仍是另一时间域的原生指标。可显式 `SGLangEngineClient(..., stream=False)` 使用非流式，此时两个客户端 token 时刻未知；专用 profiler 保持非流式。`queue_ms/prefill_ms/restore_prefill_ms` 保持未知；当前记录提供联合阶段墙时。成本标定使用 `phase=first_token, timing_scope=server_request` 联合阶段，在同一并发压力桶中比较，避免重复加排队或 H2D。

完整最终文本和 token IDs 经过原生 detokenizer，终态响应之前通过一次 IPC 发布回 scheduler。guarded SSE 统一为累计输出，兼容原生 incremental streaming 开关。最后新增 token 先以不带 receipt/finish 标志的帧发送，再执行发布 IPC，最后发送重复累计 token 的完整终态 receipt 帧，避免将发布延迟算入最后 token 到达时刻。客户端只有收到终态 receipt 和 `[DONE]` 才返回结果；断帧、部分断连、缺少终态均走未知结果恢复，不能提交部分答案。`status` 在默认 900 秒 TTL、64 MiB 总结果预算、1024 个 attempt receipt 的范围内返回原样结果；通过 `CONTEXTKV_RESULT_TTL_SECONDS` 和 `CONTEXTKV_RESULT_MAX_BYTES` 调整。超限/过期、终态后发布前的进程崩溃可能只剩 receipt，此时按恢复协议处理；UNKNOWN 表示执行结果未知。

研究层的退役 epoch 记录和终态操作回执各复用 `receipt_limit` 上限，默认1024；活动及仍在退役中的记录保留到完成。原生[会话追踪器](https://github.com/sgl-project/sglang/blob/3f36fc08331f64cd63c4fbbdf39cc402f38bb38b/python/sglang/srt/mem_cache/unified_cache/session_ref_tracker.py)另保留最近8192个关闭 ID。两处记录都过期后，显式 prepare 可以获得新的原生 generation，旧 generation 的请求仍被拒绝。迟到的退役重试在确认原生引用已不存在、或当前 incarnation 更新时可确认旧引用已解除，不能释放新引用，也不证明旧生成已完成。`control_metadata` 指标分别报告活动和保留的终态记录数量。

原生 `/flush_cache` 会重建 generation 表并从头计数。`get_runtime` 以该表的对象身份检测重置，先更换研究 restart 标识和控制状态，再处理请求；不扫描缓存树或权重。专用 profiler 冷重置复用同一替换逻辑。这样旧请求即使遇到相同数值的 generation，也会被旧 restart 标识拒绝。

## 可执行的真机验收与标定入口

以下入口用于H200环境的协议检查、成本标定与缓存机制验证；系统比较见[数据与评价](../docs/data_and_evaluation.md)。

先在正常研究服务器验证协议，不 flush 共享缓存：

```bash
python -m engine_patches.smoke --config configs/qwen3_32b_scope.example.yaml \
  --report runs/sglang-smoke.json
```

`smoke`验证指纹与固定提交、prepare幂等、生成receipt与服务端时序、status结果恢复、重复只读报价、退役ACK及旧epoch拒绝。

成本标定需要另起**专用**引擎实例（端口和相应 config 的 `base_url` 一致）：

```bash
CONTEXTKV_ENABLE_PROFILING=1 CONTEXTKV_SAMPLE_POOL=1 \
  scripts/launch_sglang.sh --host 127.0.0.1 --port 30001
python -m profiling.sglang_probe --config configs/qwen3_32b_scope.example.yaml --base-url http://127.0.0.1:30001 \
  --output runs/calibration.jsonl --cases-file runs/selected-cost-cases.json \
  --warmup 1 --repetitions 3
```

配置中的`model.scope_path`与服务端`CONTEXTKV_MODEL_SCOPE_PATH`指向同一Qwen3身份，客户端和引擎使用相同tokenizer。标定数据按该模型、缓存及采样配置装配，缺少所需测点时定向补采。

`/context/profiling/prepare`默认关闭，启用后在空闲专用实例执行。cold调用原生cache flush并更新restart identity；gpu_ready通过原生生成建立前缀并等待写入完成；host_only随后执行write ACK确认、软引用退役和eviction。每轮测量前后等待写ACK，前后游标划定本轮传输范围，准备耗时单列。客户端通过quote核对前缀和页边界，容量或写策略不满足时报告准备失败。GPU-only省略host_only；write_through_selective需要实际保存host前缀。

每个 cache/shape/concurrency case 默认先运行 `--warmup 1`：执行相同 prepare/probe，但只在 telemetry 标记 `warmup=true`，不写入 Measurement 训练行。随后正式 repetitions 标记 `warmup=false`；manifest 记录 warmup 数量。这样避免首执行 JIT/初始化开销直接污染成本表，仍保留全部暖机证据。

输出包括原始 Measurement JSONL、同目录 `.telemetry.jsonl`（逐请求 receipt/timing 与 pool/DMA 事件）和 `.manifest.json`（指纹、配置、实际 pool、seed、case、完成状态）。专用合成标定请求设置 `ignore_eos=True` 以覆盖指定输出长度；所有记录仍采用实际输出 token 数，业务生成默认行为不变。DMA 缺少原生计时的事件保留在 telemetry，但不写入可标定样本。

## 运行配置与原生观测

启动脚本默认`kernel/layer_first/TMA=0`，与H200/Qwen3运行配置一致，可用环境变量覆盖。早期`page_first`的staged D2H路径出现过segfault，原因尚未定位，因此默认采用layer_first。

报价复用实际 native ratio/clip、当前 running_batch 成员和原生 running offset；最近 match 只复制标量结果，不重复匹配。读取成本为 O(active+waiting)，缺失观测保留 None。它们已进入当前 `native_budget` 选择，字段及近似边界见[等待模型](../docs/guides/stage_cost_model.md)。

源码核验以当前 `source_manifest.json` 与补丁为准，真实运行以各批次源码清单为准。旧补丁的阶段纪要不再混排为当前状态。

## C：受控生命周期归因验收

```bash
python -m engine_patches.lifecycle_probe --config configs/qwen3_32b_scope.example.yaml \
  --base-url http://127.0.0.1:30001 --tiers gpu,host \
  --shared-tokens 128 --branch-tokens 128 --report runs/lifecycle-paired.json
```

该工具仅用于显式启用 profiling 的空闲独立实例，会 cold reset。每个 tier 做 `keep_old` 与 `retire_old` 两个配对 trial：固定 token 序列建立共享前缀与两个独有分支，验证新 epoch 完整结果；再触达旧分支使它比新分支更新，防止“普通 LRU 本来就淘汰旧分支”掩盖引用退役作用。只有 treatment 模拟业务新结果提交后退役旧 epoch；随后施加同量原生 eviction。它不模拟 SQLite 事务，提交边界明确标为 `simulated_after_verified_new_complete_result`。

新增 `profile_prepare(mode="pressure", tier="gpu"|"host", tokens=N)` 仅在 profiler 开关与空闲检查通过后接受，调用 native `evict(EvictParams(...))` 或 `evict_host(N)`，**不释放任何 session 引用**。host trial 先通过相同 pressure 清除 GPU 副本并 settle 原生写入，继续保留两个 epoch 的引用，再单独施压 host。原生算法可能按整节点回收超过请求量，结果保留实际 native eviction counters 和唯一 pool 的前后差额。

输出保存旧/新/共享路径的前后quote、两层unique slot、retire ACK、原生eviction数量和epoch状态。`effect_observed`要求新分支保留改善、旧分支保留降低、共享前缀存活，且retire本身保持物理占用；其余情况记为`no_effect_observed`。host副本构造失败时报告fixture不满足，GPU-only指定`--tiers gpu`。该工具评价引用退役对分支保留的作用。

## 可选 Host 备份准入

`CONTEXTKV_HOST_ADMISSION=off|fixed|conditioned`默认off，现有系统对照使用off。启用时将业务session映射到epoch，在原生分配、锁和DMA前按压力与每请求累计新页额度筛选BackupKV action；write_back淘汰和原生去重/补齐照常执行。具体机制及EfficientAgent适配见[Host准入说明](README.md)。

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
