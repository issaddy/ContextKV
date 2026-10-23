# SGLang ContextKV 协议补丁

基准为 SGLang `3f36fc08331f64cd63c4fbbdf39cc402f38bb38b`。`0001-contextkv-protocol.patch` 是基于该提交真实源码生成的可应用补丁，覆盖 12 个源文件。`source_manifest.json` 保存审查文件 SHA256；`contextkv_runtime.py` 是补丁新增模块的同内容镜像，供 CPU 合同测试。准备脚本校验二者一致。

已在 macOS 完成固定源码独立下载、原文件 hash 验证、`git apply --check`、应用和 Python 语法编译。协议与树视图有 CPU 单元测试。`verification_report.json` 只记录源码验证，不汇总 GPU 实测；H200 的 GPU-only/兼容 HiCache 功能冒烟与小规模标定已运行，范围和配置见下文。完整性能、质量结论仍须独立实验报告。

## 实现范围

- `/context/engine/info`、`/cache/quote`、`/context/epoch/prepare`、`/context/epoch/retire`、`/context/epoch/status` 全部经 tokenizer communicator 进入拥有缓存的 scheduler；HTTP 线程不读取可变树。
- 批量报价使用同一个观察周期，对完整最终 token IDs 共用原生 `Req._compute_max_prefix_len` 与页对齐。native radix key 使用 `array('q')`，与真实树一致。
- `UnifiedTreeCore.peek_full_prefix` 只遍历匹配路径，返回连续 GPU ready、GPU/host recoverable、需新加载的 host-only、H2D pending、D2H pending。节点中间命中不 split；不触碰 LRU、引用、分配器、传输队列或 CUDA fence。
- H2D 的 `contextkv_h2d_pending_id` 是新增观测字段，对所有原生写策略在提交加载时标记，节点 split 继承、原生加载 ACK 后清除。它不参与原生 admission、锁、淘汰决策。D2H 使用已有 `write_through_pending_id`；已发布地址但未完成传输的副本不计 ready。
- NATIVE 加载预测参考原生 `load_back_threshold`；旧 `native_estimate_uncertain` 保留 pending 或瞬时 free GPU slot 不足的原语义，同时 `uncertainty_reasons` 区分 `inflight_transfer` 与 `capacity_pressure`。后者表示可能需要原生淘汰或等待，不表示缓存信息不可读取。报价不是预留，实际匹配、加载、排队与预算决定仍由原生路径执行。
- prepare 同一 operation 幂等，不重开 native generation；生成在原生 `ensure_session_generation` 前同时验证 restart、registry 状态和当前 tracker generation。研究 epoch 禁止经原生 open/close 重新开启或绕过退役。
- retire 先阻止新请求，等待已接纳请求终态，再撤销 native 软引用并 ACK。软引用退役的 free 计数恒为零；缓存物理占用不会被假定已回收。原生已接纳请求在 RETIRING 期间可能先完成注册、再立即被退役撤销，因此 receipt 的 `session_ref_registered` 表示本次曾成功经过注册路径，不是 ACK 后仍持有引用的证明。
- 终态 receipt 通过原生 customized-info 输出，包含真实 tracker 注册结果、cache inserted length 和 native 分层命中。实际 prefill 工作来自每次成功 worker forward 的 `extend_lens` 累计，覆盖 chunk/retraction 重算并排除 mixed batch 的 decode 行，不用 input-minus-hit 冒充真实执行工作。

首版运行合同是单 GPU、TP/DP/PP=1、Python TreeCore、dense FULL、Qwen2.5-7B-Instruct 固定 revision、单条累计 SSE 流式或非流式请求、无 LoRA、默认 cache key。HiCache 的 cache 模式受支持，buffer-only、L3、外部 linker、SWA/Mamba、speculative、PD/PDMux 不在此补丁范围。HTTP client 必须先验证研究能力，绝不把 guard 字段直接提交给原生未补丁服务器。

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

固定源码的 `python/pyproject.toml` 实际声明 `torch==2.14.1`、`transformers==5.19.0`、`tokenizers==0.23.2`；文件原 hash 纳入 `source_manifest.json`，来源与模型 config hash 另见 `model_manifest.json`。这是一份明确依赖来源的安装入口，未在当前 macOS 环境验证整套 CUDA 构建。正式运行应冻结 Linux 环境包清单、驱动和编译器版本。

在控制器和服务端使用相同 tokenizer 栈，从本地固定模型快照生成指纹：

```bash
export CONTEXTKV_MODEL_PATH=/path/to/pinned/Qwen2.5-7B-Instruct
export CONTEXTKV_FINGERPRINT_JSON="$(python - <<'PY'
import json, os
from dataclasses import asdict
from controller.templates import HuggingFaceTemplate
adapter = HuggingFaceTemplate(
    os.environ['CONTEXTKV_MODEL_PATH'],
    revision='a09a35458c702b33eeacc393d103063234e8bc28',
    local_files_only=True,
)
print(json.dumps(asdict(adapter.fingerprint)))
PY
)"
scripts/launch_sglang.sh --host 127.0.0.1 --port 30000
```

启动脚本固定模型 revision，默认开启原生 HiCache（host 容量示例 16 GB）；通过 `CONTEXTKV_ENABLE_HICACHE=0` 运行 GPU-only 对照，通过 `CONTEXTKV_HICACHE_GB` 与 `CONTEXTKV_PAGE_SIZE` 设置实验容量与页大小，`CONTEXTKV_WRITE_POLICY` 可选择原生 `write_through`、`write_through_selective`、`write_back`。默认值是启动配置，不是经验证的最佳参数。scheduler 首次建立协议时验证实际运行 tokenizer 的 backend 序列化 hash、chat template hash、配置中的 model revision 和本地 `config.json` 的固定 canonical hash；这些检查不等于对所有权重文件做过完整 hash 校验。

## 观测与恢复合同

`/context/metrics` 经相同 scheduler 控制面读取原生 allocator 的 `capacity − free`，返回 GPU/host 唯一物理占用和当前 running/waiting 请求数。它不扫描树，不按 session 引用重复计数。事件环最多保留 4096 条记录，`after_seq` 增量拉取，并通过 `events_truncated/dropped_before_seq` 显式暴露丢失区间。

报价还返回 `gpu_capacity_tokens` 和原生 `full_evictable_size()` 的 `evictable_gpu_tokens`；两者都是 O(1) 计数。GPU 可淘汰量是 FULL device 未硬锁 token 总数，包含可能被当前候选命中的共享前缀。不能把它与候选 ready prefix 重复计算成 admission 保证。原生 [PrefillBudget](https://github.com/sgl-project/sglang/blob/3f36fc08331f64cd63c4fbbdf39cc402f38bb38b/python/sglang/srt/mem_cache/prefill_budget.py#L98-L113) 使用 free + evictable；[PrefillAdder](https://github.com/sgl-project/sglang/blob/3f36fc08331f64cd63c4fbbdf39cc402f38bb38b/python/sglang/srt/managers/schedule_policy.py#L1350-L1407) 再保护候选前缀并考虑已运行请求的剩余输出预算、页开销、chunk 和其他约束。这个补丁不模拟或修改它。Host 没有对应的精确 O(1) token 总计；`evictable_host_tokens=null` 明确未知，既不报假零，也不为报价扫描所有 host 节点。

报价与 metrics 的 `native_running_requests` / `native_waiting_requests` 只传出每个请求的 `input_tokens`、`generated_tokens`、`max_new_tokens`、`ignore_eos`、`kv_allocated_tokens`、`kv_committed_tokens` 和 `cache_protected_tokens`，没有文本、token IDs 或请求标识。输出预算取 native 已按上下文和池容量裁剪后的 `sampling_params.max_new_tokens`，已生成数只取 scheduler 当前已处理的 CPU 状态；不读取未来输出。[ReqKvInfo](https://github.com/sgl-project/sglang/blob/3f36fc08331f64cd63c4fbbdf39cc402f38bb38b/python/sglang/srt/managers/schedule_batch.py#L974-L987) 的三种 KV 数字是**逻辑序列边界**：树拥有 `[0, cache_protected_tokens)`，请求自有尾部位于其后；共享前缀、页边界和完成后的缓存插入都使其不能直接作为唯一物理占用或即将释放 slot 数。Overlap 时 committed 状态还可能落后已执行 forward。

运行集合复用 native `collect_inflight_reqs()` 对 running/last/overlap batch 去重，纳入独立 chunked request，排除已结束请求；已回到 waiting 的请求只计 waiting。Running 摘要无顺序语义；waiting 保留当前列表顺序，但原生优先级、staging 和 admission 可改变实际执行顺序。读取成本为 O(active + waiting)，每个请求只读标量和数组长度，不遍历 token 内容、不维护新缓存、不处理 ACK、不改变锁和 LRU。没有 scheduler 时摘要为 `null`，已知空集合为 `[]`。这些观测可用于区分容量压力和传输未完成，尚不构成经过 GPU 验证的新等待预测模型。

- `request_terminal`：receipt 与原生完成时序。
- `transfer`：在原生 HiCache ACK 已完成后复用 `num_bytes`、`start_event.elapsed_time(finish_event)`，输出 H2D/D2H 字节与 CUDA event 时间。计时不支持时为 `null`，不能将 CPU enqueue 时间替代它。该观察点不增加 CUDA 同步，不改变传输调度。
- `retire_ref`：软引用退役，free 为零。
- `pool_snapshot`：仅 `CONTEXTKV_SAMPLE_POOL=1` 开启；每个 scheduler 边界用 O(1) allocator 计数采样，只有占用变化时写事件。可计算**采样峰值**与占用积分，不能据此声称覆盖每个瞬时 alloc/free 峰值。

生成结果中的 `server_ttft_ms` 为 scheduler 收到请求至原生首 token 完成，`server_elapsed_ms` 为收到至终态，`decode_ms` 为首 token 至终态，`tpot_ms` 为该段除以后续输出 token 数。`forward_to_first_token_ms` 单独表示原生 forward-entry 至首 token 完成。全部来自同一服务端 monotonic 时钟；原生 forward-entry 在部分 cache 准备之后，不能拿它精确拆出排队与恢复。

`elapsed_ms` 是客户端提交至完整 HTTP 响应。默认累计 SSE 客户端通过首次非空 output_ids 和最后新增 token 的帧观察 `ttft_ms/last_token_ms`，重复累计帧不更新时刻。客户端 TPOT 由首末 chunk 间隔除以 N−1 计算，报告明确标为 chunk 观察；服务端 `tpot_ms` 仍是另一时间域的原生指标。可显式 `SGLangEngineClient(..., stream=False)` 使用非流式，此时两个客户端 token 时刻未知；专用 profiler 保持非流式。`queue_ms/prefill_ms/restore_prefill_ms` 也保持未知，不能将阶段墙时冒充独立 CUDA kernel 时间。成本标定使用 `phase=first_token, timing_scope=server_request` 联合阶段，在同一并发压力桶中比较，避免重复加排队或 H2D。

完整最终文本和 token IDs 经过原生 detokenizer，终态响应之前通过一次 IPC 发布回 scheduler。guarded SSE 统一为累计输出，兼容原生 incremental streaming 开关。最后新增 token 先以不带 receipt/finish 标志的帧发送，再执行发布 IPC，最后发送重复累计 token 的完整终态 receipt 帧，避免将发布延迟算入最后 token 到达时刻。客户端只有收到终态 receipt 和 `[DONE]` 才返回结果；断帧、部分断连、缺少终态均走未知结果恢复，不能提交部分答案。`status` 在默认 900 秒 TTL、64 MiB 总结果预算、1024 个 attempt receipt 的范围内返回原样结果；通过 `CONTEXTKV_RESULT_TTL_SECONDS` 和 `CONTEXTKV_RESULT_MAX_BYTES` 调整。超限/过期、终态后发布前的进程崩溃仍可能只剩 receipt；控制器必须明确恢复或放弃，不能声称无限期 exactly-once。UNKNOWN 不是“未执行”证明。epoch 墓碑保持到引擎重启；专用 profiler 冷重置同时更换 restart identity。

## 可执行的真机验收与标定入口

工程验收先通过 CPU 合同和源码检查，再在已授权的 H200 环境运行以下入口。协议冒烟与小规模标定的通过，不代替正式基线、质量和容量实验。

先在正常研究服务器验证协议，不 flush 共享缓存：

```bash
python -m engine_patches.smoke --config configs/sglang.yaml \
  --report runs/sglang-smoke.json
```

该入口验证指纹与固定提交、prepare 幂等、真实生成 receipt/服务端时序、原样 status 结果恢复、重复只读报价、退役 ACK 及旧 epoch 拒绝。`smoke` 是功能验收，不是性能或质量结论。

成本标定需要另起**专用**引擎实例（端口和相应 config 的 `base_url` 一致）：

```bash
CONTEXTKV_ENABLE_PROFILING=1 CONTEXTKV_SAMPLE_POOL=1 \
  scripts/launch_sglang.sh --host 127.0.0.1 --port 30001
python -m profiling.sglang_probe --config configs/sglang.yaml --base-url http://127.0.0.1:30001 \
  --output runs/calibration.jsonl --lengths 1024,4096,8192 \
  --prefix-fractions 0,0.5,0.99 --outputs 128,512 --concurrency 1,4 \
  --cache-states cold,gpu_ready,host_only --warmup 1 --repetitions 3
```

设置 `CONTEXTKV_MODEL_PATH` 指向固定模型目录，或给两个 CLI 传 `--model-path`。客户端本地 tokenizer 与引擎必须一致。正式标定前先用 `--lengths 128 --prefix-fractions 0,0.5 --outputs 8 --concurrency 1 --repetitions 1` 冒烟。

`/context/profiling/prepare` 默认禁用，并仅在空闲专用实例接受操作。cold 调用原生 cache flush 并更新 restart identity；gpu_ready 由普通原生生成暖前缀并 settle 写入；host_only 在暖前缀后经原生 write ACK、软引用退役和 eviction 构造。每轮测量前后均在专用实例 settle 写 ACK，前游标排除暖缓存传输，后游标包含本轮生成写回；它不进入 server request 阶段耗时。客户端随后用只读报价核实**实际**前缀和页边界，容量/写策略不满足时停止，不能用标签伪称“冷/暖/host”。GPU-only 服务须省略 host_only。write_through_selective 若未保存 host 前缀也会显式失败，不能误记为 host 测量。

每个 cache/shape/concurrency case 默认先运行 `--warmup 1`：执行相同 prepare/probe，但只在 telemetry 标记 `warmup=true`，不写入 Measurement 训练行。随后正式 repetitions 标记 `warmup=false`；manifest 记录 warmup 数量。这样避免首执行 JIT/初始化开销直接污染成本表，仍保留全部暖机证据。

输出包括原始 Measurement JSONL、同目录 `.telemetry.jsonl`（逐请求 receipt/timing 与 pool/DMA 事件）和 `.manifest.json`（指纹、配置、实际 pool、seed、case、完成状态）。专用合成标定请求设置 `ignore_eos=True` 以覆盖指定输出长度；所有记录仍采用实际输出 token 数，业务生成默认行为不变。DMA 缺少原生计时的事件保留在 telemetry，但不写入可标定样本。

## H200 上已验证的 HiCache 兼容配置

最新只读观测补丁 `3aac1dfd…c1ac4b` 在此基础上增加实际 native ratio/clip、当前 running_batch 的预算成员与原生 offset 复算值，以及最近一次原生 match 的标量结果和服务端 monotonic 纳秒时点。该版本已完成 CPU 原源码/HTTP 合同与干净固定源码应用检查，**本次扩展尚未进行 GPU/IPC 验收**；下面已有实测不能自动算作新 hook 的运行验证。字段、O(active+waiting) 成本、None/过时观测语义和下一轮 probe 合同见[等待模型观测扩展](../docs/admission_wait_model.md#已实现的只读观测扩展等待下一轮-gpu-采样)。本次不改变原生调度或控制器预测策略。

只读工作量扩展补丁 `80147e2b…4476a5` 已在同一 H200 完成四方法两重复共 328 轮 16K 闭环及 8 次冻结输入生成，均成功；运行/等待摘要在实际请求进行中核对过。原始补丁 `69bf539f…76912` 的功能与受控生命周期记录继续保留。新阶段的范围、成本与负面结果见[工作量验证](../docs/stage_workload_validation.md)，表格局部实验见[冻结历史验证](../docs/table_relations_validation.md)。源码验证 JSON 仍只描述静态检查，不用于汇总 GPU 效果。

2026-10-23 的远端 H200 环境使用 driver 595.58.03、PyTorch 2.14.1+cu130 和固定 SGLang 源码。GPU-only 协议冒烟与多轮生成闭环已通过。原生 `kernel + page_first` 在 D2H staged 路径出现进程 segfault，栈经过 `transfer_hicache_all_layer_staged_lf_pf → staged_write_back → cudaMemcpyBatchAsync/cuMemcpyBatchAsync_v2`；尚未隔离到驱动、运行时或上游实现的单一原因，不能将其断定为驱动问题。

复用原生配置的兼容路径已经通过 host 协议冒烟及 128-token、8-token 输出、c=1 的 cold/GPU-ready/host-only 小 grid：

```bash
CONTEXTKV_HICACHE_LAYOUT=layer_first CONTEXTKV_HICACHE_IO_BACKEND=kernel \
SGLANG_HICACHE_TMA_TRANSFER=0 CONTEXTKV_ENABLE_PROFILING=1 \
  scripts/launch_sglang.sh --host 127.0.0.1 --port 30001
```

固定 [MHA host pool](https://github.com/sgl-project/sglang/blob/3f36fc08331f64cd63c4fbbdf39cc402f38bb38b/python/sglang/srt/mem_cache/pool_host/mha.py) 只为 `page_first` 创建 staged D2H 路径；`layer_first` 绕开该路径。固定 [HiCache kernel dispatcher](https://github.com/sgl-project/sglang/blob/3f36fc08331f64cd63c4fbbdf39cc402f38bb38b/python/sglang/kernels/ops/kvcache/hicache.py) 支持 `SGLANG_HICACHE_TMA_TRANSFER=0` 选择传统 register kernel。仅关闭 TMA **不会**关闭 page_first 的 staged D2H。这次没有修改任何 CUDA kernel。

启动脚本默认值仍为原生 `kernel/page_first`；用 `CONTEXTKV_HICACHE_IO_BACKEND`、`CONTEXTKV_HICACHE_LAYOUT` 显式选择兼容组合。cache_config_hash 与 metrics.cache_config 保存实际 controller.io_backend、host pool.layout 和原生 TMA 环境开关；TMA 开关不代表每个 transfer 必然选用 TMA kernel。正式标定和全部对照策略须使用同一配置，不能混合不同 layout/backend/TMA 的样本。早期无 warmup 的 30 条小 grid 观测用于功能核验，正式成本标定应使用当前带 warmup 的入口重采样。用 `--runtime-manifest runs/runtime-manifest.json` 将 `profiling.runtime_manifest` 的只读采集结果原样绑定到标定 manifest.runtime_manifest，probe 不另行探测 GPU。

## C：受控生命周期归因验收

```bash
python -m engine_patches.lifecycle_probe --config configs/sglang.yaml \
  --base-url http://127.0.0.1:30001 --tiers gpu,host \
  --shared-tokens 128 --branch-tokens 128 --report runs/lifecycle-paired.json
```

该工具仅用于显式启用 profiling 的空闲独立实例，会 cold reset。每个 tier 做 `keep_old` 与 `retire_old` 两个配对 trial：固定 token 序列建立共享前缀与两个独有分支，验证新 epoch 完整结果；再触达旧分支使它比新分支更新，防止“普通 LRU 本来就淘汰旧分支”掩盖引用退役作用。只有 treatment 模拟业务新结果提交后退役旧 epoch；随后施加同量原生 eviction。它不模拟 SQLite 事务，提交边界明确标为 `simulated_after_verified_new_complete_result`。

新增 `profile_prepare(mode="pressure", tier="gpu"|"host", tokens=N)` 仅在 profiler 开关与空闲检查通过后接受，调用 native `evict(EvictParams(...))` 或 `evict_host(N)`，**不释放任何 session 引用**。host trial 先通过相同 pressure 清除 GPU 副本并 settle 原生写入，继续保留两个 epoch 的引用，再单独施压 host。原生算法可能按整节点回收超过请求量，结果保留实际 native eviction counters 和唯一 pool 的前后差额。

输出保存旧/新/共享路径的前后报价、两层 unique slot、retire ACK、真实原生 eviction 数量和 epoch 状态。只有新分支保留改善、旧分支保留降低、共享前缀存活、retire 本身未改变物理占用等证据一致，才标 `effect_observed`；否则报告 `no_effect_observed`，不预设收益。无法构造 host 副本（例如 selective 未准入）会明确报 fixture 不满足。GPU-only 服务指定 `--tiers gpu`。这是可归因的机制验收，不能外推一般时延、吞吐或回答质量收益。

### H200 受控实测结果

2026-10-23，补丁 `69bf539f582ad7065d1b16c6fa6efe348c3e69d9d4aec8a296f3206996b76912` 在上述 `kernel/layer_first/TMA=0` 环境完成实际 GPU 验收。runtime manifest、协议 smoke、带每 case 一次 warmup 的小 grid 均通过；小 grid 产生 30 条正式观测，warmup 不进入训练数据。

C 配对 fixture 使用 128-token 共享前缀、每个 epoch 各 128-token 独有后缀；**新 epoch 完成后再次触达旧 epoch**，使旧分支更近使用。这样普通 LRU 倾向保留旧分支，配对差异才能说明 soft reference 优先级与退役的作用，而不是把“旧分支本来更旧”误作机制收益。GPU 和 host 两层都观察到以下相同结果：

| 配对条件 | 施压后旧路径可用前缀 | 施压后新路径可用前缀 | 共享前缀 | 原生实际淘汰 |
|---|---:|---:|---:|---:|
| keep_old：保留旧引用 | 256 tokens | 128 tokens | 128 tokens | 128 tokens |
| retire_old：退役旧引用 | 128 tokens | 256 tokens | 128 tokens | 128 tokens |

在两层 trial 中，retire ACK 前后 GPU/host unique pool 的变化均为 **0**；之后实际回收来自明确施加的 native eviction。新路径多保留 128 tokens，旧独有路径让出 128 tokens，共享路径仍然存在，因此该限定 fixture 报告 `effect_observed=true`。

这验证的是“软引用退役改变原生淘汰选择，并为当前分支保留空间”的受控机制，**不证明一般工作负载上的吞吐、完成时间或质量收益**。本工具只在已验证完整新回答之后模拟业务 commit 边界；持久化、CAS、outbox、异常恢复等生产事务由 controller 的生命周期测试覆盖，该 GPU fixture 不能替代那些测试或故障注入。

## 复用依据与后续验收

本轮优先复用已有原生能力，没有另造 DMA 计时器、allocator 或 kernel：

| 固定来源 | 复用内容 |
|---|---|
| [req_time_stats.py](https://github.com/sgl-project/sglang/blob/3f36fc08331f64cd63c4fbbdf39cc402f38bb38b/python/sglang/srt/observability/req_time_stats.py) | 原生 perf_counter 请求阶段时钟 |
| [batch_result_processor.py](https://github.com/sgl-project/sglang/blob/3f36fc08331f64cd63c4fbbdf39cc402f38bb38b/python/sglang/srt/managers/scheduler_components/batch_result_processor.py) | native copy completion 之后的 prefill 完成与终态观察点 |
| [cache_controller.py](https://github.com/sgl-project/sglang/blob/3f36fc08331f64cd63c4fbbdf39cc402f38bb38b/python/sglang/srt/managers/cache_controller.py) · [l2_transfer.py](https://github.com/sgl-project/sglang/blob/3f36fc08331f64cd63c4fbbdf39cc402f38bb38b/python/sglang/srt/mem_cache/l2_transfer.py) | HiCacheAck 的实际 bytes 与已启用 CUDA event pair；无计时支持时明确未知 |
| [unified_radix_cache.py](https://github.com/sgl-project/sglang/blob/3f36fc08331f64cd63c4fbbdf39cc402f38bb38b/python/sglang/srt/mem_cache/unified_radix_cache.py) | 原生完成 ACK、写回、淘汰与 session 引用路径 |
| [allocator/base.py](https://github.com/sgl-project/sglang/blob/3f36fc08331f64cd63c4fbbdf39cc402f38bb38b/python/sglang/srt/mem_cache/allocator/base.py) · [memory_pool_host.py](https://github.com/sgl-project/sglang/blob/3f36fc08331f64cd63c4fbbdf39cc402f38bb38b/python/sglang/srt/mem_cache/memory_pool_host.py) | 唯一物理 pool capacity/free 计数 |

这些审查源码的固定 hash 均纳入 `source_manifest.json`。prepare 脚本检查新增模块镜像、guard 顺序、native timing/ACK seams 与完整结果先发布后 HTTP 返回的路径。

下一阶段 GPU 验收仍包括：真实进程 IPC/完整依赖导入、空/非空树、跨页部分命中、两会话共享前缀、三种 host 写策略 pending/ACK、split 与加载重叠、退役并发、断连/取消、重启、容量压力和持续运行；之后进行等预算基线/消融、P3 检索、质量评估与按会话聚类统计。纯 CUDA Prefill/Decode 内核分解和全分配器精确瞬时峰值仍不在当前测量合同内，需要时再通过 GPU profiler 扩展，不能从现有数据推断。
