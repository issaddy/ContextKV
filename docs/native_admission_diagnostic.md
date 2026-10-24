# 可选原生 admission 诊断

此观测解释 `scheduler_recv → first forward` 的前置联合区间，不改变 planner、成本预测、原生调度或缓存写策略。它不能将联合区间精确分成纯排队、CPU 准备和 DMA 时间，也不直接生成候选对其他请求的 externality 分数。当前等待模型的已有请求完成事件在候选进入前已经确定，不含候选插入后的 continuous-batching 干扰。

## 开启与记录

仅在专用研究服务同时设置 `CONTEXTKV_ENABLE_PROFILING=1` 和 `CONTEXTKV_ADMISSION_DIAGNOSTICS=1`。缺任一个均不启用。`/context/metrics` 的 `admission_diagnostics_enabled` 是实际运行开关；开启时 `/context/engine/info` 也声明它。客户端 `EngineInfo` 默认该字段为 false，旧 wire 省略字段仍可读取；启用服务的 true 值由真实 HTTP 边界合同测试覆盖。仅通过 ContextKV guard 接纳的 Req 建立诊断状态，普通原生请求不被悄悄纳入研究日志。

每个请求仍只有原有的一条 `request_terminal` event，开启后增加 `admission_diagnostic`；HTTP response 的 receipt/timing schema 保持原样。默认关闭仍有固定 helper/getattr/len 调用，但不建立诊断对象、读取额外预算或调用诊断时钟，不能因此声称零开销。`cache_config` 和其 hash 保持旧值。开启后的观测开销须由 runtime 开关、源版本和实验 manifest 单列，配置 hash 本身不能证明运行时开销相同。

```json
{
  "event": "request_terminal",
  "attempt_id": "原有 attempt/RID",
  "admission_diagnostic": {
    "version": "native_admission_diagnostic_v1",
    "request_id": "原有 req.rid",
    "clock": "server_perf_counter_seconds",
    "timestamps": {
      "scheduler_recv_time": 100.0,
      "forward_entry_time": 101.0,
      "prefill_finished_time": 101.1,
      "completion_time": 103.0,
      "wait_queue_entry_time": 100.1
    },
    "queue_duration_s": 0.9,
    "gates": {},
    "adder_outcomes": {"accepted:NO_TOKEN": 1}
  }
}
```

该 JSON 仅说明字段，不是测量样本。时间戳直接复制 `req.time_stats` 原值；未设置的零值导出为 `null`，不在终态重新取时钟替代它们。这些值是同原生 scheduler 的 `perf_counter` 秒，不能与客户端、wall clock 或 CUDA event 时间直接相减。`request_terminal.observed_time_ns` 仍是较晚的 owner 观测时刻，不能替代原生 completion。

正常执行中 `forward_entry_time` 保留首次值；`wait_queue_entry_time` 是最近入队，重入时可能晚于首次 forward。`queue_duration_s` 只累加**已经闭合**的等待 episode，包含原生 cache preparation；等待中 abort 的最后一个 episode 尚未进入累计值。原生 quick-finish 路径还会把 forward 字段置为 completion，并不证明真正执行过 forward。分析必须保留 receipt 的 abort/finish 状态；不能从累计 queue 值推导纯 compute、DMA 或全部等待时间。[固定时间实现](https://github.com/sgl-project/sglang/blob/3f36fc08331f64cd63c4fbbdf39cc402f38bb38b/python/sglang/srt/observability/req_time_stats.py)。

## 真正的 gate 与接受状态

`gates` 以有限原生 reason 名称保存 `{count, first, last}`，只更新计数和末条标量快照，不保留无界 attempt 列表。`observed_at` 是该 gate 的服务器 `perf_counter` 秒；它不表示此原因持续到下一次观测。

| 观测 | 含义 |
|---|---|
| `prefill_budget_accepted/rejected` | 实际 `_check_prefill_budget` 返回结果；仍在原生 prefix 临时 pin 内。保存总需求、extend、所考虑 host hit、选定 chunk 上限 |
| `prefill_request_limit`、`prefill_delayer`、`load_back_budget/unavailable` 等 | 对应原生 return 分支；不把所有 OTHER 合成容量不足 |
| `adder_accepted/not_accepted` | add_one_req 返回后，本次 can_run_list 增长且最后成员正是该 Req 才算 accepted |
| `scheduler_pass_budget`、`scheduler_request_slots_after_admission` | 已创建 adder 的 pass，在原生 break 处采样；没有伪称调用过 add_one_req |
| `scheduler_batch_full/request_slots/min_free_slots/prefill_deferred` | 尚无 adder 的早期 gate，只采当时 queue 首项，scope 为 scheduler_pass_head_observation，budget 为 null；不遍历归因其余队列 |

实际预算含 `remaining_total/current/input/chunk`、`total_offset/current_offset`、`page_size`、`new_token_ratio`、`admitted_requests`、`budget_type`，以及 Req 的输入/已生成/预算长度。`full_fill_tokens` 和 prefix/host 是原生当时值；pass gate 可能早于首次 match，不能当新鲜 cache quote。`considered_host_hit_tokens` 是该次容量选择实际传入的值；load-back 后重试可能与 Req 原来的 host hit 不同。

`adder_outcomes["accepted:NO_TOKEN"]` 或 `accepted:OTHER` **不是失败**：它可能表示本 Req 已提交后，剩余预算使下一候选应停止。`not_accepted:*` 才是该次未加入。一次请求可有多次检查或重入；这些不是独立请求数或所有 prefill chunk 数。继续执行的 chunk 走 add_chunked_req，不经过本次 add_one_req outcome。grammar 等未覆盖入口、队列中未访问成员不补造原因。[固定 adder](https://github.com/sgl-project/sglang/blob/3f36fc08331f64cd63c4fbbdf39cc402f38bb38b/python/sglang/srt/managers/schedule_policy.py)、[固定 scheduler](https://github.com/sgl-project/sglang/blob/3f36fc08331f64cd63c4fbbdf39cc402f38bb38b/python/sglang/srt/managers/scheduler.py)。

诊断不新增树遍历、缓存锁、内容 hash、GPU 同步或每 token 事件。预算读取使用原生只读属性，不重新计算树内 occupancy，也不把 logical KV 长度当 unique slots。它是粗 gate 证据，不是完整原生调度重演或等待时长归因。

## 验证与 fixture 入口

CPU 测试提取固定源码并应用本仓实际 patch，覆盖严格容量边界、页取整后成功加入却返回 NO_TOKEN、成功加入后 OTHER、真正容量失败、host-load 不可用、重复检查的有界记录、默认关闭、终态原时间/仅发一次和重入语义。原/新 native 返回、成员数量、剩余预算保持一致。`scripts/prepare_sglang.py --source-only` 核原文件 hash、实际 patch 应用、14 个变更文件语法和 outcome hook 次序。这不是 GPU 验收。

短功能入口复用已有模块，不启动服务：

```sh
python -m engine_patches.observation_probe \
  --config configs/hardware_stage_controller.yaml --base-url http://127.0.0.1:30001 \
  --model-path /绝对路径/Qwen模型 --cache-state gpu_ready \
  --report runs/admission-functional.json
```

调用者使用独占、显式开启两个开关的 16K 研究服务，并检查 metrics_after 中三个实际 RID 的 terminal diagnostics。warm/retire 请求也可能在日志内，不能混入三请求集合。

两 peers + A/R 候选使用研究目录 `runs/peer-interference-prep/plan.json` 及同目录 runner 的冻结合同：固定外生 100 ms candidate 相位，不以首 token 自适应；记录实际 native receive 顺序、peer 仍在运行的覆盖、两臂输入/输出预算和初态。缺覆盖、失败及相位误差保留。先测 own/peer 完成增量，再决定模型；不能将固定 ignore-EOS 机制实验当普通 EOS 或质量验证。
