# 16K / 32K 有限容量控制器开发验证

`python -m bench.controller_capacity_sweep` 默认只打印计划；不创建结果目录、不连接引擎、不加载模型、不启动或停止进程。只有显式 `--execute` 才在**当前主机**运行。工具没有 SSH 能力，远端操作由操作者在目标服务器执行。

入口复用 `bench.native_policy_sweep` 的 launch argv、readiness 和 owned-process-group 清理，以及 `bench.experiments` 的 job、manifest、事件、审计、遥测和结果汇总。每个容量/重复组使用一个新 patched SGLang 进程，组内各方法通过已有 `scripts.prepare_cold_job` 清空 KV；不重写在线 planner，不自动拟合或修改成本 artifact。

## 固定设置与方法

默认 GPU KV 池分别为 16384、32768 tokens，由真实 `--max-total-tokens` 启动参数控制。模型 context window 仍为 32768、输出上限 1024，Cloud dev、P2、c4、seed42。host 配置固定 16 GB、cache mode、write-through、layer-first、kernel backend、TMA transfer=0、page64；启用 profiling 与 `CONTEXTKV_SAMPLE_POOL=1`。启动后记录 native runtime、实际 GPU/host token slots，并检查两个容量组的实际 host slots 相同。配置里的 mock capacity 默认字段不控制 SGLang，不能当作实际分配值；实际值来自启动后的 runtime/pool 记录。

默认六个固定配置：

| 方法 | 文本策略 | 替代旧 epoch 的退役 |
|---|---|---|
| Ours | 已给定参数的 cost planner | 开启 |
| Ours_no_retire | 与 Ours 完全相同 | 关闭 |
| Fixed_4096 | APPEND 超过 4096 tokens 则 REBUILD | 开启 |
| Fixed_8192 | APPEND 超过 8192 tokens 则 REBUILD | 开启 |
| Rebuild | always_rebuild | 开启 |
| Rebuild_no_retire | 与 Rebuild 完全相同 | 关闭 |

两对 retirement 消融仅改变 `ablation.epoch_retirement`，继承同一 base config 的其它参数，不单独调参。关闭替代退役不禁止正常会话结束或错误恢复时关闭引擎 session。提示词、材料策略和其它可选成本设置仍来自明确给定的 base config；默认示例为 Q0。工具不添加 quality eligibility 或 B6 eligibility。

## 哪些可以直接执行

已有完整 patched checkout、固定本地模型及可用 Linux/GPU 环境时，固定策略和其退休消融不需要成本标定。以下命令先打印计划；确认目标端口没有另一个服务后，操作者可加 `--execute`：

```bash
RAYON_NUM_THREADS=4 python -m bench.controller_capacity_sweep \
  --config configs/hardware_dev_controller.yaml \
  --model-path models/Qwen2.5-7B-Instruct \
  --checkout vendor/sglang \
  --gpu-tokens 16384 32768 \
  --methods Fixed_4096 Fixed_8192 Rebuild Rebuild_no_retire \
  --output runs/controller-capacity-fixed-v1
```

`--methods` 可选任意子集；若不包含 Rebuild，现有矩阵 runner 会输出空的参考比较列表，单方法原始结果和 summary 仍保留。默认 `--repetitions 1`，可以增加重复；每次固定 seed42，并轮换容量顺序和方法顺序。该轮换不是随机化重复实验，单次运行也不能消除顺序影响。

服务子进程的 `RAYON_NUM_THREADS=4` 与控制器父进程是两份环境。工具不修改父进程环境，plan 记录其实际启动设置；因此命令应显式写上前面的 `RAYON_NUM_THREADS=4`。记录的是线程预算设置，不是实测活跃 worker 数。server 环境另外保存在每组计划中。

## Ours 必须使用各容量自身的标定

原 `cost-dev-expanded-lookup.json` 的 profile manifest 声明 GPU 容量 **121920**，不能作为 16K/32K 的校准来源。plan 会显示 `requires_calibration`；若包含 cost 方法，执行会在加载模型和启动 GPU 服务前拒绝这个已知不匹配配置。固定策略的生成配置会移除无需使用的成本 artifact。

每个小池标定完成后，分别绑定对应文件：

```bash
RAYON_NUM_THREADS=4 python -m bench.controller_capacity_sweep \
  --config configs/hardware_dev_controller.yaml \
  --model-path models/Qwen2.5-7B-Instruct \
  --checkout vendor/sglang \
  --gpu-tokens 16384 32768 \
  --cost-artifact 16384=runs/capacity-calibration/16384/cost-lookup.json \
  --cost-artifact 32768=runs/capacity-calibration/32768/cost-lookup.json \
  --output runs/controller-capacity-all-v1
```

这仍是 plan-only，追加 `--execute` 才运行六方法。静态容量声明吻合仅得到 `requires_runtime_scope_check`，不是自动认可：服务启动后使用原 `CostModel.check_calibration_scope`，要求 fingerprint 与实际 `cache_config_hash` 都是 `matching_scope`。同时核对真实 pool、host layout/backend/write policy、TMA 与 sample_pool 设置。不允许改写旧 manifest 的容量/hash 来通过检查。

`matching_scope` 只确认引擎配置身份，不证明 GPU/driver 相同，也不保证每个输入形状都有充分标定或每个动态 quote 能保持到执行时。每组 runtime 包含真实硬件/软件记录供核查；逐轮 calibrated/fallback、错误与实际 cache receipt 继续保留。有限池下请求可能排队、回收、重执行或失败，不能以静默缩短上下文、去掉 required 材料或丢弃失败轮次获取结果。

## 小池采样的物理可行性

当前 `profiling.sglang_probe` 校验模型 context limit，但**没有预检 GPU 池总量**。`prepare()` 严格要求每个并发请求观察到指定 ready-GPU/host prefix；没有因为池小就放松这一合同。本次 runner 没有扩展采样器。

每个请求至少需要满足 `prompt + output < pool` 的保守预算；GPU-ready 的预热还要求所有并发 prefix 加 sentinel/页对齐开销能驻留。16K 下 `c4 × 8192 × .99` 或 `c4 × 12288 × .99` 的 GPU-ready fixture 不可能同时驻留，因此不能直接跑旧大池完整 Cartesian grid。host-only 的前缀可以通过 native 排队、写回和驱逐积累到 host，但仍须等真实 ACK、最终 quote 验证，而不是由“曾预热”推断已驻留。

本次开发采用的分组建议如下，两种容量都保留同一组配置以便比较：

| 组 | prompt lengths | concurrency | prefix fractions | cache states |
|---|---|---:|---|---|
| single | 1024,4096,8192,12288 | 1 | 0,.5,.99 | cold,gpu_ready,host_only |
| concurrent-small | 1024,4096 | 4 | 0,.5,.99 | cold,gpu_ready,host_only |
| concurrent-long-device | 8192,12288 | 4 | 0,.25 | cold,gpu_ready |
| concurrent-long-host | 8192,12288 | 4 | .5,.99 | host_only |

每组输出长度 128/512、每 case 一次 warmup、三次正式重复。16K 中最紧的 GPU-ready 单元为 `n4096,pf.99,c4`：页对齐 prefix 是 4032，四份 prefix=16128，warm sentinel/输出按 64 页取整最多正好占满 16384，余量很小。若实际临时页需求使 fixture 不匹配，保留失败记录，再明确修改该单元为更低 fraction 或从计划中移除，不能改验证器把它伪装成成功。c4 的正式生成总工作集可以超过池容量；此时真实 TTFT 包含 native 调度压力，不表示四条请求都同时完整驻留。

每组用现有 profiler CLI 采样，例如第一组：

```bash
python -m profiling.sglang_probe \
  --config runs/capacity-calibration/16384/config.yaml \
  --runtime-manifest runs/capacity-calibration/16384/runtime.json \
  --output runs/capacity-calibration/16384/single.jsonl \
  --lengths 1024,4096,8192,12288 --prefix-fractions 0,.5,.99 \
  --outputs 128,512 --concurrency 1 --cache-states cold,gpu_ready,host_only \
  --warmup 1 --repetitions 3
```

先在对应容量的专用服务上保存 `profiling.runtime_manifest --research`，再采样；这些命令不会由 sweep 隐式执行。合并同容量各组 measurements 后，`profiling.fit --profile-manifest` 为每份原始 manifest 重复传入一次。它要求 measurement run IDs 与 manifests 完整对应、共同 fingerprint/cache hash 一致；16K 与 32K 不能混合拟合为同一身份。输出长度 prior 可继续显式引用已有 dev generation report，不能从当前答案即时拟合。

## 初态、输出和解释边界

原生 `native_policy_sweep` 为每个策略启动全新进程并 skip warmup；这里每个容量/重复组新建一个进程，方法间只做 KV reset，JIT/kernel 状态会共享。before-job cold 报告验证 GPU/host unique slots 为零、无活动或排队请求，但这**不等于全新进程初态**。计划明确记录 `native_initial_state_matched=false` 和 `b6_comparison_established=false`；不能把两种入口的结果直接称为匹配的 B6 对比。

结果目录保留 `plan.json`、每组 `controller.yaml`/`matrix.yaml`、owned PID、startup log、runtime-before/after、scope、group 状态，以及 `matrix-results` 下的完整 benchmark/manifest/events/audit/summary。执行只停止自己创建的 process group，端口已占时拒绝使用，不查找或终止外部服务。过程失败和 generation failure 不会被删除；任一组有 generation failure，仍完成已计划组的记录与清理，最终 CLI 返回非零。原生 policy sweep 同步采用这一退出码语义，避免“保存失败结果但 shell 显示成功”。

有限容量 C 机制应结合相同预算下的实际 device/host hits、重执行/失败、DMA 和 unique-slot 积分观察；退役 ACK 不等于物理槽位释放。两种 P2 运行的生成答案与后续历史也可能分叉，应保留输入/动作/输出一致性及质量审核。此入口提供可复现的机制诊断，不自动产生质量通过、一般吞吐收益或最终论文结论。
