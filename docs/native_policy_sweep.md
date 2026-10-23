# 原生 HiCache 写策略开发比较

受版本控制入口为 [`bench.native_policy_sweep`](../bench/native_policy_sweep.py)。默认只输出计划，不访问服务、加载模型或创建实验目录；操作者用固定 stock SGLang 的隔离 Python 环境执行 `--execute`。已有 `runs/native-policy-sweep/run.py` 保留为早期 32K 实验的原始源码，不随模块推广修改。

```bash
# 先查看 16K 池的三策略计划。
python -m bench.native_policy_sweep \
  --model-path models/Qwen2.5-7B-Instruct \
  --checkout vendor/sglang-stock \
  --gpu-tokens 16384 \
  --output runs/native-policy-sweep/gpu-16384

# 自行停止旧服务后，对已查看的同一命令添加 --execute。
```

三策略为 `write_through / write_through_selective / write_back`，每次都启动全新 stock 进程，测量结束只关闭该脚本创建的进程组。发现已有监听端口会停止，不会寻找或终止其他服务器。没有 SSH、研究 quote/prepare/retire 接口或 patched-engine 伪装路径。

共同设置为：固定 SGLang `3f36fc08331f64cd63c4fbbdf39cc402f38bb38b`、Qwen2.5-7B-Instruct 固定 revision、GPU 0、BF16、TP=1、page size 64、host 16GB、kernel/layer_first、显式 `SGLANG_HICACHE_TMA_TRANSFER=0`。脚本一次检查 stock HEAD、Python 源码无修改、无 ContextKV marker/runtime 文件，并确认 Python 实际解析到该 checkout。启动仍复用 `scripts/launch_sglang.sh`，设置 `SGLANG_CHECKOUT=vendor/sglang-stock`。

`--gpu-tokens` 默认 32768，可设 16384，必须为正的 64 倍数；它直接进入服务启动的 `--max-total-tokens`。**模型上下文仍固定 32768**，不因为池变小就偷偷裁短输入或减少输出预算。GPU 池总容量和模型上下文上限不是同一配置；小池并不保证任意 32K 单请求可执行，固定工作负载中的实际失败会保留。

负载固定 native B1、P2、Cloud dev 5 会话/41 轮、客户端并发 4、输出 1024、Q0、greedy、seed 42，默认 split 为 `data/mtrag/split_manifest.json`。不读 test，不更换问题或回答上限。每次使用 `--skip-server-warmup`，没有生成预热；CUDA graph 初始化使用同一原生设置。默认每策略一次是开发诊断，`--repetitions 3` 会轮换先后顺序。KV 新池为空不代表操作系统文件缓存和编译缓存被清空，首轮效应仍需结合日志解释。

实际容量必须有观测依据：GPU slots 读取 `/get_server_info.max_total_num_tokens` 并与本轮请求值匹配；host slots 来自该进程启动日志的实际 allocation 记录，缺记录时停止。host bytes 由实际 slots 与本地固定模型的 TP1 BF16 KV 几何计算，注明不含 allocator overhead，不能直接把“16GB”声明当测量值。原生 `meta_info.cached_tokens_details.device/host` 汇总真实分层缓存命中；固定源码规定零命中可返回 None，只有 `cached_tokens==0` 才计两层为零，正命中而缺分层记录保持未知。

各 job 保留 startup/benchmark 日志、真实 manifest、前后 server info、Prometheus metrics、完整 native/audit/official 输出以及汇总；根目录记录每组状态。复用现有客户端 SSE 端到端计时，补充等会话完成均值、输出总量、截断/失败、device/host hit tokens。没有 native receipt 时不伪造 actual Prefill、物理 DMA bytes 或瞬时峰值；native close 也不冒称 scheduler retire ACK。

测试覆盖：plan 无副作用、32K/16K 实参及不变模型上下文、真实容量不符/日志缺失、stock purity 边界、owned-process cleanup、分层命中缺失不记为零。单次 P2 的输出/历史会分化，Host load 为零应报告为机制未激活；速度结果不能直接宣布同质量最优或 B6，质量和重复测量需另行验证。

固定上游合同：[写策略参数](https://github.com/sgl-project/sglang/blob/3f36fc08331f64cd63c4fbbdf39cc402f38bb38b/python/sglang/srt/arg_groups/fields/memory.py)、[host allocation 日志](https://github.com/sgl-project/sglang/blob/3f36fc08331f64cd63c4fbbdf39cc402f38bb38b/python/sglang/srt/mem_cache/pool_host/base.py)、[native cache breakdown](https://github.com/sgl-project/sglang/blob/3f36fc08331f64cd63c4fbbdf39cc402f38bb38b/python/sglang/srt/managers/scheduler_components/output_streamer.py)。
