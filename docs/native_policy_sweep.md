# 原生服务器测量 helpers

固定模型的 `bench.native_policy_sweep` CLI已删除，不再启动7B写策略实验。正式执行统一使用显式Qwen3 [owned矩阵](b6_owned_matrix.md)，只读历史结果见[native_policy_validation.md](native_policy_validation.md)。

模块仍保留被当前代码调用的 `check_stock`、`server_command`、`pool_measurements`、`native_manifest`、`wait_ready`、`stop_owned_server`与结果汇总。新native_manifest必须给Qwen3 scope；池容量来自原生server info与allocation日志，显式head_dim用于KV字节换算。`summarize_native`仍能只读原始历史payload，缺失cache breakdown保留unknown，不改原件。

进程helper只清理自己启动的process group，不查找或停止其它服务。这里没有独立sweep、默认实验组合、模型下载或scope兜底。
