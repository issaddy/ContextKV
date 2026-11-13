# SGLang 上下文与 KV 执行

`/generate` 的 `contextkv_compute` 元数据随实际输入一次提交：

```json
{
  "session_id": "session-1",
  "request_id": "turn-1:answer",
  "cache_scope": "shared",
  "mode": "auto",
  "blocks": [{"source_id": "document/version/span", "start": 48, "end": 2096}],
  "demand": [{"source_id": "document/version/span", "consumer_ids": ["next-answer"]}],
  "recompute_ratio": 0.15,
  "quality_loss": 0.0,
  "quality_price_ms": 0.0
}
```

`start/end` 是实际输入 token 的半开区间，块只包含稳定原文。来源 ID 由文档、版本和原文地址组成。需求随会话的下一次提交更新；`/context/session/close` 在会话结束时释放该会话的需求优先级。当前编译的原文顺序与后续需求共享字面前缀时，需求附带 `native_prefix_end`，同时登记 Native 的前缀消费者。

`/context/evidence/state` 一次接收当前候选的来源地址，返回独立块的 GPU/Host 驻留、可用 Native 前缀顺序和实测费用系数。此接口只读取 owner 状态，不编译提示或分配 KV。控制器用同一快照为候选集合计价，SGLang 在实际提交时按当前驻留完成执行选择。

相同 `cache_scope` 内，独立块按实际 token 前缀共享原生页；训练冷分支使用独立 scope，暖分支共用预建块所在的 scope。请求私有重算不会写回这些共享页。

## 执行

- `infoflow.py` 使用 SGLang 已加载的 Qwen3 权重，执行独立块 prefill、全局 RoPE 恢复、末两层 prompt-to-context 注意力评分、因果稀疏重算与解码。
- `infoflow_attention.py` 把页读取、位置恢复和因果 attention 融入同一 Triton 算子，解码不复制完整 KV。
- `contextkv_compute.py` 把冷建、打分和重算编成内部工作段，交给原生 chunked prefill 分批执行。共享块使用同一套 Radix/HiCache 页，打分和重算结果使用请求私有页；请求释放时，共享块页继续归原生缓存管理。
- 精确请求沿原生执行；非前缀请求的生成 KV 不登记为精确前缀。`auto` 先取得原生费用观测，再按当前命中、Host 搬运、冷建、打分、重算和质量损失选择执行方式。
- 后续独立块复用按相对 Native 的边际 prefill 收益估值：每 token 的正收益为 `max(0, native_prefill_ms − recompute_ratio × replay_ms)`。Native 精确前缀按节省的 prefill 计价。同一未来消费者只选择收益较高的一种表示，共享页容量按实际 GPU 或 Host 页计一次，保留优先级随实测成本更新。GPU/Host 容量、搬运和淘汰仍由原生缓存管理。
- 输出长度估计只使用正常结束的生成；达到长度上限的 warmup 或截断输出不作为短答案样本。

`meta_info.contextkv_compute` 返回真实输入/输出 token、prefill、GPU 命中、Host 回载、完整服务耗时及必要费用均值。内部工作段长度与语义输入长度分别计量；不记录正文或逐 token 轨迹。

## 准备

集成对应容器现有 SGLang 0.5.23（`297e54fd105e974cdf401e5b4d20253a22c16f90`）。准备脚本定位默认 `python3` 已加载的 SGLang 源码并原地应用补丁：

```bash
bash scripts/bootstrap_sglang.sh
source runs/engine-runtime.env
bash scripts/launch_sglang.sh
```

准备脚本应用源码补丁并复制执行模块。`--source-only` 仅下载相关源码和检查补丁语法；正式运行直接使用容器已安装的源码。

方法依据与归属见 [NOTICE](NOTICE.md)，项目入口见 [架构](../docs/architecture.md)。
