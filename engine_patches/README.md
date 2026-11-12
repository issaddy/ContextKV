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

`start/end` 是实际输入 token 的半开区间，块只包含稳定原文。来源 ID 由文档、版本和原文地址组成。需求随会话的下一次提交更新，最终空需求释放软优先级。

相同 `cache_scope` 内，独立块按实际 token 前缀共享原生页；训练的同题候选共用一个 scope。请求私有重算不会写回这些共享页。

## 执行

- `infoflow.py` 使用 SGLang 已加载的 Qwen3 权重，执行独立块 prefill、全局 RoPE 恢复、末两层 prompt-to-context 注意力评分、因果稀疏重算与解码。
- `infoflow_attention.py` 把页读取、位置恢复和因果 attention 融入同一 Triton 算子，解码不复制完整 KV。
- `contextkv_compute.py` 把冷建、打分和重算编成内部工作段，交给原生 chunked prefill 分批执行。共享块使用同一套 Radix/HiCache 页，打分和重算结果使用请求私有页；请求释放时，共享块页继续归原生缓存管理。
- 精确请求沿原生执行；非前缀请求的生成 KV 不登记为精确前缀。`auto` 先取得原生费用观测，再按当前命中、Host 搬运、冷建、打分、重算和质量损失选择执行方式。
- 后续消费者对同一来源的收益只计一次，保留采用原生软优先级。GPU/Host 容量、搬运和淘汰仍由原生缓存管理。

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
