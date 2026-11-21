# SGLang 上下文与 KV 执行

`/generate` 的 `contextkv_compute` 元数据随实际输入一次提交，默认精确执行仍接收明确的后续消费者需求：

```json
{
  "session_id": "session-1",
  "request_id": "turn-1:read",
  "cache_scope": "shared",
  "mode": "exact",
  "blocks": [{"source_id": "document/version/span", "start": 48, "end": 2096}],
  "demand": [{"source_id": "document/version/span", "consumer_ids": ["next-read"], "uses": 1.0, "native_prefix_end": 2096}]
}
```

`start/end` 是实际输入 token 的半开区间，块只包含稳定原文。来源 ID 由文档、版本和原文地址组成。当前顺序与已知后续读取共享字面前缀时，`native_prefix_end` 绑定实际 Native 前缀。需求随下一次提交更新；`/context/session/close` 清除会话需求，页面继续归原生缓存管理。

`/context/evidence/state` 是只读快照：接收候选地址，返回 Native 前缀、驻留、容量与实测费用系数，不编译提示或分配 KV。控制器据此选择合法来源的执行顺序，引擎准入时按实际 token 和最新容量重新判断。

## 默认执行与保留

- Native 使用 SGLang 原生执行及 Radix／HiCache。总体可行但暂时空闲不足的请求进入排队；`kv_cost.py` 保留执行和等待成本的同一口径。
- `exact` 与语义需求提示独立。最终回答不生成基于访问次数的软保留概率；只有已明确的后续消费者改变特殊优先级。
- 需求更新、请求结束不触发全局保留扫描或主动下放。原生分配器实际回收 GPU／Host 页时临时应用需求优先级，回收后恢复原优先级。
- Native 保留按真实前缀及依赖计价。GPU 有 Host 后备时节省回载，无后备时避免重算；共享页按物理节点去重。保留已有页面不重复收取已经支付的构建费用。

`meta_info.contextkv_compute` 返回真实输入／输出、prefill、GPU 命中、Host 回载与服务耗时。只记录必要费用统计，不记录正文或逐 token 轨迹。

## 显式 InfoFlow 实验

`infoflow.py` 与 `infoflow_attention.py` 保留独立块 prefill、全局 RoPE 恢复、注意力打分、稀疏重算和分段解码。共享块使用同一 Radix／HiCache 页池，重算写入请求私有页，不登记为精确前缀。启用 `auto` 或 `infoflow` 需单独提供校准资产；默认 `exact` 不构建独立块，不加载该质量资产。全局质量均值不能保证单题无损，历史结果见[实测报告](../docs/study/results.md)。

## 准备

集成对应 SGLang 0.5.23（`297e54fd105e974cdf401e5b4d20253a22c16f90`）。准备脚本定位默认 `python3` 已加载的源码并应用补丁：

```bash
bash scripts/bootstrap_sglang.sh
source runs/engine-runtime.env
bash scripts/launch_sglang.sh
```

`--source-only` 仅下载源码和检查补丁语法。方法归属见 [NOTICE](NOTICE.md)，项目入口见 [架构](../docs/architecture.md)。
