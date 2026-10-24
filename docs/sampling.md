# 显式采样与标定来源

研究 Controller、`SGLangEngineClient` 和真实 profiler 现在可使用同一个不可变 `SamplingRecipe`。默认仍是 `sampling: null`：`/generate` 实际发送 `temperature: 0` 与本次 `max_new_tokens`，不新增 `top_p/top_k/min_p`。模型 scope 不会自动启用 Qwen3 推荐采样，已有 7B、B0/B1/B6 原始记录不改写。

在已有配置中显式增加：

```yaml
sampling:
  temperature: 0.7
  top_p: 0.8
  top_k: 20
  min_p: 0
context:
  max_new_tokens: 1024
```

这四个字段在配置构造时检查有限数值与范围；没有任意参数字典。生成预算仍只来自 `context.max_new_tokens` / 单次 `generate(max_new_tokens=...)`；`ignore_eos` 仍由 profiler 等调用者按原协议选择。没有新增 request seed 或确定性执行承诺。服务端进程 seed、执行顺序、运行环境应照原实验 manifest 记录；profiler 的 `--seed` 只控制合成输入，并非 sampler seed。

`sampling: {}` 是有明确配置默认的标准贪心配方，会显式发送四项。只有完整标准配方 `(temperature=0, top_p=1, top_k=-1, min_p=0)` 与旧 `None` 视为等价；不推广其他看似等价的参数组合。固定 SGLang 在 `SamplingParams` 中将零温度归一为贪心 top-k=1，并验证各项范围；执行 sampler 时贪心走 argmax，随机采样可走温度缩放、softmax 与过滤路径。[固定 SamplingParams 源码](https://github.com/sgl-project/sglang/blob/3f36fc08331f64cd63c4fbbdf39cc402f38bb38b/python/sglang/srt/sampling/sampling_params.py#L156-L227)、[固定 Sampler 源码](https://github.com/sgl-project/sglang/blob/3f36fc08331f64cd63c4fbbdf39cc402f38bb38b/python/sglang/srt/layers/sampler.py#L191-L283)。

采样改变下一 token 的分布；对已经确定的输入 token 和权重，原生仍可复用其 KV 前缀。因此不修改 model fingerprint、cache key 或 cache_config_hash，也不因为换配方清空缓存。需要重新核对的是 sampler 所在的执行时间区间与输出长度分布。

## 三类数据分别处理

| 数据 | 新配方处理 | 仍保留什么 |
|---|---|---|
| 输入/实际输出长度上的执行时间点 | `configure_sampling` 在初始化时核 profile manifests 的配方。不同配方或非贪心配方遇到未知来源，取消 timing 校准/selection eligibility | 原始 prefill/decode 点和诊断预测，不重写 artifact，不改时间数值 |
| artifact 的全局 mean/P90 及 stage 的 development lengths | 独立核 `output_prior_sampling` / distribution 自身 `sampling_recipe`。不匹配时 mean/P90 使用配置先验，stage 不传旧经验长度 | 配置先验和实际已提交历史仍是启发式；native waiting 的既有 known-budget fallback 及 reason 保留 |
| 可选 learned output-length 模型 | 启动核 fingerprint、budget、target recipe / 已绑定 source configuration，以及已记录的 global prior 残差基准；不匹配直接拒绝加载 | 未加载模型时的共享先验路径；不把未知来源称新分布已校准 |

当前时间点是 server TTFT、forward→first token、decode 等 **wall 区间**，不是移除了 sampler 的 transformer CUDA 内核成本。按 token 长度建表不证明更换 sampler 后其时间仍准确，因此旧点可诊断复用而不能直接取得新配方的校准资格。该检查与原 engine/cache scope 独立：服务重启后的 scope 核对不会恢复已取消的采样资格。配方相同也不证明硬件、负载、提示词、输出分布或质量相同。

历史数据缺少采样字段，以及空的离线 `{}` 记录，都保持 `unknown/legacy_unbound`。为保持原工程行为，标准贪心运行仍可加载旧单 artifact，但报告明确这是兼容例外，不是从缺失字段证明其采样来源。非贪心不能使用这个例外。离线显式记录只接受实际当前完整四项，或历史实际 wire `{"temperature": 0}`，不从省略的 temperature 推测执行配方。

真实 benchmark 输出增加 `sampling_recipe`，使仅接收 completed-report 文件的 `profiling.fit` 能把经验长度配方与原始文件绑定。报告缺该字段仍为未知；不拿 profiler 的固定输出配方冒充正常 EOS 的输出分布。`length_fit` 则从已经冻结的 pairs manifest 的真实 config 记录目标配方，并复用服务端的配方兼容处理选择实际共享 mean/P90，而非无条件继承旧 artifact 的经验先验。模型残差围绕该基准训练；加载时用已有 `metadata.global_priors` 与服务实际先验比较，防止配方相同却中心不同。旧模型缺中心仅保留标准贪心兼容；非标准配方要求明确的中心绑定。后续旧 stage 细化仍保留原 development-report 来源与分布配方。

## 采集、拟合与执行边界

`profiling.sglang_probe --config ...` 消费同一配置，manifest 记录实际请求配方与 `ignore_eos: true`、逐 case 输出预算。该固定输出采集并不估计自然 EOS 长度。`profiling.decision_pairs` 的实际请求、在线/离线成本组装也消费该配方；模型校准报告保留独立的 `sampling` 子项。离线 `decision_pairs rescore` 在加载模型前一次核对原测量 config 的采样配方；不可把旧配方的实际时间/输出当作新配方 outcomes 计算 regret。完整旧 config 未声明采样有既定贪心语义；整个来源 config 缺失则记录未知兼容，仅允许标准贪心重评分。

多 `--profile-manifest` 合并必须同 fingerprint/cache 配置且配方一致。全部旧未知来源可继续原有合并，但新产物仍标未知；未知与显式来源不能混合。`stage_fit` 也拒绝把不同/未知与显式配方的旧 decode 表和新 forward 点拼成同配方表。所有绑定沿用现有 manifest 和文件来源记录，在离线或初始化边界完成，没有新增逐请求 hash。

Controller 的 `request_submitted` 事件记录由同一配置构造的实际请求参数及本次输出预算。流式与非流式共用同一 payload；completion receipt、原生 admission、失败恢复、模型 token 和 full-QA 合同不变。MockEngine 不模拟采样或质量；其事件只记录请求设置，simulation 标签保持。

本次没有扩展 native benchmark 的正式采样协议。包含实际 `kind: native` 路径的实验矩阵，在启动任何 job 前拒绝非标准贪心，包括 controller candidate 单独覆盖采样的情况，避免与仍贪心的 native 组混比。纯 Controller 矩阵可以使用显式配方；两个 owned cost preflight 会在生成前拒绝采样不匹配的成本 artifact。成本点保留供离线诊断，不能据此让 Ours 静默全回退后仍声称运行了已标定策略。

当前新增验证是本地 CPU/HTTP 合同：默认 wire、显式四项、stream/nonstream、预算唯一来源、事件/collector manifest、旧/新配方、未知来源合并、length 模型、owned preflight 与 native 矩阵拒绝。已完成的 a876 Qwen3 硬件 scope probe 没有部署这次采样代码，不能作为这条新 research client 接线的 GPU 验收。
