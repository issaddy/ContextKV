# Qwen3模型身份与采样配置

Controller、原生基线和profiler共用`RunConfig.model.scope_path/context/sampling`。真实服务使用已有本地Qwen3和非思考模板；模型启动命令见[部署](../deployment.md)，原生对照调用见[Native B0/B1](native_model_scope.md)。

## 模型身份与加载

`engine_patches/contextkv_model_scope.py`统一读取本地scope。当前记录为：

- `configs/model_scopes/qwen3-8b.local.json` → `/media/ssd1/qwen3-8b`
- `configs/model_scopes/qwen3-32b.local.json` → `/media/ssd1/qwen3-32b`

支持dense Qwen3、BF16、TP/DP/PP=1及现有无SWA/RoPE扩展配置。scope由模型config、分片库存/大小、tokenizer及模板形成，`local-snapshot:`表示元数据身份；`hf_revision:null`、`weight_content_verified:false`记录其验证范围。32B的`head_dim`按config中的128计算。

配置初始化解析一次scope，加载时核对本地文件、分片和tokenizer；`create_model_template`以`revision=None`加载，身份单独进入fingerprint。引擎启动后检查实际路径、精度、拓扑和模板。scope文件路径是记录位置，canonical model path与identity用于关联实际模型。

[8B共享配置](../../configs/qwen3_8b_scope.example.yaml)与[32B共享配置](../../configs/qwen3_32b_scope.example.yaml)提供单次REBUILD入口；[8B成本配置](../../configs/qwen3_8b_cost.yaml)接入现有H200阶段标定。模型、缓存池和采样来源的匹配规则见[阶段成本](stage_cost_model.md)。

## 采样配置

默认`sampling: null`发送`temperature: 0`及本次`max_new_tokens`。Qwen3示例显式使用以下配方：

```yaml
sampling:
  temperature: 0.7
  top_p: 0.8
  top_k: 20
  min_p: 0
context:
  max_new_tokens: 1024
```

`SamplingRecipe`接受四个采样字段，生成预算来自`context.max_new_tokens`；`ignore_eos`由具体采集协议决定。`sampling: {}`显式发送标准贪心四项`0/1/-1/0`，在配方比较时与null等价。服务端进程seed和负载seed分别记录；profiler的`--seed`控制合成输入。

固定SGLang在零温度下使用贪心argmax，其他配方按温度及过滤规则采样，见[SamplingParams](https://github.com/sgl-project/sglang/blob/3f36fc08331f64cd63c4fbbdf39cc402f38bb38b/python/sglang/srt/sampling/sampling_params.py#L156-L227)和[Sampler](https://github.com/sgl-project/sglang/blob/3f36fc08331f64cd63c4fbbdf39cc402f38bb38b/python/sglang/srt/layers/sampler.py#L191-L283)。采样改变下一token分布，但相同输入与权重的KV仍可复用，因此采样配方单独记录，保持原model/cache指纹。

## 时间与长度数据的匹配

| 数据 | 装配行为 |
|---|---|
| Prefill/Decode等实测时间点 | `configure_sampling`核对采集配方；不匹配的点保留为诊断预测，取消对应校准/选择资格 |
| 全局mean/P90与开发输出长度 | 分别核对`output_prior_sampling`及分布的`sampling_recipe`，回答协议另核`output_prior_system_prompt`；不匹配时使用配置先验和已完成历史 |
| 可选学习式长度模型 | 加载时核对模型身份、输出预算、目标采样和`metadata.global_priors`残差中心 |

时间表记录实际wall区间，包含其执行时的sampler成本。运行报告保存实际`sampling_recipe`，自然EOS输出形成经验长度分布；profiler的`ignore_eos`固定输出用于时间测量。

历史缺字段或空采样记录标为`unknown/legacy_unbound`，标准贪心按既有兼容规则加载旧artifact并保留来源状态。来源绑定完整的测量按实际配方比较；多组profile合并使用同一模型、缓存配置和采样配方。

`length_fit`使用冻结pairs中的目标配方及对应共享先验拟合残差；`decision_pairs rescore`按完整生成协议复用已测输出。配方、来源及残差中心在加载或离线边界检查，流式与非流式共用同一请求构造。

## 相关入口

- [部署](../deployment.md)：准备环境和启动生成/解释服务。
- [Native B0/B1](native_model_scope.md)：原生服务manifest与CLI对照。
- [自有服务器矩阵](b6_owned_matrix.md)：共同配置、资源和消融装配。
- [阶段成本](stage_cost_model.md)、[成对决策验证](decision_pairs.md)：采集、拟合与已保存结果复算。
