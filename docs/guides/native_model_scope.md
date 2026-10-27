# 原生B0/B1对照入口

`bench.native_baseline`使用共享的Qwen3 `RunConfig.model/context/sampling`连接预先启动的stock SGLang。模型身份、支持范围及采样设置统一见[模型与采样](model_scope.md)；由程序管理服务器的方式见[矩阵入口](b6_owned_matrix.md)。

- **B0**：提交完整共享候选，不发送native session ID，关闭HiCache和SessionRadix，保留原生radix缓存。
- **B1**：同一会话使用稳定native session ID，每轮重建完整输入并保留实际QA；close是原生请求派发，资源后果另从观测记录。

## 调用

```bash
python -m bench.native_baseline data/mtrag/RAG.jsonl \
  --baseline B0 --base-url http://127.0.0.1:30001 \
  --run-config configs/qwen3_8b_scope.example.yaml \
  --engine-manifest /absolute/path/to/actual-native-manifest.json \
  --protocol P2 --domain all --split-manifest configs/mtrag_split_manifest.json \
  --split dev --audit-prompt-ids --output runs/native-local-scope-dev
```

`--run-config`提供模型、context policy和采样；`--base-url`及`--engine-manifest`指向已运行服务。显式context/output覆盖与服务manifest一致。历史manifest可离线读取，实际生成使用当前scope。

## 原生服务manifest

| 字段 | 内容 |
|---|---|
| `model.name` | scope的canonical model path |
| `model.revision` | 本地加载为null |
| `model.identity_kind` / `model.identity` | `local_snapshot_metadata` / scope identity |
| `model.tokenizer_hash` / `model.chat_template_hash` | 实际模板fingerprint |
| `model.dtype` / `model.kv_dtype` | bfloat16 |
| `model_scope` | `LocalModelScope.manifest()` |
| `sampling_recipe` | 实际四项采样或标准贪心`{"temperature":0}` |
| `server` / `runtime` | 实际引擎、TP/DP/PP、page/context、缓存flags、GPU/Host池及硬件环境 |

`verify_native_settings`核对server info中的模型/tokenizer路径、精度、加载方式及拓扑。GPU池来自实际`max_total_num_tokens`；Host容量由原生分配日志或指标提供。模板与模型解析复用`create_model_template`。

## 共同配置与输出

`bench.experiments`向原生基线和Controller传递同一ModelConfig、SamplingRecipe及context policy；方法间模型、输出要求和采样保持一致。原生基线直接走其执行通道，不使用Controller成本artifact。

结果保存`shared_configuration`、engine manifest、实际server info、模板fingerprint、HTTP参数、完整输入/回答、原文span、失败和截断信息。完整系统结果见[数据与评价](../data_and_evaluation.md)，源调用合同由`tests/test_native_baseline.py`和`tests/test_native_model_scope.py`覆盖。
