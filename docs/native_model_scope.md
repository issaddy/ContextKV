# Native B0/B1 的显式本地模型与采样入口

正式 `bench.native_baseline` 现在可以使用与研究 Controller 相同的 `RunConfig.model/context/sampling`。这是一条预启动 stock 服务的入口接线；没有扩大 owned B6 启动器、自动部署、下载权重、改写历史 manifest 或执行新的 GPU 批次。

B0 仍提交完整共享候选，不发送 native session ID，不开启 HiCache/SessionRadix；它不是完全无 radix 缓存。B1 仍对同一会话使用稳定 native session ID，完整重建输入并保留实际 QA；`close_session` 只作为 dispatch，不伪造 scheduler retirement ACK 或物理回收。

## 调用

先由操作者启动与实际 manifest 一致的固定 stock SGLang 服务，然后运行：

```bash
python -m bench.native_baseline data/mtrag/RAG.jsonl \
  --baseline B0 --base-url http://127.0.0.1:30001 \
  --run-config configs/qwen3_8b_scope.example.yaml \
  --engine-manifest /absolute/path/to/actual-native-manifest.json \
  --protocol P2 --domain all --split-manifest configs/mtrag_split_manifest.json \
  --split dev --audit-prompt-ids --output runs/native-local-scope-dev
```

scope example仍为显式固定策略/默认贪心。要使用其他配方，在独立共享配置中声明 `sampling` 四项，见[sampling.md](sampling.md)，不修改例子全局默认。CLI `--base-url` 和 `--engine-manifest` 继续是 native endpoint/runtime 的显式来源；`--run-config` 的 engine 字段不启动或选择服务器，只取其 model/context/sampling。`--context-config` 旧入口继续只共享文本策略，两种配置入口互斥。原显式 context/output CLI 覆盖仍可用，最终预算必须与真实服务 manifest 一致。

模板复用 `create_model_template(ModelConfig)`，本地 scope 的 tokenizer 加载 revision 为 `None`，fingerprint 使用已有 `local-snapshot:…` metadata identity。没有把本地标签传成 HF revision，也没有绕过正式 validator 的 `test_mode`。scope 仍只支持目前已验证的 BF16 dense Qwen3、无 SWA/RoPE 扩展，以及同容器完整 snapshot 文件验证；不是任意模型自动发现或远程 tokenizer-only 支持。

不带新共享配置/显式scope/recipe的旧调用继续使用固定 Qwen2.5-7B revision、旧 manifest 校验和原 `temperature:0` wire。显式标准贪心与旧请求等价；旧无采样字段的7B manifest可以继续兼容，但不会被补写成新的采样来源记录。

## 新 scope manifest 的明确字段

原 `engine/server/runtime` 部分继续需要真实来源与原生池/硬件记录。新入口不替操作者制造这些观测。新增模型部分使用以下合同：

| 字段 | 要求 |
|---|---|
| `model.name` | scope 的 canonical model path |
| `model.revision` | `null`，不编造已知 HF revision |
| `model.identity_kind` / `model.identity` | `local_snapshot_metadata` / 已解析 scope identity |
| `model.tokenizer_hash` / `model.chat_template_hash` | 与实际模板 fingerprint 相同 |
| `model.dtype` / `model.kv_dtype` | `bfloat16` |
| `model_scope` | `LocalModelScope.manifest()`记录；校验identity和canonical path，不以备注或scope文件路径当模型身份 |
| `sampling_recipe` | 当前实际完整四项或标准贪心的 `{"temperature":0}`，必须与请求设置一致 |
| `server.pp_size` | 1；原TP/DP1、B0/B1 flags、page/context合同保留 |

运行时另外复用 `scope.verify_native_settings`，检查 actual model/tokenizer路径、dtype/KV dtype、无revision/quantization/JSON模型覆盖以及非dummy加载。新scope不接受“manifest有但实际server_info缺失”的必需服务设置；实际`max_total_num_tokens`也必须等于manifest记载的GPU池。失败的实际server_info仍保存，不能只留下期望值。

stock HTTP没有完整host allocator观测，因此host容量仍需要原生allocation日志等真实来源，不能把旧7B的279040 tokens抄到Qwen3。已有pool换算优先使用显式`head_dim`，32B是128，不能用hidden5120/heads64算成80。元数据/分片库存不是权重内容attestation；stock source声明也不替代owned进程来源证明。

## 共同矩阵

`bench.experiments`将同一已解析 `ModelConfig`、`SamplingRecipe`和context policy传给native，Controller也使用这份共享配置。含native的矩阵要求方法/候选的model与sampling等价，拒绝native自带另一份scope/recipe/config以及与共享scope不同的model_path。不能通过方法级温度变化把采样干预伪装成B0/B1/Controller策略差异。

这一入口允许**静态、实际已观测的 engine_manifest + 预启动服务**运行新scope/recipe。带 `server_profile` 的现有owned native路径仍只允许legacy7B/标准贪心，并在启动前拒绝新scope或recipe：旧owned manifest工厂和启动配置本轮未泛化。原13格及其旧freeze身份保持；新代码不能追认旧结果同源。

新 native 输出保存 `shared_configuration`、原engine manifest、实际server info、模板fingerprint及每次真实HTTP参数/输入输出；现有完整QA、正文span、失败、length与质量导出分母继续复用。这个入口不使用研究cost artifact，不将Qwen3绑定到7B的时间/输出长度分布，也不自动授予quality eligibility或B6资格。

## 本地验证范围

CPU合同使用真实 `NativeSGLangClient` + `httpx.MockTransport`、明确的合成scope/ByteTemplate。正式validator运行，未使用`test_mode`放行新路径。覆盖B0/B1 session定义与完整自有历史、共享recipe与唯一预算、8B/32B identity错配、32B显式head_dim128、错误actual路径/dummy/override/precision/PP/context/缺字段/池容量零生成拒绝，以及CLI和matrix的真实透传。旧default与stream/nonstream HTTP合同另保留。

这些测试没有加载权重或生成答案。此前stock318与patched Qwen3有限scope实测证明相关底层机制可运行，但不追认为这一新正式入口已做GPU验收。需要后续使用时先做最小入口功能检查，不自动扩展为159轮或13格运行。
