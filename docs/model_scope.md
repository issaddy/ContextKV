# 显式 Qwen3 本地模型 scope

本入口将用户已有本地 Qwen3 元数据接到 controller、研究启动脚本、原生 guard 和 profiler。它不改变原生调度、FULL KV、软引用或 DMA 算法，不把新模型套进旧成本表，也不改已有7B正式manifest。当前只支持 **controller与engine在同容器、都可读取该完整本地snapshot**；尚不支持controller只有tokenizer文件而权重在另一台机器的scope部署。

## 来源与身份

`engine_patches/contextkv_model_scope.py` 是唯一的标准库解析器；同一文件原样包含在SGLang补丁中，安装检查核其字节一致。示例记录：

- `configs/model_scopes/qwen3-8b.local.json` → `/media/ssd1/qwen3-8b`
- `configs/model_scopes/qwen3-32b.local.json` → `/media/ssd1/qwen3-32b`

两记录来自2026-10-24的实际本地config、分片库存和tokenizer检查；记录内保存来源摘要。它们没有已确认的HF revision，`hf_revision=null`、`weight_content_verified=false`。内部`local-snapshot:`身份由已有config/tokenizer/template摘要和分片数量/字节元数据派生，不是HF commit，不证明同大小权重文件内容没被替换。

读取scope不加载模型。启动边界核实际config、index/shard库存和大小，再核真实tokenizer与非思考模板。32B显式`head_dim=128`，不能用hidden5120/heads64推成80。当前类型限定为BF16 dense Qwen3ForCausalLM，无SWA/RoPE扩展；仍要求原生Python FULL单组件、TP/DP/PP1及既有session/cache合同。

HFTemplate将加载revision与指纹identity分开：旧调用仍传原HF revision；显式scope调用省略AutoTokenizer的revision，只把local identity写入fingerprint。原生启动也不传假的`--revision`。模板仍用`enable_thinking=False`；不修改原生模板。完整历史若被模板变换而丢失，现有候选校验会失败，不清洗原文伪造full QA。

## 配置与启动

`ModelConfig.scope_path`是显式入口，未提供时仍为原7B默认。提供时从scope派生model path、identity和local_files_only=True；若同时给name/revision/加载模式且冲突，会拒绝配置。scope在配置初始化解析一次；模板创建和原生guard分别在各自进程的启动边界核实际文件/对象，不逐请求读取scope或哈希模型。

可从以下固定策略配置开始，不加载时间/长度artifact：

```bash
python -m controller.cli --config configs/qwen3_8b_scope.example.yaml demo
# 或 configs/qwen3_32b_scope.example.yaml
```

服务须由操作者先启动。研究launcher的scope入口为：

```bash
export CONTEXTKV_MODEL_SCOPE_PATH=configs/model_scopes/qwen3-8b.local.json
unset CONTEXTKV_MODEL_PATH  # 也可明确设成scope中的同一规范化路径
export CONTEXTKV_FINGERPRINT_JSON="$(python -c 'import json,os; from dataclasses import asdict; from controller.config import ModelConfig,create_model_template; print(json.dumps(asdict(create_model_template(ModelConfig(scope_path=os.environ["CONTEXTKV_MODEL_SCOPE_PATH"])).fingerprint)))')"
export SGLANG_CHECKOUT=vendor/sglang
export CONTEXTKV_HICACHE_LAYOUT=layer_first
export SGLANG_HICACHE_TMA_TRANSFER=0
bash scripts/launch_sglang.sh --host 127.0.0.1 --port 30000 \
  --max-total-tokens 32768 --max-running-requests 1 --random-seed 42
```

这是命令示例，不是已执行的GPU验收。启动前使用固定SGLang完整checkout及依赖，应用当前补丁；实际GPU/host容量与显存参数由操作者确认，不能照搬旧7B小池的mem_fraction_static=.15。新scope的原生guard会核实际BF16权重/KV、上下文不超过config、无HF revision、无模型override/quantization，并排除`load_format=dummy`。原有topology/cache限制继续生效。

无scope的旧launcher参数顺序与默认模型行为保持。新scope模型路径只由记录提供，拒绝尾部`--model-path/--tokenizer-path/--revision`覆盖。研究client首次及重读info时比较实际fingerprint和scope的identity/规范化模型目录；scope文件保存位置与source注记不是模型身份。两个指向同真实目录的路径别名可用；metadata相同但另一个模型目录不会仅凭相同fingerprint被当作同一scope。

## 采样与标定范围

research client仍保持原temperature0默认；**没有**把独立Qwen3能力参考的`.7/.8/top_k20/min_p0`悄悄应用到主工程。后续`b3d3c2f`已提供独立[显式采样配置](sampling.md)，同时记录实际request/profile配方，并分别核对时间表、经验长度与learned模型的来源和残差基准。此处示例配置仍是默认贪心的身份/协议烟测，不用于宣称推荐采样的质量。已完成的`a87638a`硬件验收不包含后续采样接线，新配方的主工程GPU实验另行记录。

显式scope若配置cost artifact，factory要求其profile manifest绑定相同实际fingerprint及cache_config_hash。旧7B artifact、没有来源绑定的legacy lookup、只有raw calibration rows都会在创建客户端前拒绝；不靠legacy_unbound让新模型成为calibrated。实际cache_config匹配仍由原有启动/重启scope检查完成。无artifact的固定策略可运行，但没有GPU时间校准资格。旧7B未选scope时，其已有legacy处理不变。

`profiling.sglang_probe`与smoke/lifecycle/observation工具使用同一模板factory及预期scope；probe清单增加已解析model_scope，同时保留真实fingerprint、engine、pool、cache_config和runtime。Measurement列、fit/interpolation与旧artifact不改变。`controller.cli._manifest`本来记录真实config/fingerprint/engine，scope数据经engine info进入该清单。

`profiling.common.run_manifest`是未被当前controller/probe生产路径调用的旧7B清单模板，本切片保留其历史默认；不能把它用于Qwen3再称真实模型记录。后续[正式native B0/B1薄入口](native_model_scope.md)已允许共享显式scope/recipe与实际静态manifest，不使用test_mode绕过身份。owned B6启动器仍限定原7B/标准贪心；该新入口不追认旧a876硬件批次或历史manifest。

## 本地验证与后续GPU验收

25项scope CPU测试覆盖实际HF adapter加载参数、默认7B值/旧wire/缓存hash、模型显式冲突、config与分片库存、native实际settings、dummy拒绝、真实runtime→HTTP info解码、跨目录错误连接、路径别名、scope单次初始化、成本绑定拒绝、shell启动参数与probe来源。测试使用明确的CPU模型/文件替身，不是GPU或质量验证。

当前完整本地套件：1236 passed、6 subtests；真实7B tokenizer启用，Ruff通过。另有4项独立合同覆盖旧guard、缺scope连接、cold reset和实际缓存配置变化，共29项scope专项通过；[独立审查](../runs/qwen3-scope-review/README.md)保留具体边界。固定原源码SHA、补丁应用、15个相关Python文件静态编译、统一guard时序及解析器副本一致性已通过source-only检查；报告在`runs/qwen3-engine-scope-sourcecheck.json`，明确GPU/runtime依赖导入未验。

随后已在H200对两个模型执行[有限GPU验收](qwen3_scope_gpu_validation.md)：info/scope、实际pool、prepare/generate/status/retire、四轮完整QA与实际IDs、GPU/Host配对生命周期及并发观测均有真实记录；并发观测的已知match等待状态仍未覆盖，明确为partial。部署源为`a87638a`，两模型独立state，未迁移旧7B历史或pending attempt。这不验证新模型成本预测、质量或一般性能收益；独立标定及正式负载验证仍需继续。
