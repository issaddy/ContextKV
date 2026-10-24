# 显式 Qwen3 本地模型 scope

真实运行必须声明 `ModelConfig.scope_path`；未声明模型的默认配置只服务CPU mock，不下载或隐式选择生成模型。当前支持同容器的dense Qwen3-8B/32B、BF16、TP/DP/PP1，无SWA/RoPE扩展，完整本地snapshot可读。旧7B运行路径已删除，历史manifest只供只读复核。

## 身份与启动边界

`engine_patches/contextkv_model_scope.py`是唯一标准库解析器，补丁包含其原样副本。已有记录：

- `configs/model_scopes/qwen3-8b.local.json` → `/media/ssd1/qwen3-8b`
- `configs/model_scopes/qwen3-32b.local.json` → `/media/ssd1/qwen3-32b`

记录来自实际config、分片库存/大小和tokenizer/template检查，`hf_revision:null`、`weight_content_verified:false`。`local-snapshot:`是已有metadata的身份，不是HF commit，也不证明同大小权重内容没有变化。32B显式head_dim128，不能由hidden5120/heads64推成80。

scope初始化解析一次，加载边界核文件/库存与tokenizer，原生guard核实际对象/配置；不逐请求读取scope或哈希权重。真实加载revision为None，identity单独进入fingerprint。name/revision/加载模式与scope冲突时拒绝，不默默覆盖。

## 使用

[8B共享配置](../configs/qwen3_8b_scope.example.yaml)和[32B配置](../configs/qwen3_32b_scope.example.yaml)都显式使用`.7/.8/20/0`、non-thinking、完整历史与单次REBUILD，不继承旧时间/长度artifact。先按[部署说明](deployment.md)启动对应引擎，再运行`controller.cli --config ... serve`或`bench`。`demo`仅用于mock，不接受硬件配置。

研究启动脚本必须设置 `CONTEXTKV_MODEL_SCOPE_PATH`。模型路径来自scope，禁止尾部参数重写model/tokenizer/revision；实际BF16、无quantization/JSON override/dummy加载和拓扑约束由原有guard核对。client首次及重读info时核fingerprint、scope identity和规范化模型目录；source注记或scope文件位置本身不是模型身份。

## 成本和分析复用

配置cost artifact时要求实际fingerprint/cache绑定匹配，采样及长度模型再按其来源核对。旧模型测量不获得Qwen3校准资格；无artifact的固定策略可正常运行。

`profiling.sglang_probe`与smoke/lifecycle/observation工具使用共享模板factory。decision-pairs freeze也复用它，不把metadata identity交给HF当revision；实际采集必须与冻结病例同scope。保留Measurement、离线拟合、历史JSON解析与不加载模型的候选政策复算，删除无生产消费者的旧manifest模板。

当前模板仍使用原Qwen3 non-thinking渲染。历史assistant前导变化造成的前缀边界见[结构审计](qwen3_candidate_opportunity.md)；未经明确新范围和验证不改变默认模板或复用不匹配KV。已有[有限GPU验收](qwen3_scope_gpu_validation.md)和[统一入口验收](owned_local_scope_validation.md)各自绑定原源码，本轮迁移不改写它们。
