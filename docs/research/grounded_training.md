# 有源多轮回答的离线训练链路

> 当前状态：可选支线，暂缓扩展。一次30-step Qwen3-8B训练（一个完整遍历）和独立CPU合并已完成；未执行准备中的32次对比生成，尚无语义收益结论。课题主线及后续优先级见[项目范围](../project_scope.md)。下文接口、方法和材料保留供复用，不是当前必做清单。

2026-10-24。`training`包复用 Transformers Trainer、PEFT LoRA 和官方collator，实现明确训练样本→精确推理前缀/目标labels→有步数上限的训练→adapter保存→离线合并为独立BF16 checkpoint及新scope。编码、CPU训练及导出重载合同已通过；已执行的小规模8B训练仅证明运行与更新，未证明质量提升，当前服务主线仍使用原基座。

## 方法依据与本项目适配

[RAFT](https://arxiv.org/html/2403.10131v2)通过正确材料与干扰材料训练来源阅读；[PA-RAG](https://aclanthology.org/2025.naacl-long.459/)进一步在相同输入上训练完整性、抗干扰和引用偏好。其实体范围混接与当前产品/参数错用有直接对应，值得复用学习目标。固定代码、实际小样、许可与CPU反例见[研究记录](../../runs/grounded-training-reuse/README.md)。

本项目第一步采用**证据可用条件下的最终回答SFT**。不复制RAFT在移除正确材料后仍训练原答案的记忆设置；不额外生成长CoT；完整来源、历史和问题全部mask，只监督当前目标与EOS。这是针对decode占主导的可检验适配，不预设短回答质量更好。随后有价值的扩展是同输入的范围/必要内容偏好DPO，尚未实施；不机械执行全部作者阶段或部署在线修订器。

原Doc2Dial target和grounding均可能有噪声，不能把所有refs当充分支持、未标注材料当负例，或用span loss绕过语义问题。PA-RAG公开CQ小前缀也有未声明的引用编号，且仓库/数据未声明整体许可。本阶段独立实现方法接口，使用许可明确的既有Doc2Dial材料，没有复制这类数据进入发行包。

## 精确训练边界

原版Qwen3在当前assistant成为历史时可能改变空thinking前导；完成聊天后整体分词也会合并目标前导换行。实际tokenizer反例证明，不能用“完整chat tokens减prefix tokens”定位labels。

`encode_example`直接取现有`HuggingFaceTemplate.render`的真实非思考推理前缀IDs，并全部置为`-100`；另编码未经strip的目标原文，附一个EOS，仅该段进入loss。不合并角色、不重排历史、不packing、不截断。目标为空时显式监督立即EOS；数据选择者决定是否使用。模型控制token/think标记不作为普通目标文本接受，因为它们会终止生成或改变后续历史渲染。来源文本仍遵循现有推理合同，不被训练器偷偷改写。

`DataCollatorForSeq2Seq(model=None)`保留既有labels，补齐位置为`-100`。不是会重建所有labels的语言模型collator。Trainer负责标准causal shift和优化器循环，没有新增训练框架或逐请求计算。

## 输入与准备命令

独立环境可按锁文件安装，避免修改既有推理环境：

```bash
UV_PROJECT_ENVIRONMENT=.venv-training uv sync --extra training --locked
```

核心版本为Transformers5.19.0、tokenizers0.23.2、PEFT0.21.0、Accelerate1.15.0；锁文件解析Torch2.14.1。常规服务不安装此extra也可运行。输入JSONL每条显式包含：

```json
{"id":"source-dialogue/target-turn","split":"train","messages":[{"role":"user","content":"完整来源和原问题"}],"target":"经核对的目标回答","provenance":{"kind":"original_or_declared_revision"}}
```

messages应由严格历史前缀和真实来源构造；DA、gold span、审核理由放在provenance/独立侧车。准备器不自动判断语义、修正target或检测人为预先塞入messages的未来答案，也不依据provenance文字给予质量资格。

输入已审核材料后，可使用既有准备入口：

```bash
uv run --extra tokenizer python -m training.sft prepare \
  --examples data/reviewed-training.messages-target.jsonl \
  --output runs/NEW-SFT-PREPARED \
  --model-scope configs/model_scopes/qwen3-8b.local.json \
  --tokenizer-path runs/remote-h200-qwen3-reference/model-metadata/8b \
  --max-tokens 8192
```

`--tokenizer-path`可指向本地metadata而不加载权重；指纹仍与scope核对。输出`examples.jsonl`包含实际IDs/labels/attention mask及离线身份，`manifest.json`绑定模板、输入/输出文件摘要和计数。来源、目标及EOS必须共同适配预算；遇到超长、非train、重复身份或无效输入时，整次准备失败，已写原件及失败/未处理分母保留。没有静默裁剪或跳样。加载阶段只接受完成且内容/指纹相符的产物，不重新分词或逐样本hash。

六条来自四个真实train对话/全文：四个原回答仅附文档引用、两个澄清原文不硬加引用，合计11870tokens、最大3425。审核明确为assistant弱监督，不是最终训练集或独立验证集。20,431个user后目标的单全文/完整历史输入最大6429tokens，在8K内可预留1024输出；这个预算结果不适用于top-k多文档，也不是训练显存估计。证据见[材料设计](../../runs/grounded-supervision-design/README.md)。

## 有上限的LoRA训练与产物

对已经选定的训练分区，入口为：

```bash
UV_PROJECT_ENVIRONMENT=.venv-training uv run --extra training --locked \
  python -m training.sft train \
  --prepared runs/SELECTED-TRAIN-PREPARED \
  --model-scope configs/model_scopes/qwen3-8b.local.json \
  --output runs/NEW-SFT-RUN --max-steps 32
```

这是入口示例，不是对六条起步数据执行32步的质量实验建议。正式CUDA模式读取本地BF16基座、使用SDPA、batch1和gradient checkpointing；max-steps必须显式给定且为正。LoRA默认rank16、alpha32、七类线性投影、dropout0、AdamW与线性调度，可明确改变rank/学习率/累积步数，不自动搜索超参。输出目录独占，不下载模型、不上传Hub、不自动启动推理服务。当前没有断点续训入口或完整optimizer checkpoint，失败记录保留，不隐式续跑。

产物`adapter/`含PEFT权重/config和tokenizer，`trainer_state.json`保留实际步骤；manifest记录原scope、准备产物、有效训练参数、库版本、结果及adapter文件身份。文件摘要只在产物边界计算。全量base参数冻结，adapter不能冒充原base identity或沿用已有KV/成本标定。

**当前native guard不支持动态LoRA。** adapter训练产物仍标记`native_serving_compatible=false`。下面的离线导出将它转换为普通dense checkpoint，使用新模型身份接现有控制器、stock SGLang和研究引擎，不在请求时切换adapter。

## 独立checkpoint导出与接入

对已完成的训练产物运行：

```bash
UV_PROJECT_ENVIRONMENT=.venv-training uv run --extra training --locked \
  python -m training.export \
  --training-run runs/COMPLETED-SFT-RUN \
  --model-scope configs/model_scopes/qwen3-8b.local.json \
  --output runs/NEW-MERGED-CHECKPOINT \
  --max-shard-size 5GB
```

导出器先核对训练完成状态、adapter文件及其绑定的基座身份，再独立加载CPU BF16基座，通过PEFT `merge_and_unload(safe_merge=True)`生成dense权重。输出必须是新的独立目录，不覆盖基座、训练目录或旧导出；失败记录保留，不自动重试、启动服务器或发布Hub。原始tokenizer/template文件原样复制，正常保存模型config和推理所需的`use_cache=True`，不靠修改架构或虚构HF revision制造身份差异。

`safe_merge=True`检查合并权重是否有限，不检验回答是否正确。普通LoRA合并遵循`W' = W + (alpha/r)BA`；复用[PEFT官方合并接口](https://huggingface.co/docs/peft/developer_guides/checkpoint)及[固定0.21.0实现](https://github.com/huggingface/peft/blob/v0.21.0/src/peft/tuners/lora/layer.py)，不复制另一个合并器。BF16合并会舍入更新，微小LoRA增量可能被舍入，因此不承诺未合并adapter与dense模型逐值相等或生成逐token相同。保存/重载的一致性、实际模型可加载与任务质量需要分别验证。

输出包括普通HF checkpoint、`export-manifest.json`和最后发布的`model-scope.json`。只在权重保存完成且库存检查通过后发布scope；manifest保留基座身份、adapter身份、导出参数和库版本。`quality_validated=false`与`native_serving_validated=false`明确不授予语义或GPU运行资格。单文件`model.safetensors`无需伪造index；分片输出使用保存器实际产生的index。

新scope采用schema v2、`identity_kind=local_checkpoint_artifact`：每个weight文件记录`file/bytes/sha256`，按文件名规范排序，与config、tokenizer和template共同绑定`local-checkpoint:`身份。相同大小但不同权重得到不同身份；目录位置和来源说明不改变身份。原基座schema v1及`local-snapshot:`指纹保持不变。

每个新权重文件只在这次**导出发布边界**读取全量内容并计算摘要。随后启动核对config、文件名/大小和index；单文件只读8字节header长度及JSON offsets验证tensor字节数，不读或哈希tensor body。`weight_content_verified=false`表示启动未全量复验当前内容，发布摘要不是持续防篡改监控。产物发布后按不可变目录使用；若改动权重，应导出到新目录并生成新scope，不能保留旧身份冒充原产物。

复制所需Qwen3共享运行配置，将其中`model.scope_path`改为导出的绝对路径，例如：

```yaml
model:
  scope_path: /home/relay/songjiayun/ContextKV/runs/NEW-MERGED-CHECKPOINT/model-scope.json
```

继续使用同一共享配置的采样、预算与现有服务入口。启动器的`CONTEXTKV_MODEL_SCOPE_PATH`也指向该scope，并重新生成对应fingerprint；stock/native基线按同一scope加载dense模型。使用独立state/run目录和新引擎实例，不复用基座session/KV、成本表或输出长度标定。已有指纹和成本绑定负责拒绝混用，无需新缓存或调度器。详细启动流程见[固定引擎部署](../deployment.md)及[引擎scope合同](../../engine_patches/README.md)。接口接通不代表这一新权重已完成GPU验收。
