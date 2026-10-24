# 有源多轮回答的离线训练链路

2026-10-24。新增`training`包，复用 Transformers Trainer、PEFT LoRA 和官方collator，实现明确训练样本→精确推理前缀/目标labels→有步数上限的训练→adapter保存。线上仍使用原单次生成。当前已验证编码、真实CPU训练与保存重载合同，尚未训练Qwen3-8B语义模型、部署adapter或证明质量提升。

## 方法依据与本项目适配

[RAFT](https://arxiv.org/html/2403.10131v2)通过正确材料与干扰材料训练来源阅读；[PA-RAG](https://aclanthology.org/2025.naacl-long.459/)进一步在相同输入上训练完整性、抗干扰和引用偏好。其实体范围混接与当前产品/参数错用有直接对应，值得复用学习目标。固定代码、实际小样、许可与CPU反例见[研究记录](../runs/grounded-training-reuse/README.md)。

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

现有六条起步包可运行：

```bash
uv run --extra tokenizer python -m training.sft prepare \
  --examples runs/grounded-supervision-design/starter.messages-target.jsonl \
  --output runs/NEW-SFT-PREPARED \
  --model-scope configs/model_scopes/qwen3-8b.local.json \
  --tokenizer-path runs/remote-h200-qwen3-reference/model-metadata/8b \
  --max-tokens 8192
```

`--tokenizer-path`可指向本地metadata而不加载权重；指纹仍与scope核对。输出`examples.jsonl`包含实际IDs/labels/attention mask及离线身份，`manifest.json`绑定模板、输入/输出文件摘要和计数。来源、目标及EOS必须共同适配预算；遇到超长、非train、重复身份或无效输入时，整次准备失败，已写原件及失败/未处理分母保留。没有静默裁剪或跳样。加载阶段只接受完成且内容/指纹相符的产物，不重新分词或逐样本hash。

六条来自四个真实train对话/全文：四个原回答仅附文档引用、两个澄清原文不硬加引用，合计11870tokens、最大3425。审核明确为assistant弱监督，不是最终训练集或独立验证集。20,431个user后目标的单全文/完整历史输入最大6429tokens，在8K内可预留1024输出；这个预算结果不适用于top-k多文档，也不是训练显存估计。证据见[材料设计](../runs/grounded-supervision-design/README.md)。

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

**当前native guard不支持动态LoRA。** 新产物明确标记`native_serving_compatible=false`。后续需合并到独立checkpoint并建立区别于基座的新权重身份、核对生成结果后接入现有服务；现metadata scope仅凭config和shard库存不能保证区分同大小的新权重，这一集成尚未实现。不能将训练loss下降或adapter保存当作系统交付完成。

## 已执行验证

真实Qwen3 tokenizer的编码/数据合同34项通过，覆盖连续assistant、目标前导换行、EOS、padding接口、超长失败、严格数据分母与metadata隔离；六条实际起步数据通过同一prepare CLI。

容器CPU使用实际Torch2.14.1/Transformers5.19.0及隔离目录的PEFT0.21.0/Accelerate1.15.0，随机初始化1层/32hidden的小Qwen3，执行两个optimizer steps。1024个LoRA可训参数、9717184总参数；原base/embedding/lm_head参数不变、七类投影正确、LoRA B发生更新、目标/EOS/padding mask保留、手算shifted CE一致、同一base重载adapter后logits一致。临时小模型与adapter随合同验证结束删除，不替换8B权重。

首次CPU尝试在创建TrainingArguments时发现5.19已移除`warmup_ratio`，尚未执行训练；已按固定版本改为接受比例的`warmup_steps=0.03`，保留失败和修复后的记录。小模型loss约11.9仅证明训练接线，不能解释为语义改善。没有GPU训练或8B生成。后续只有经过来源/对话分组的实际训练与独立任务对照，才可判断范围、条件、覆盖和完整响应成本的综合收益。

整仓回归为1448 passed、8 subtests、1 skipped（本地未安装训练运行库，显式opt-in的CPU训练测试未启用）；该项另在容器实际执行为1 passed/7.29s，检查同一训练函数、两步更新与逐值重载一致性。命令为：

```bash
CONTEXTKV_TEST_TRAINING=1 CONTEXTKV_TEST_HF_TOKENIZER=1 \
CONTEXTKV_TEST_TOKENIZER_PATH=/media/ssd1/qwen3-8b \
python -m pytest -q tests/test_training_runtime.py
```

容器验证将新增纯Python依赖放在独立目录，只为该命令设置PYTHONPATH，未升级或修改SGLang环境。验证前后GPU空闲。远端pytest未装异步插件产生`asyncio_mode`配置提示，另有既存SWIG弃用提示；该测试同步执行且通过，不将这些提示隐藏为无警告运行。

`f72d152`已打包并同步容器，92模块/31资产/367sdist/13许可及368个远端tracked条目核验通过，training CLI help可在未安装PEFT的常规环境运行。CPU合同执行的staged源码与最终部署逐字相同。12文件公共记录为`runs/grounded-training-public-v1.tar.xz`，19160 bytes，SHA256 `3a795864b93c3711abd475cc3a5603e6bf5f9e596812c23d6d99fe0195c495f2`，11个payload核验通过；保留首次失败及后续成功，不包含临时随机模型权重。

[方法、样本与准备产物快照](../runs/grounded-training-review-v1.tar.xz)共96文件、539176 bytes，SHA256 `31420cc8e284f9205533ea2e0b2e1a36ef0c28c200c619f9bb73077720a92dce`，95个payload核验通过。源数据仍绑定先前下载的作者train；没有额外复制验证/测试对话、完整公开训练集、第三方运行环境或大模型权重。
