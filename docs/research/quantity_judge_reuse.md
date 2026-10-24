# 数值与范围错误：下一轮开源评审复用

本轮只读研究针对 [Atomic v1/v2 实测](../archive/atomic_judge_validation.md) 已出现的具体问题：2048 KiB→2 MiB 与 96 小时→4 天换算失败；每项与合计混淆；Enterprise 脚注被默认值覆盖；历史 assistant 的数值被当作来源；正确的单附件上限被判矛盾。JSON Schema 只能处理结构损失，不能解决这些错误。因此，下一轮优先比较专门训练的判别模型及有来源绑定的数值运算，继续反复改 Selene prompt 不是唯一途径。

未下载模型权重、未运行 GPU、未访问项目或元评测 test 样本。这里只读取作者论文、代码、模型卡、config 与公开 API 元数据；两个新模型的首个权重文件仅做无鉴权 HEAD，均为 200。小型来源快照与访问记录在 `runs/atomic-development-review/quantity-reuse-sources/`。所有预期收益均为待验证推断，不是本项目效果结论。

## 两个可实际接入的新模型

| 候选 | 固定模型 revision / 许可 | 架构与窗口 | 可复用合同 |
|---|---|---|---|
| `ibm-granite/granite-guardian-4.1-8b`，2026-04 | `ab01ccca5dcfb80246369a086a4a87a29198f5af`；Apache-2.0 | `GraniteForCausalLM`，config=131072 tokens；约 8.38B BF16，权重下限约 16.76 GB，未测运行显存 | 原生 groundedness 风险 yes/no、明确目标为最后一个 assistant；独立 Granite 家族，与当前 Qwen2.5 回答器不同 |
| `KRLabsOrg/lettucedect-v2-mmbert-base`，2026-07 方法 | `8831ce063dff760b01efe4519b4188f5ee2456f2`；权重 Apache-2.0、框架 MIT | `ModernBertForTokenClassification`，config=8192 tokens；307.5M、公开 FP32 权重 1,230,141,424 bytes，未测运行显存 | 给 answer token 直接判 supported/unsupported 并还原字符 span；没有自由生成 JSON 的接口损失 |

Granite Guardian 4.1 是训练过的风险判别器，支持自定义标准和 think/no-think 输出。应优先用作者原生 groundedness 标准做完整 29 条对照，保留 `<score>yes/no</score>` 的风险方向：`yes` 表示满足“不受来源支持”的风险标准，不能误写成答案正确。不能把它直接塞进当前 Atomic 的 JSON checker prompt，并把结果称为作者原生用法。[固定模型卡](https://huggingface.co/ibm-granite/granite-guardian-4.1-8b/blob/ab01ccca5dcfb80246369a086a4a87a29198f5af/README.md)、[固定 config](https://huggingface.co/ibm-granite/granite-guardian-4.1-8b/blob/ab01ccca5dcfb80246369a086a4a87a29198f5af/config.json)。

LettuceDetect 的 2026 论文把文档、代码、工具结果等统一成回答 span 检测任务；mmBERT 是其中的小型编码器路线。其二分类头不需要模型自己生成字段、证据列表或真假解释，适合作为现有 ledger 的另一类诊断。它仍可能误判换算、条件及来源组合，也不输出五维质量；作者在其他 benchmark 上的结果不是我们的数值条件测试结果。[作者论文](https://arxiv.org/abs/2607.00895)、[固定模型卡](https://huggingface.co/KRLabsOrg/lettucedect-v2-mmbert-base/blob/8831ce063dff760b01efe4519b4188f5ee2456f2/README.md)、[固定 config](https://huggingface.co/KRLabsOrg/lettucedect-v2-mmbert-base/blob/8831ce063dff760b01efe4519b4188f5ee2456f2/config.json)。

这里没有按“更新就更强”选模型：作者卡片明确指出 v2 mmBERT 的 RAGTruth 指标低于专门做英语 RAG 的 `lettucedect-large-modernbert-en-v1`。该 v1 候选固定 `22296c700ef0ba4ab3e5c9afffa0185caaf61e52`、MIT、约 395.8M，可以在首轮之后作为小型敏感性对照。首轮先不同时扩大到多个模型、taxonomy head 与 Qwen generative detector；后者与回答器属于相关家族，且再次引入生成格式问题。[作者固定 README 的模型区分](https://github.com/KRLabsOrg/LettuceDetect/blob/fbce1991f26e210cc49ace3913562ab4f2f31a60/README.md#typed-spans-v2)、[v1 权重](https://huggingface.co/KRLabsOrg/lettucedect-large-modernbert-en-v1/tree/22296c700ef0ba4ab3e5c9afffa0185caaf61e52)。

## 不能直接照搬的两处实现

**Lettuce 的 chunk max 会改变“被来源支持”的含义。** 固定作者源码在超过默认 4096 时按 passage 分块，再对每个 answer token 取各块 hallucination probability 的最大值。这样，一块包含正确端口事实，另一块仅含 timeout，后者的不支持就可能覆盖前者的支持；需要两个来源联合换算的判断也无法靠独立块合并恢复。这是代码口径与本项目全部可见来源集合的冲突，不能把该 max 解释成可靠的全证据判定。单次输入优先完整放入模型 config 允许的 8192 范围；实际 tokenizer 计数超限则记录未覆盖，或另做明确分组的探索实验，不能默默换成 min 后宣称解决了冲突和联合蕴含。[固定分块及聚合源码](https://github.com/KRLabsOrg/LettuceDetect/blob/fbce1991f26e210cc49ace3913562ab4f2f31a60/lettucedetect/detectors/transformer.py#L338-L370)。

其 `predict_prompt_batch` 与模型卡快速示例还使用 `truncation="only_first"`。薄适配必须先完整计数，并关闭静默截断，记录全部输入覆盖；“模型支持 8K”不等于包装器默认已经使用 8K。可直接复用标准 Transformers token-classification 推理、作者输入格式与 offset 解码，固定源 revision，避免为了一个判别头导入框架全部 OpenAI、rag-fact-checker、sklearn 等依赖。[固定 tokenization](https://github.com/KRLabsOrg/LettuceDetect/blob/fbce1991f26e210cc49ace3913562ab4f2f31a60/lettucedetect/detectors/transformer.py#L280-L325)、[依赖](https://github.com/KRLabsOrg/LettuceDetect/blob/fbce1991f26e210cc49ace3913562ab4f2f31a60/pyproject.toml)。

**Granite 需要它自己的原生渲染与风险方向。** 作者用末尾 guardian 指令选择最后一个 assistant，并通过 tokenizer 的 `documents` 参数提供来源；它不使用 Selene Result 格式。应按固定官方 helper 与 chat template 先本地渲染，保留实际 query/history 的角色、最后一个待评分答案，并让 documents 只含白名单可见原文。历史回答不能混入 documents。渲染后的原生 prompt 需走不再次套模板的 completion/generate 路径，或先证明服务端 documents/template kwargs 等价；不能把完整渲染文本再作为 chat user 消息包一次。已有固定 SGLang 源码包含该架构，但这只是静态兼容依据，仍需一次实际加载与原生短判分合同检查。[作者使用示例](https://huggingface.co/ibm-granite/granite-guardian-4.1-8b/blob/ab01ccca5dcfb80246369a086a4a87a29198f5af/README.md#example-2-detect-groundedness-in-rag-settings-think-mode)、[固定 tokenizer 模板](https://huggingface.co/ibm-granite/granite-guardian-4.1-8b/blob/ab01ccca5dcfb80246369a086a4a87a29198f5af/chat_template.jinja)、[SGLang 固定 Granite 实现](https://github.com/sgl-project/sglang/blob/3f36fc08331f64cd63c4fbbdf39cc402f38bb38b/python/sglang/srt/models/granite.py#L350)。

## 针对数值关系的可复用算法

Q-REAS/EQUATE 虽然是 2019 工作，但比再换一个通用打分 prompt 更直接：先建立带 entity、unit、频率、位置等语义字段的 quantity 表示，再确定哪些量兼容，最后进行算术/范围判断。固定作者源码为 `3f19e77c119940ebcfb4b77d19bbdd82180faa01`，MIT。完整旧实现依赖 CoreNLP、Gurobi 7.5 与很旧的科学计算栈，不适合整包引入；值得复用的是兼容先于计算的结构、显式证据组合和可审计推导，而非照搬其词典匹配器。[作者仓库](https://github.com/AbhilashaRavichander/EQUATE/tree/3f19e77c119940ebcfb4b77d19bbdd82180faa01)、[quantity 结构](https://github.com/AbhilashaRavichander/EQUATE/blob/3f19e77c119940ebcfb4b77d19bbdd82180faa01/numset.py)、[依赖](https://github.com/AbhilashaRavichander/EQUATE/blob/3f19e77c119940ebcfb4b77d19bbdd82180faa01/requirements.txt)。

最小算法扩展可以复用现有 table_relations 的原文对象—值偏移，再接 Pint 的单位 registry/换算以及精确有理数或 interval 运算：每次输出使用了哪些原始行/quote、单位变换和关系结果。Pint 先以已核查的 `0.25.2` 为固定参考版，不声称这是最新版，许可为 BSD-3-Clause；`Quantity.to()` 检查量纲相容并换算，但它不理解产品对象或自然语言蕴含。能直接处理显式单位换算与可确定的上下界，但不能把“同单位同数字”当作整句蕴含；文档和附件同为 MB 仍是不同对象，默认 TTL 与 Enterprise TTL 仍受不同条件约束。没有建立对象、条件、版本和整体/单项关系时，数量层只能返回未决，不能自行补出合计公式。[Pint 固定转换合同](https://pint.readthedocs.io/en/0.25.2/getting/tutorial.html#converting-to-different-units)、[Pint 许可](https://github.com/hgrecco/pint/blob/0.25.2/LICENSE)。原论文也明确数量全部匹配是蕴含的必要条件而非充分条件；这是该复用的核心限制。[Q-REAS 论文](https://aclanthology.org/K19-1033/)。

范围比较还必须区分“某个实际值落在区间里”和“声明的上限相同”：观测到 5 不等于证明容量上限为 5，最多 7 与最多 28 的关系也不能脱离主语和谓词解释。先支持精确量、显式比较符和明确条件的有限操作集，保留运算轨迹；不执行模型生成的任意 Python/SymPy 表达式，不把代码执行本身当语言理解。这里是方法适配与计划中的库复用，尚未实现新的数值 evaluator。

对于已确定语义的实际取值区间，来源集合 S 包含于声明集合 C 可以支持数值分量，S 与 C 不交可以反驳，部分相交仍未决；许可上限、容量与实际取值需要先区分。计数与百分比可能都无物理量纲，仍须核同对象和分母。当前只有少量比较符时，Pint 加标准库的有界区间比较比引入整个 CAS 更小；若以后需要更复杂集合，可用 SymPy 的显式 Rational/Interval 构造与 `is_subset`，不使用对原始模型文本的 `parse_expr`，后者官方明确使用 eval。本轮没有安装这两个库。[SymPy 区间与包含关系](https://docs.sympy.org/latest/modules/sets.html#sympy.sets.sets.Set.is_subset)、[解析器边界](https://docs.sympy.org/latest/modules/parsing.html)。

## 下一轮最小验证

Selene guided 2-call probe 与 29 条完整对照现已完成：schema 路径可用，但 24 条 prose 的语义对应情况与 v2 相同，详见 [实测记录](../archive/atomic_judge_validation.md)。后续两种新模型仍读取同一 29 audit，不改 query/answer/source、不传 sidecar 标签；每个模型使用自己的固定原生格式，格式差异和 tokenizer 覆盖写入协议。Lettuce 当前接口没有完整对话语境槽位，应明确记录 history 的适配方式；不能把旧 assistant 文字拼成“来源”以凑相同输入。本轮未启动这些新模型。

两种新判别器都不能输出 Atomic 的完整五类 verdict 或五维分数，故先报告它们实际覆盖的 supported/unsupported 或 groundedness risk，并分别列出联合来源、脚注、量词、单位、历史污染的正例误伤/反例漏检。缺少三分类或拒答可答性时标未覆盖，不用二分类结果补出缺失维度。保留所有失败、超限和无分项，代码两例仍与 prose 分开。真实三臂逐来源做 assistant audit，不能填成人工评分。

如果数量层随后实现，先用单独的 assistant 合成变体检验对象、单位、比较符和条件组合；数值/对象不固定成当前 1/10/11 MB 或 7/28。开发可迭代，但不能在已看 29 条上训练后再把相同样本当独立收益。冻结适配与聚合后才进入此前未见的正式验证。本轮不会启动下载或接入默认评审 gate。
