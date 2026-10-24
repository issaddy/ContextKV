# 从伪实体与产品—对象漏选出发的可复用方法

2026-10-24，只读官方论文、作者代码和模型元数据；未下载权重、安装新环境或修改core。当前先等八种查询方法的真实检索结果，再决定实现次序。这里提出的是可证伪的开发对照，不把抽取模型的NER/关系指标转述成ContextKV质量提升。

已见问题有两层：大写span召回把By、To、跨段`Addressing Storage Space Limits\n\nTo`当作候选；而即使目录同时含Cloudant与Attachments，selector仍可能只取对象。前者需要语义类型与原文边界，后者需要当前意图、产品—对象关系和相关历史选择。仅升级JSON enum不会解决；NER也不会自动解决后者。依据见[同41轮v1/v2验证](source_topic_selector_validation.md)。

## 三个有价值但职责不同的候选

| 候选及固定公开工件 | 可复用能力 | 本次优先级与限制 |
|---|---|---|
| **GLiNER2.5 small/base**。作者代码[`fastino-ai/GLiNER2@ca4f68235b50fe4c3ecf3876851d748fe4b15c6c`](https://github.com/fastino-ai/GLiNER2/tree/ca4f68235b50fe4c3ecf3876851d748fe4b15c6c)；small权重[`df5910e44bc4ffdb0d95399a83b0ca4516349aa5`](https://huggingface.co/fastino/gliner2.5-small-v1/tree/df5910e44bc4ffdb0d95399a83b0ca4516349aa5)，base权重[`54785d51df8d86a0d8f2eb212fec15871aad2c4d`](https://huggingface.co/fastino/gliner2.5-base-v1/tree/54785d51df8d86a0d8f2eb212fec15871aad2c4d)；代码和模型均Apache-2.0、模型非gated | 用户给类型/描述，模型返回原文实体span；2.5另有typed JointIE，将候选实体与关系一起解码 | **最先值得验证的候选提取替换**。small为73,881,879参数，base193,581,591；可以直接用作者推理库，无需训练。语义类型可能减少连接词/标题污染，但尚无本任务实测，不能保证不漏Cloudant、Carthage或普通名词Attachments |
| **GLiNER-Relex large-v0.5**。作者GLiNER代码[`f3c945702b9b5cde381c27b95310c47909bcfd80`](https://github.com/urchade/GLiNER/tree/f3c945702b9b5cde381c27b95310c47909bcfd80)；权重[`9c4171ae1e690fc29b87f33579e50bcd65faf2cc`](https://huggingface.co/knowledgator/gliner-relex-large-v0.5/tree/9c4171ae1e690fc29b87f33579e50bcd65faf2cc)，均Apache-2.0、非gated | 一次encoder前向联合提实体和关系，对被识别实体对与指定关系标签评分；返回关系两端的字符span | **产品—对象关系的独立比较**，不必立刻与GLiNER2.5同时部署。467,758,080参数，比small明显重；固定checkpoint的max_len1024、max_width12及实体对开销需要处理。不能把NER正确率或有关系边当作来源事实成立 |
| **EXIT**（ACL Findings2025）。作者源码快照[`4c0141a7f19c192f1be24a2c83e672078d6ba29b`](https://github.com/ThisIsHwang/EXIT/tree/4c0141a7f19c192f1be24a2c83e672078d6ba29b)；公开LoRA[`doubleyyh/exit-gemma-2b@e286ce9096178a4aefb743d0919705250fdf1b23`](https://huggingface.co/doubleyyh/exit-gemma-2b/tree/e286ce9096178a4aefb743d0919705250fdf1b23) | 对问题、完整所属段落、候选句作Yes/No末位logits判别，按原顺序保留有用句；比孤句相似度多保留父语境 | **历史/来源候选选择的相邻机制**，原论文做RAG文档句选择，不能称CQR复现。adapter为Apache-2.0；base是[`google/gemma-2b-it@96988410cbdaeb8d5093d1ebdc5a8fb563e02bad`](https://huggingface.co/google/gemma-2b-it/tree/96988410cbdaeb8d5093d1ebdc5a8fb563e02bad)，受Gemma许可且gated。固定作者code未发现LICENSE，故不直接复制代码；如采用，可依据论文/模型卡合同独立薄适配并先核许可。成本与可取得性使它排在小型抽取之后 |

模型参数和revision来自本次HF官方API记录；可重查元数据在`runs/source-topic-extraction-research/`。GLiNER2.5 small/base的F32参数体积约296/774MB，Relex约1.87GB，仅是参数字节量，**不含激活、tokenizer和执行器，也不是实际RAM/显存测量**。CPU支持不等于本机低延迟，当前没有这些模型的本地性能数字。

## GLiNER2与2.5：可直接复用的具体边界

[2025 GLiNER2论文](https://aclanthology.org/2025.emnlp-demos.10/)是schema驱动的encoder多任务IE；2026-08作者发布的[2.5](https://fastino.ai/blog/gliner2-5-span-free-information-extraction)将固定宽度span枚举改成边界预测，新增联合解码等接口。两者不可混称同一模型。旧`fastino/gliner2-base-v1@f9634218e53580c56edf0de97ca1a7d3f1c2354e`仍是span架构、max_width8，不能用它加载后宣称测试了2.5。2.5通过`AutoExtractor`选择boundary实现；`GLiNER2.from_pretrained`仍只支持旧span模型。[固定README](https://github.com/fastino-ai/GLiNER2/blob/ca4f68235b50fe4c3ecf3876851d748fe4b15c6c/README.md)

**先用实体接口，关系另作消融。** `batch_extract` / `extract_entities`支持类型描述、`include_spans=True`与confidence。可预先声明`software_product_or_service`、`integration_tool_or_sdk`、`data_object_or_feature`、`product_version`等类型，召回产品、工具和普通名词对象；不把Cloudant、Attachments或已知错误列表写成特定白名单/黑名单。这些标签是本项目待验证设计，不是作者给本任务验证过的ontology。

字符offset可接现有`SourceSpan`，不需要新hash或持久化库。固定word splitter在原字符串上匹配，随后只lower token值，专门避免Unicode lowercase改变长度破坏offset；JointIE结果也是原文半开区间。但仍需入口一次核对`text[start:end]`，保留原角色/turn/source键；不要先清洗引用或拼接段落再忘记offset映射。[word splitter](https://github.com/fastino-ai/GLiNER2/blob/ca4f68235b50fe4c3ecf3876851d748fe4b15c6c/gliner2/processing/word_splitter.py#L1-L58)、[JointIE结果合同](https://github.com/fastino-ai/GLiNER2/blob/ca4f68235b50fe4c3ecf3876851d748fe4b15c6c/tutorial/15-joint_ie.md#L160-L192)

**4096不是Qwen的4096 tokens，也不是无限历史。** 固定2.5 config的max_len4096与作者训练长度说明涉及word-token视角；API的`max_len`也是分词后的word cap。推理默认`None`并不自动保证在模型训练范围内；显式max_len会裁掉尾部，输出offset仍可合法，因此“offset有效”不能证明全输入已覆盖。必须记录word/subword/schema token数、实际编码范围；若分段，按原message/来源段边界显式分段并保留未覆盖与跨段关系未知，不能悄悄裁到4096后说完整三轮历史已处理。[processor裁切](https://github.com/fastino-ai/GLiNER2/blob/ca4f68235b50fe4c3ecf3876851d748fe4b15c6c/gliner2/processor.py#L670-L718)、[推理API](https://github.com/fastino-ai/GLiNER2/blob/ca4f68235b50fe4c3ecf3876851d748fe4b15c6c/gliner2/inference/runtime.py#L250-L334)

**JointIE可保结构，不能保真。** 它能约束关系两端必须是已选实体及合格类型，利用greedy/beam选图；不能证明“Cloudant—supports—Attachments”在给定来源成立，更不能证明当前用户正问这个产品。固定small/base配置已有`enable_relations=true`，可以在同一checkpoint上比较实体单独vs联合边。应保留空图/不可行状态、阈值和候选cap，不能把库的结构约束误称置信保证；不要先加“一对象只能属一个产品”这类会误伤比较问题的硬约束。[固定JointIE说明](https://github.com/fastino-ai/GLiNER2/blob/ca4f68235b50fe4c3ecf3876851d748fe4b15c6c/tutorial/15-joint_ie.md)

依赖方面，固定库`[local]`包括torch、transformers、peft、numpy、safetensors等，并非“零软件依赖”；不需要Fastino云API或训练服务。其声明允许transformers<6，但这不是与当前5.19运行栈已集成成功的证据。若后续选中，应独立可选依赖、固定库commit/权重快照，先做无网络加载与全41输入覆盖检查。[固定pyproject](https://github.com/fastino-ai/GLiNER2/blob/ca4f68235b50fe4c3ecf3876851d748fe4b15c6c/pyproject.toml)

## Relex与历史选择：为什么还需要不同方向的对照

[GLiNER-Relex论文](https://arxiv.org/html/2605.10108v1)用共享encoder表示文本、实体类型和关系类型，并对实体对打分。论文报告标准RE数据集表现，也明确实体密集时的二次方pair数量与长文限制；没有证明当前对话产品—对象解析、CQR或端到端GraphRAG有效。其`GLiNER.inference(..., return_relations=True)`可直接复用，关系输出含head/tail的原文位置；源资格和当前意图仍由本项目分开保留。[固定模型卡/API](https://huggingface.co/knowledgator/gliner-relex-large-v0.5/blob/9c4171ae1e690fc29b87f33579e50bcd65faf2cc/README.md)

固定Relex config使用DeBERTa-v3-large、max_len1024、max_width12；作者processor会发warning后裁尾，不能直接把19k-token的selector输入塞入并把未抽到实体归为模型语义判断。先声明每个原message/来源段的完整覆盖边界，或者报告超过上限未评估；关系跨段必须另测，不能从两个片段分别出现产品与对象就拼出支持边。[配置](https://huggingface.co/knowledgator/gliner-relex-large-v0.5/blob/9c4171ae1e690fc29b87f33579e50bcd65faf2cc/gliner_config.json)、[processor](https://github.com/urchade/GLiNER/blob/f3c945702b9b5cde381c27b95310c47909bcfd80/gliner/data_processing/processor.py)

**同名合并不是NER的附赠能力。** “IBM Cloudant/Cloudant”可成为alias候选，但同一个“Registry/Cache”可能属于不同产品；字符串相同、embedding相近或模型给了same-as边都不应直接合并来源scope。首轮可保留mention ID与全部occurrences，只有原来源明确别名/同指的文本才把合并作为可审计提案；其余并列不补造canonical实体。GLiNER2.5和Relex都不应被写成已经提供知识库级entity linking。

对真正的历史选择，核查了[HAConvDR](https://arxiv.org/html/2401.16659v1)：其算法用当前检索效果改善产生历史turn有用性标签，涉及qrels；传统历史还使用gold passage，检索历史passage的变体也不取消当前监督。固定[作者code `2c0dcf6`](https://github.com/fengranMark/HAConvDR/tree/2c0dcf65a926ac38a249f5cf2955ae08fb0ac17a)未找到可直接加载的trained selector固定权重或LICENSE。因此可借鉴**只在训练侧学习历史贡献**的设计，不能在线用当前dev qrels挑历史，也不把它列为已就绪模型。已有作者ConvSearch-R1独立基线继续执行，不因这项缺权重而停工。

[EXIT论文](https://aclanthology.org/2025.findings-acl.253/)则提供公开判别权重，可按完整父段+候选句保留上下文。把它用于相关历史句是本项目迁移假设：先用原用户句或先前可见source句形成候选，assistant句仅作意图/指代线索，不能因“相关”高分就变成事实证据。每候选复制完整父段，N句就是N份prefill；即使批量且只取Yes/No logits，不生成长rewrite，也可能比一次小encoder甚至原selector更贵。不能搬用论文对其他reader/vLLM设置的速度结论。它适合在八方法结果显示历史选择仍是主因时作后续对照，当前不建议先引入新的2B栈。

## 最小可证伪实验，待八方法结果后确定优先级

1. **固定同41轮，不挑恢复成功的轮。** 使用已有frozen prior-only inventory和真实QA。先只替换assistant候选提取，继续原3轮selector窗口、source资格、enum_v2、原query分组与RRF/总50重排预算；GLiNER实体模型看哪些原message、source模型看哪些原段，逐项记录。完整生成QA与required正文不动。
2. **分开消融。** 词法候选→GLiNER2.5实体候选是第一条；在同一实体结果上再给产品—对象关系是第二条。small与base只做明确的成本/召回比较，不同时调selector prompt、RRF、生成模板和阈值。Relex作为不同关系模型的备选对照，EXIT作为相关句选择的备选，避免一开始把三套系统全部叠加。
3. **原文与语义都看正反例。** 全41保留空结果、超长、失败、未覆盖；记录By/To/跨段标题误提是否减少，同时检查Cloudant、Carthage、版本号与普通对象是否被漏掉。添加assistant明确构造的合成最小对照：连接词By vs来源明确名为“BY”的工具；同名Cache分属两个产品；同产品别名；相邻段落标题与正文；明确切换工具、否定旧工具、同轮比较两个产品；实体保留但关系跨段缺证。合成expected只在离线侧，不进模型，不冒称人类标注。
4. **避免把错误数字变成新事实。** 只用past assistant定位名称/指代，关系支持单独指向原source span；故意改变历史错误数字，验证产品/对象提案是否无理由跟着变化。NER/关系confidence仅是模型分数，不当支持概率或正式quality标签。
5. **计算全成本与真实结果。** 记录新增模型的input word/subword数、窗口数、候选/关系数、覆盖率、CPU/GPU时间及selector tokens变化；全部排名完成后才读取dev qrels，比较R@5/nDCG@5、按会话改好/变坏和原八方法。仅删掉By却漏掉产品、低成本来自隐式裁切、关系加重跨产品污染、成本超过检索收益，都反证该配置有用。保持test未触及；全41已反复探索，不能作独立质量非劣证据。

因此当前最有价值的准备是一个可替换的**有原文位置的语义候选提供器**，随后单独验证产品—对象关联和相关历史选择。它可复用成熟开源推理代码与已有SourceSpan/证据库存，不需要重造训练框架、知识图谱或新增每请求hash。实现优先级由即将得到的完整检索结果决定。
