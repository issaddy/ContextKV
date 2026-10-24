# 来源绑定的回答生成：设计与验证路线

**当前执行顺序：** 本文保留候选机制与已核上游依据；全局审查后先以共同提示的 D/N/P 小诊断检验额外规划，暂不启动原全159三臂。最终回答先保持原生文本/IDs不改写，宿主绑定出处和两阶段业务事务按机制收益决定。见[研发策略](research_strategy.md)及[集成边界](source_bound_generation_integration.md)。

2026-10-24。**首选方向是在生成前选定有原文位置的支持单元，再由程序携带其出处，而非在自由回答后补一个引用评分。** 这是待验证的机制设计，尚未实现或取得质量收益。它延续 AFTG / ReClaim 的已有方法，并针对本项目完整 QA、稳定证据身份、条件依赖和多轮错误传播做适配，不预设贡献只能停留在当前三条路线。

## 当前证据指向哪里

[B6 完整质量审查](b6_quality_validation.md)已见：答案给出正确的附件上限，却把原句引到没有该数值的文档，Selene 仍给高分。[P3 来源审查](current_topic_fusion_p3_validation.md)还保留产品适用性与无据扩写；[API 原文包实验](api_evidence_validation.md)表明，补入正确方法声明也未阻止答案编造其他调用。输入正文与引用映射正确，不足以约束输出主张。

本次与 [pipeline 根因审查](../runs/quality-next-review/README.md)对接的三个缺口是：**支持单元选错、转述扩大条件、正确事实错绑出处**。MiniCheck 增补、table focus、source topic、query fusion 和 ConvSearch 各有已有作用，但没有完整连接“本句选择的依据”与“本句实际输出的内容/引用”。格式约束也不能修复语义判断。研究原件、源码与许可快照见 [quality-next-research](../runs/quality-next-research/README.md)，没有为本设计下载或运行新模型。

## 1. 优先实现：支持片段计划与确定性出处

### 理论依据与可复用代码

[Attribute First, then Generate（AFTG，ACL 2024）](https://aclanthology.org/2024.acl-long.182/)先选择来源片段，再聚类规划句子并生成；一个句子可以联合多个片段。[ReClaim（NAACL Findings 2025）](https://aclanthology.org/2025.findings-naacl.55/)交替生成原文 reference 与 claim，reference 受 token trie 约束，claim 仍是模型生成。两者提供过程分解和可追溯支持的依据，不保证任何转述都被来源蕴含。

| 固定作者实现 / 许可 | 可复用入口与必须处理的差异 |
|---|---|
| [AFTG `cc7b995dfb288da7982ae388994f182bef030b75`](https://github.com/lovodkin93/attribute-first-then-generate/tree/cc7b995dfb288da7982ae388994f182bef030b75)，Apache-2.0 | `few_shot_experiments/run_full_pipeline.py` 有 selection→clustering→逐句生成及 selection→fusion 分支；Highlight 契约含文档、句子、字符 span。复用分解/数据思想及适合的局部函数，不直接运行默认外部 API pipeline |
| [ReClaim `bc997dda1340eb6facaf1787294bad3377754914`](https://github.com/pdxthree/ReClaim/tree/bc997dda1340eb6facaf1787294bad3377754914)，Apache-2.0 | `generate/Trie.py` 是独立的 49 行 token trie；若已有 XGrammar 单元 enum 能完成同一合同，就复用现有约束器，避免重复实现。主脚本不是 Qwen/SGLang 插件 |

源码核查发现几个真实移植问题：AFTG 的 [`adapt_highlights_to_doc_alignments`](https://github.com/lovodkin93/attribute-first-then-generate/blob/cc7b995dfb288da7982ae388994f182bef030b75/few_shot_experiments/subtask_specific_utils.py#L207)会去掉空白/标点匹配，并把错文档片段改绑到第一个命中文档；重复文本和不同版本不能这样处理。其溢出分支会删正文，逐 token copy processor 又包含 CPU 扫描/重新分词，本项目不照搬这些路径。ReClaim 的 [`inference_unified.py`](https://github.com/pdxthree/ReClaim/blob/bc997dda1340eb6facaf1787294bad3377754914/generate/inference_unified.py)硬编码 Llama token IDs、假设编码首 token 可删除，并修改传入 reference list。移植应保留纯约束机制，重新绑定实际 tokenizer/来源身份；本次未取得并验证其 SFT 权重，不能声称完整复现。

较新的 [CAGE（2026-07 v1）](https://arxiv.org/html/2607.24236v1)把答案单元与支持文档显式连成 map，再训练两个阶段的模型。这是直接近邻：固定单元语义后可减少引用分配自由度，但不证明单元语义正确。尚未确认作者可用代码/权重；同名 attribution-graph 仓库属于另一工作，不能混用。当前只借其支持关系表示的思想。

### 最小输入、计划和输出合同

令 `S` 表示“答案单元→来源 span 集合”的计划。先求 `S`，再实现答案，并由宿主计算 `cite(S)`，可分离证据选择、转述和引用身份三个错误通道。给定真实支持关系时，支持集合选择可视为带依赖的覆盖问题；现实中的支持/意图有推断误差，不能据此声称最优选择或事实保证。

1. **来源集合。** 使用当前完整可见的外部 EvidenceBlock/manifest；完整实际前序 QA 和全部原正文在两次调用中保留。历史助手文本可帮助理解问句，不能变成外部支持单元。不得输入未来回答、reference 或 qrels。
2. **原文单位。** 复用现有 prose/表格 row+header+footnote/API 声明与显式依赖；每单元保存原 EvidenceKey/ref 与连续字符 span 列表。多个 span 可构成联合单元，依赖只由原文/已有显式元数据提供。相同文字的不同来源不合并；没有明确结构就保留原段落，不猜父条件。局部 unit 序号只供本轮枚举，不新增 hash，不替换历史稳定 E_ID。
3. **第一次调用。** 输出 `groups: [{support_units: [合法 unit_id…], role: direct_answer | necessary_qualification}]`，另可输出 `coverage: supported | partial | insufficient`。枚举约束禁止虚构 source/span；模型仍可能合法错选或误判 coverage。允许联合来源、合法空计划；格式/length 失败保留且不重试，不能伪装成空计划成功。
4. **第二次调用。** 模型对既定 group 生成句段，或输出明确未覆盖段；宿主在句段后附该组的稳定 E_ID。逐字 quote 由宿主复制 `source[start:end]`，不让模型重写引文或再猜引用 ID。原响应、计划、最终渲染都保存。未覆盖段若含实质事实，仍计入完整质量分母，不能因状态标签而免责。

结构合同只有：`rendered_refs(segment) = refs(selected_group)`，且 `rendered_quote = source[start:end]`。**它不推出 `source ⊨ generated_text`。** 合法组仍可能属于错误产品/版本，遗漏必要条件，无法支持复合句，或被转述成无据包含关系/性能保证；正确 quote 也不代表回答了用户问题。完整 schema、边界与渲染规格见 [原型规格](../runs/quality-next-research/prototype-spec.md)，全开发材料的结构覆盖和最小接口见[支持单元设计](support_unit_design.md)。

与已有方法的增量应具体归因：source_topic/ConvSearch 选择检索语境，本机制选择回答依据；table focus 提供结构单元，本机制继续绑定最终出处；MiniCheck 在事后提出信号，本机制改变生成过程。仍然保留这些原有对照，不将其重新命名为新贡献。

### 三臂控制额外调用

下列是拟冻结的原型预算，不是根据答案效果选出的参数。三臂共享模型/版本、完整材料、历史、温度、正常 EOS 和最终 cap1024。不得用静默增加输出预算来减少截断。

| arm | 第一次调用 | 最终调用 | 用途 |
|---|---|---|---|
| A0 原稿 | 无 | 原 Q0 单次生成，cap1024 | 原方法与完整成本锚，不为凑调用数强迫额外计算 |
| A1 普通两段 | 普通 notes/提纲，cap384；同样可见原文及单位表 | 依 notes 自由生成，cap1024 | 控制多一次调用/更多思考的收益 |
| A2 来源计划 | 合法支持单元分组，cap384 | 按组实现并由宿主携带出处，cap1024 | 检验来源计划与绑定的组合机制 |

A1/A2 同阶段预算，不代表实际 token/耗时相同。必须计入所有中间输入/输出、grammar、宿主处理、额外 prefill、完整完成时间与失败。若 A2 优于 A1，尚不能把计划、结构约束、宿主引用分别报为独立收益；后续固定同一计划/实现文本再切换自由引用与宿主引用做消融。另可直接把同一选中计划渲染成 extractive 支持回答，作为无需第二次模型调用的强简单对照：若它同样有效且更便宜，复杂转述没有得到支持。

## 2. 同一支持表示上的条件与逻辑保持

[DRAGged into Conflicts（2025-06 v1）](https://arxiv.org/html/2506.08500v1)区分同义、互补、意见、时效和误信息冲突，说明不同值未必应相互排斥。作者仓 [`81ba921dd684a93db41a7e9dda6b6a7c67348a88`](https://github.com/google-research-datasets/rag_conflicts/tree/81ba921dd684a93db41a7e9dda6b6a7c67348a88) 为 Apache-2.0，但只发布 README/LICENSE/数据，没有可运行 detector；复用的是分类与评价思想，不是不存在的 pipeline 代码。

可检验扩展是在已有单元中保留对象、谓词、数值/单位、逐项或合计、显式条件/版本和原文位置。只对已明确字段做兼容判断：相同对象/范围且条件重叠的互斥值才构成候选冲突；不同对象的值并列，未知条件不自动合并。输出受保护的原值、表头、明确操作符及 API 声明由宿主投影，其余自由转述继续单列未验证。不能用任务、产品或已知数字分支造成功。

这比再加一条“注意条件”提示更具体，但不是通用自然语言证明器。固定同一支持计划，比较是否投影原关系，检查 `one/either/both`、单项/总量、版本、精确单位换算、联合来源和缺条件；逐项统计范围强化、新增错误、遗漏与回答性。CPU 只能验证复制/操作合同，不能证明模型选对了关系。它可复用路线 1 的数据结构，避免另建图服务。

### 新模型诊断后的最小适配决策

Qwen3同输入诊断进一步说明，应分开问：**来源是否支持某个断言，以及这个断言是否适用于当前问题。** 例如一般心脏健康页面中的减重比例不自动适用于儿童，某时期的数值不能与另一图的时间区间拼接；有原文的安全建议也不自动成为法定义务。[Govt逐轮来源审查](../runs/qwen3-reference-semantic-review/govt/README.md)保留这些反例及32B修复部分年龄/地域条件的正例，属于assistant审查，不能当人工评分或总体优胜结论。

[JCA（2026-10-19 v1）](https://arxiv.org/html/2610.27114v1)提供直接可借鉴的分解：从问题抽取共享要求，对各候选按同一要求检查，再联合比较理由。作者[MIT代码 `3cb4413e4b305893d3206a302eea29954279b66e`](https://github.com/jenhsia/joint_constraint_aggregation/tree/3cb4413e4b305893d3206a302eea29954279b66e)的阶段prompt与检查表示可复用；它只验证正确候选已在集合内的选择，不能视为多轮长答、候选召回或转述蕴含的证明。其题目条件与候选属性明确关联的要求，比材料同段共现或按支持票数选值更贴合本项目范围错误。

**优先验证的最小方向：将共享问题要求与来源支持组的对应关系作为A2第一次计划的候选组织方式。** 要求来自当前问题及可核的前序用户语境，历史助手文本不能提供新的事实条件；支持组保留原对象、期间、分母和显式例外的span。适用性未说明时保留未知，不补条件或硬选单一赢家。第二次调用仍在完整QA/全部正文上实现回答，由宿主携带原出处。这里尚未冻结新增schema或实施模型调用，也不预先保证384-token计划容纳全部关系；计划length/遗漏必须保留为失败证据。

这不是原JCA协议的复现。原staged选择需要 **n+2次调用**（抽取、n个候选检查、聚合），merged-check也有 **3次调用**；如再生成长答或生成候选，还须另计。把这些操作合入现A2的一次384-token计划，再用1024-token实现，是需独立检验的简化，不能继承其准确率或成本结论。先保留既有A0/A1/A2与同计划extractive强对照，不扩大调用网格；若仅多了一次思考、合法ID增加而范围/回答性不改善，假设不成立。正文与完整QA不删除，因此也不预设节省prefill。

[Claimify（ACL2025）](https://aclanthology.org/2025.acl-long.348/)的上下文消歧与[VeriTrail（固定v1；后发表于ICLR2026）](https://arxiv.org/html/2505.21786v1)的来源回溯适合**定位错误阶段**：原来源是否支持计划、原来源与计划是否支持实现句。抽取声明忠于原答案不等于原答案忠于来源；局部选句后仍需恢复其所属原块及条件。可复用的机构代码是[Microsoft `eval-recipes` MIT快照 `dd07383fb6045448139a288917a92f621cfe1372`](https://github.com/microsoft/eval-recipes/tree/dd07383fb6045448139a288917a92f621cfe1372/src/eval_recipes/evaluations/claim_verification)，其README明确不是两篇论文的官方复现。这里只借上下文/offset与阶段追踪合同，不安装另一个judge替代独立质量审查，也不照搬其概率阈值或排除声明的分母。

已核源码有具体移植边界：JCA模糊候选回绑去标点，可能把`10.5 MB`映到`105 MB`，且预算路径会截正文；应沿用本轮unit/group身份及完整材料合同。Microsoft适配的`ClaimExtraction.run()`把本句而非输入excerpt传给后续消歧/分解；格式解析失败还可能成为“无声明”。[固定源与CPU反例](../runs/source-bound-claim-research/README.md)已保存，不能因采用方法名就假定条件已经保全。后续仍按完整159轮及全部未覆盖/截断检验，已见反例只帮助解释，不按任务名或已知数值写特例。

## 3. 独立路线：上下文对比解码

[AdaCAD（NAACL 2025）](https://aclanthology.org/2025.naacl-long.581/)用有/无外部上下文的 token 分布 JSD 调节对比强度，区别于统一固定权重。作者 MIT 代码 [`57508c3ec77f2ffc58e002afb55b0ad1846f3f9b`](https://github.com/HanNight/AdaCAD/tree/57508c3ec77f2ffc58e002afb55b0ad1846f3f9b)中的 `get_jsd` 和 logit 混合可复用；原脚本的 distributed 双分支、阈值 floor、batch 平均和采样假设须明确适配。它可能减少“外部材料已足够，却仍沿参数/历史错误续写”，但分布分歧不是来源真实性或蕴含判断。

先用同一已有权重、同一输出前缀的独立双 KV runner 比较 greedy/固定 CAD/AdaCAD，再决定是否值得原生接线。含实际 history 的 prior 分支是多轮适配，应与作者纯问题 prior 区分。每 token 两次 forward、双 KV 和完整成本都计入；有限 top-logprobs 不是完整分布，不能拼出不等价实现。错误来源、材料不足和 no-conflict 回归与已知错误一起保留。

较新 [CoCoA（2025-08 v1）](https://arxiv.org/html/2508.17670v1)列为相关替代而非自动升级：作者 MIT 快照 [`1f776c9108edd5f673d9ffa21c13392af8c0e1ba`](https://github.com/infusion-zero-edit/CoCoA/blob/1f776c9108edd5f673d9ffa21c13392af8c0e1ba/group_decode_fileio_CoCoA.py#L198)把算出的 adaptive weight 覆盖为 0.5，并加入额外 logit 放大。该代码不能直接标为论文公式的忠实复现，作者双 GPU 时延也不能移作本项目单卡开销。

## 4. 区分来源隔离、评分与可计算证据

新近的 [MADARA（2026-06 v1）](https://arxiv.org/html/2606.25191v1)值得作为机制对照：逐文档生成再投票，可隔离多来源干扰；作者用不评分的隔离对照区分额外评估的作用。它的优势取决于模型与任务，跨文档多跳时隔离可能破坏推理链。其Qwen3-8B结果也不能推导本项目应选择同一种处理。RSC扰动相关性不是回答正确率，路由中的No-Filter准确率仍需要有标签的离线校准，不能称整个路由无需标签。

作者 [MIT实现 `a22bce0b11e514dd5eb4e89a2069a9afef45f942`](https://github.com/js-lee-AI/MADARA/tree/a22bce0b11e514dd5eb4e89a2069a9afef45f942)已核对7个原始Git blobs，尚未接入本项目。可复用的是隔离对照和纯排序/聚合思想；源码有必须适配的具体差异：

- `generate`固定64-token事实短答，不携带多轮历史；PDE用标准化字符串聚合，不适用于本项目带条件、代码和引用的长答案。
- Conflict Detector提示输出`doc_adjustments`，但通用`_parse_score`要求`score`，失败后取文本首个0–5数字；可能把文档ID当分数。它不是一个可直接信赖的三评估器接线。
- 去标点会把`10.5 MB`和`105 MB`合并；SDA双`argsort`给并列分数不同ordinal rank。旁路材料截200字符、RSC文档截2000字符，也不满足当前完整原文合同。[可复算CPU反例](../runs/quality-next-research/madara/contract-witnesses.json)只证明这些接口行为，不重判论文的模型结果。

本项目可检验的适配是**来源隔离的支持提取 → 联合来源计划 → 完整材料上的回答实现**，允许联合多来源，并保留未决适用范围；不能把短字符串多数票当事实保证。它与一次全材料支持计划做同材料比较，另设不评分的隔离对照，计入每个来源调用及最终综合成本。这是候选后续实验，不马上把五个来源扩成十五次打分。若多来源问题的收益来自隔离，就不把它归给复杂评分；若孤立丢失必要条件，应恢复联合组而非提高投票阈值。

[RECAST（2026-10 v1）](https://arxiv.org/html/2610.10507v1)进一步把检索、关系运算和自定义计算视为证据构造动作：来源记录经操作得到派生证据，再回答。可借鉴的是保存操作、输入来源与派生结果的合同；BM25/BGE-M3/SQLite已有本项目对应组件。其主要方法训练RouterLM（SFT/GRPO），依赖CompilerLM，且保留证据最多4000字符；不是给当前通用Qwen3加提示就能复现。尚未核实作者公开可用的代码/训练权重，因此只作原理参考。应先在确需计算的开发问题上核实际需求，保留派生证据的原始来源、算子和条件；执行正确不等于输入适用，也不证明结果足够回答。不能用未知作用范围的数字进行自动合计，或把4000字符选择静默搬进完整正文基线。

## 验证顺序与失败标准

**在扩机制之前控制生成器能力。** 当前大多数既有失误来自固定Qwen2.5-7B；AFTG包含few-shot/微调实验，ReClaim完整方法也有训练，仅把其流程提示给通用7B不等于复现原效果。用户已提供 `/media/ssd1/qwen3-8b` 与 `/media/ssd1/qwen3-32b`，当前改用这两个现成本地模型，先前Qwen2.5-32B下载方案未执行。[完整Qwen3参考](qwen3_reference_validation.md)现已完成8B全159轮P2及32B同实际历史重放，318份输入逐token/模板重建通过；质量判断仍需分析。两者同系用于减少架构/协议变化，不把规模当唯一差别，也不预设32B可靠。若强原稿已解决主要问题，先比较直接换基座与小模型附加机制的完整质量/成本；若仍失败，再做模型与机制的交叉。

本地只读核验确认两者均为BF16稠密Qwen3ForCausalLM，36/64层、均8个KV heads且显式head_dim128，5/17个权重分片齐全。真实分词器/模板加载已通过，两目录的backend与模板fingerprint一致；`enable_thinking=False`的输入渲染也与原生模板调用一致。已完成独立stock引擎的318次开发集能力参考，尚无质量结论；未确认权重的上游commit，不用目录名或自行生成的40位hash冒充HF revision。先用32768窗口，不依tokenizer元数据131072启用未经验证的长上下文。

采样也应适配模型原理：[官方8B](https://huggingface.co/Qwen/Qwen3-8B#best-practices)与[32B模型卡](https://huggingface.co/Qwen/Qwen3-32B#best-practices)给出non-thinking的temperature0.7/top_p0.8/top_k20/min_p0建议。`870a250`已为原生客户端实现这四项显式可选设置并保留旧默认协议；新能力运行采用server seed42、c1与固定请求顺序，不声称逐请求确定性。固定引擎对request seed的使用依赖deterministic/backend路径，不能只给HTTP加一个seed就承诺精确复现。思考开关通过模板设置，不以删除响应内容代替；单个seed的开发比较也不当作稳定总体收益。现有launch、运行身份校验和原生基线manifest仍固定7B，先复用固定stock引擎及完整input-ID客户端做隔离参考，随后参数化扩展工程合同。不同模型的实际池、显存、标定和质量资格分别记录。[完整能力审查](../runs/quality-next-research/generator-capability-review.md)保留原方案、LongCite归因训练边界与新Qwen3决策。

1. **先完整 P2。** metadata-only [开发 cohort](../runs/quality-next-review/full-dev-cohort.json)为 **22 会话/159 轮**：Cloud 5/41、clapnq 6/47、fiqa 5/32、govt 6/39。先固定上述Qwen3-8B、每轮重建、上下文/输出预算与实际完整历史基线，再做32B同历史能力参考，不马上乘上13个缓存格。官方固定top5是输入协议，**不是oracle，也不保证每题来源充分**。旧Cloud41只作已见回归，不代替全开发集；新Qwen3批次不与旧Qwen2.5批次冒作单变量配对因果对照。
2. **冻结历史定位机制。** 保存基线完整实际前序 QA，在每轮相同历史/材料上比较三臂与 extractive 简单对照；先核 span/身份/依赖及语义错误，再做全159自有历史闭环，观察错误传播。时间比较必须另控制执行顺序和资源状态，不能把旧基线运行与新 replay 的差值冒充因果加速。
3. **随后共同 P3。** 各臂共享语料、索引、query view、检索器、候选/总 rerank 预算；完整计入检索、规划、实现成本。来源不足/足够分层在生成后独立审查，不把该标签送入在线，也不同时改检索与生成后把收益归给单项机制。
4. **再联合 B/C 与冻结 test。** 主表保留全部轮次、失败/length/未覆盖，按会话聚类并分域；重复不增加独立会话数。分别报告任务覆盖、对象/条件/版本、cited/visible 支持、机械出处、代码范围、完整成本。模型五维评分是辅助证据，已知关系/引用漏判须单列，不能以格式通过或同一模型自评分选赢家。全部22会话仍是 dev，方法和协议冻结后才使用 heldout，本设计没有读取它。

若正确 ID 增加但范围/回答性不改善，若 extractive 简单法同样有效且更便宜，或双分布只增加复制并放大错误来源，相应假设就被削弱。应据此改进支持选择、转述或生成机制，而非用更多合法引用、更多评审器或更严格格式掩盖失败。
