# 来源绑定的回答生成：设计与验证路线

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

结构合同只有：`rendered_refs(segment) = refs(selected_group)`，且 `rendered_quote = source[start:end]`。**它不推出 `source ⊨ generated_text`。** 合法组仍可能属于错误产品/版本，遗漏必要条件，无法支持复合句，或被转述成无据包含关系/性能保证；正确 quote 也不代表回答了用户问题。完整 schema、边界与渲染规格见 [原型规格](../runs/quality-next-research/prototype-spec.md)。

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

## 3. 独立路线：上下文对比解码

[AdaCAD（NAACL 2025）](https://aclanthology.org/2025.naacl-long.581/)用有/无外部上下文的 token 分布 JSD 调节对比强度，区别于统一固定权重。作者 MIT 代码 [`57508c3ec77f2ffc58e002afb55b0ad1846f3f9b`](https://github.com/HanNight/AdaCAD/tree/57508c3ec77f2ffc58e002afb55b0ad1846f3f9b)中的 `get_jsd` 和 logit 混合可复用；原脚本的 distributed 双分支、阈值 floor、batch 平均和采样假设须明确适配。它可能减少“外部材料已足够，却仍沿参数/历史错误续写”，但分布分歧不是来源真实性或蕴含判断。

先用同一已有权重、同一输出前缀的独立双 KV runner 比较 greedy/固定 CAD/AdaCAD，再决定是否值得原生接线。含实际 history 的 prior 分支是多轮适配，应与作者纯问题 prior 区分。每 token 两次 forward、双 KV 和完整成本都计入；有限 top-logprobs 不是完整分布，不能拼出不等价实现。错误来源、材料不足和 no-conflict 回归与已知错误一起保留。

较新 [CoCoA（2025-08 v1）](https://arxiv.org/html/2508.17670v1)列为相关替代而非自动升级：作者 MIT 快照 [`1f776c9108edd5f673d9ffa21c13392af8c0e1ba`](https://github.com/infusion-zero-edit/CoCoA/blob/1f776c9108edd5f673d9ffa21c13392af8c0e1ba/group_decode_fileio_CoCoA.py#L198)把算出的 adaptive weight 覆盖为 0.5，并加入额外 logit 放大。该代码不能直接标为论文公式的忠实复现，作者双 GPU 时延也不能移作本项目单卡开销。

## 验证顺序与失败标准

**在扩机制之前控制生成器能力。** 当前大多数失误来自固定Qwen2.5-7B；AFTG包含few-shot/微调实验，ReClaim完整方法也有训练，仅把其流程提示给通用7B不等于复现原效果。用户已提供 `/media/ssd1/qwen3-8b` 与 `/media/ssd1/qwen3-32b`，当前改用这两个现成本地模型，先前Qwen2.5-32B下载方案未执行。先建立Qwen3-8B完整159轮P2基线，再用32B重放其相同实际前序QA、原文、Q0与最终cap1024；不启用旧cost planner。两者同系用于减少架构/协议变化，不把规模当唯一差别，也不预设32B可靠。若强原稿已解决主要问题，先比较直接换基座与小模型附加机制的完整质量/成本；若仍失败，再做模型与机制的交叉。

本地只读核验确认两者均为BF16稠密Qwen3ForCausalLM，36/64层、均8个KV heads且显式head_dim128，5/17个权重分片齐全。真实分词器/模板加载已通过，两目录的backend与模板fingerprint一致；`enable_thinking=False`的输入渲染也与原生模板调用一致。尚未启动两模型生成，未确认权重的上游commit，不用目录名或自行生成的40位hash冒充HF revision。先用32768窗口，不依tokenizer元数据131072启用未经验证的长上下文。

采样也应适配模型原理：[官方8B](https://huggingface.co/Qwen/Qwen3-8B#best-practices)与[32B模型卡](https://huggingface.co/Qwen/Qwen3-32B#best-practices)给出non-thinking的temperature0.7/top_p0.8/top_k20/min_p0建议。`870a250`已为原生客户端实现这四项显式可选设置并保留旧默认协议；新能力运行采用server seed42、c1与固定请求顺序，不声称逐请求确定性。固定引擎对request seed的使用依赖deterministic/backend路径，不能只给HTTP加一个seed就承诺精确复现。思考开关通过模板设置，不以删除响应内容代替；单个seed的开发比较也不当作稳定总体收益。现有launch、运行身份校验和原生基线manifest仍固定7B，先复用固定stock引擎及完整input-ID客户端做隔离参考，随后参数化扩展工程合同。不同模型的实际池、显存、标定和质量资格分别记录。[完整能力审查](../runs/quality-next-research/generator-capability-review.md)保留原方案、LongCite归因训练边界与新Qwen3决策。

1. **先完整 P2。** metadata-only [开发 cohort](../runs/quality-next-review/full-dev-cohort.json)为 **22 会话/159 轮**：Cloud 5/41、clapnq 6/47、fiqa 5/32、govt 6/39。先固定上述Qwen3-8B、每轮重建、上下文/输出预算与实际完整历史基线，再做32B同历史能力参考，不马上乘上13个缓存格。官方固定top5是输入协议，**不是oracle，也不保证每题来源充分**。旧Cloud41只作已见回归，不代替全开发集；新Qwen3批次不与旧Qwen2.5批次冒作单变量配对因果对照。
2. **冻结历史定位机制。** 保存基线完整实际前序 QA，在每轮相同历史/材料上比较三臂与 extractive 简单对照；先核 span/身份/依赖及语义错误，再做全159自有历史闭环，观察错误传播。时间比较必须另控制执行顺序和资源状态，不能把旧基线运行与新 replay 的差值冒充因果加速。
3. **随后共同 P3。** 各臂共享语料、索引、query view、检索器、候选/总 rerank 预算；完整计入检索、规划、实现成本。来源不足/足够分层在生成后独立审查，不把该标签送入在线，也不同时改检索与生成后把收益归给单项机制。
4. **再联合 B/C 与冻结 test。** 主表保留全部轮次、失败/length/未覆盖，按会话聚类并分域；重复不增加独立会话数。分别报告任务覆盖、对象/条件/版本、cited/visible 支持、机械出处、代码范围、完整成本。模型五维评分是辅助证据，已知关系/引用漏判须单列，不能以格式通过或同一模型自评分选赢家。全部22会话仍是 dev，方法和协议冻结后才使用 heldout，本设计没有读取它。

若正确 ID 增加但范围/回答性不改善，若 extractive 简单法同样有效且更便宜，或双分布只增加复制并放大错误来源，相应假设就被削弱。应据此改进支持选择、转述或生成机制，而非用更多合法引用、更多评审器或更严格格式掩盖失败。
