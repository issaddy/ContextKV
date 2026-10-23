# 原文表格行关系注释

`adapters/table_relations.py` 实现可选、无模型的表格辅助表示。它保留证据正文、版本和稳定引用，另输出对象、原列名、原值、显式单位及字符位置。用途是检验：在已经取回正确表格时，显式保留行关系能否减少跨对象的错误概括。解析成功不是质量改善证据。

## 复用依据和适配边界

[TabRAG 原论文 §2.2–2.3](https://arxiv.org/html/2511.06582v1)先把单元格与行、列关联，再由语言模型产生供检索使用的描述。其[作者实现 `VLMPrompts.vlm_table_prompt`](https://github.com/jacobyhsi/TabRAG/blob/c3aba4190f141737f1a52ed1bacce95bec337667/src/prompts.py)明确使用 row/column/value/units，固定在 `c3aba4190f141737f1a52ed1bacce95bec337667`。本适配借鉴这些字段之间的结构关系，重新编写确定性文本解析，不复制上游代码或提示词，不使用视觉解析、LLM 描述或重建向量索引。因此它不是完整 TabRAG 复现。上游[仓库许可证](https://github.com/jacobyhsi/TabRAG/blob/c3aba4190f141737f1a52ed1bacce95bec337667/LICENSE)为 CC BY-NC 4.0；论文页面另标 CC BY-NC-SA 4.0。

真实开发反例的正文已经是扁平文本，不能只实现 Markdown 解析：`ibmcld_00558-23465-25360` 和 `ibmcld_12904-23555-25450` 具有编号表题、`Limit Maximum Size` 表头和三行不同对象的数量。实现没有写入这些产品名、对象名或具体数值；测试保留原表摘录用于回归，并使用不同对象、值和单位验证通用行为。

## 接口与来源位置

```python
from adapters.table_relations import extract_table_relations, render_table_relations

tables = extract_table_relations(evidence.text)
supplement = render_table_relations(existing_reference_id, tables)
```

两者均为纯函数；无表时分别返回空元组、空字符串。调用者决定是否在当前输入附加 `supplement`，并对新增 tokens 使用现有候选预算和审计。它们不访问历史、答案、标签、模型或缓存，不创建 EvidenceKey，不改写 `EvidenceBlock.text`，也不计算新的内容指纹。

`SourceSpan(text, start, end)` 使用传入原段落的 Python Unicode 字符偏移，左闭右开，保证 `original[start:end] == text`，不是 UTF-8 字节偏移。`TableRelations` 保存表题、表头、整表跨度、关系与原注释；`TableRelation` 保存 `object/column/value/unit/row_span/notes`。全部是不可变 dataclass，可用 `dataclasses.asdict` 留存审计。正文内空白、Markdown 转义、大小写和数值写法不作归一化；仅排除单元格两端的格式空白。

支持两种明确结构：

- `markdown`：单行表头、标准横线分隔行、列数一致、首列行名非空。每个值单独关联其原行名和列名。缺失行名、多级/合并/不等列数的表不作猜测。
- `captioned_numeric_rows`：编号 `Table` 表题、单行非数值表头、至少两行连续的“对象＋末尾数量和显式单位”。末尾单位只识别明确的常见大小、时间、速率、频率、质量、长度、百分比和计数写法。不能恢复的列边界保留为完整 `flattened_header`，绝不把 `Limit Maximum Size` 强拆成两个猜测列。未知单位或混合不规则行回退为空注释，原文仍然存在。

数值和单位不换算，也不相加。Markdown 单元格有唯一显式数量时保留其单位；没有数量单位时，仅从列名明确的 `(unit)` 或 `[unit]` 后缀取单位。单元格中存在多个数量时不推断一个统一单位。源表本身若有明确的合计对象行，会照录该行，不采用“一律禁止合计”的规则。

相邻、显式标记的 `Note:`、`Remark:`、中文注释或脚注只作为 `source note` 原文保留。只有行中与注释开头存在完全相同的 `†`、`‡`、单星或 `[^id]` 标记时，才建立该行的 `notes` 关联。没有标记的注释不变成所有行的共同限制条件。遇到普通叙述段落就停止搜集，不跨越章节追找脚注，因此真实例中远处 Frankfurt 的地域脚注不会被接到大小限制上。

## 验证与实验

`tests/test_table_relations.py` 覆盖真实扁平表、相同数值不同对象、显式总量行、多列 Markdown、非大小单位、源偏移、CRLF/中文/转义竖线、空单元格、脚注关联及不规则表回退。代码块不会作为表格证据解析。

实验应固定检索配置、系统提示、生成模型与 revision、完整 QA、required 证据正文和输出预算，对比关闭/开启行关系注释；使用实际可见正文逐项检查对象、版本、条件和来源支持，同时统计新增 tokens、TTFT/完成时间及原候选动作变化。真实闭环生成历史可能随答案传播，不能把后续输入差异解释成单轮关系注释的独立因果效应。固定历史或保存输入的单轮对照可辅助定位，最终保留完整闭环验证。当前没有真实生成质量提升结论。

## 2025–2026 后续方法筛选

2026-10-24 核查范围包括 ACL/Findings 2025–2026、SURGeLLM 2026、ICLR 2025 的原始论文及论文链接的作者代码，关注不训练生成底座的结构表示、行选择、数值引用与选择性修订。下面是未迁入当前实现的机制，不是新增实验结果；与既有 TabRAG、FullCite 和对象适用性研究互补。

| 方法及原始证据 | 原理、具体复用入口 | 相对当前 helper 的价值与边界 |
|---|---|---|
| **CRAFT，ACL 2026**；[论文 v2 §3/§7/Appendix C–D](https://arxiv.org/html/2505.14984v2)；作者仓库固定 `7c4be45bb292f4e050c4237b4141ccc9e91ebe30` | [MIT 源码](https://github.com/coral-lab-asu/CRAFT/blob/7c4be45bb292f4e050c4237b4141ccc9e91ebe30/LICENSE)中 [`build_row_texts`](https://github.com/coral-lab-asu/CRAFT/blob/7c4be45bb292f4e050c4237b4141ccc9e91ebe30/src/craft_tabqa/core/text.py)将表题和各列名附到行值，并保留 table_id/row_idx；[`DenseReranker._rerank_mini_table`](https://github.com/coral-lab-asu/CRAFT/blob/7c4be45bb292f4e050c4237b4141ccc9e91ebe30/src/craft_tabqa/retrieval/stage2_dense.py)按问题与行向量的相似度构造相关行子表，再编码排序 | 可复用小型行表示/排序逻辑，在现有 BGE 上为**辅助注释**选焦点行，完整 required 正文仍保留。需给其 row_meta 增加现有来源和字符跨度；其 `_clean` 只适合派生检索文字，不能覆盖原文。原方案使用额外生成描述/查询扩展，部分最终排序调用外部 embedding；OTT-QA 回答实验使用 gold passage，不能当完整开放检索或本项目多轮收益。不会直接搬入这些 oracle 输入或另建整套索引。此处 CRAFT 指 cascaded retrieval，非同名 counterfactual 论文 |
| **TreeThinker / RealHiTBench，Findings ACL 2025**；[论文 §4 及限制](https://arxiv.org/html/2506.13405v1)；作者仓库固定 `5278fa224ce2b37e71fd73059107d146b4b428da` | [`Generate_Tree`](https://github.com/cspzyy/RealHiTBench/blob/5278fa224ce2b37e71fd73059107d146b4b428da/inference/answer_prompt_tree.py)显式表示行/列表头的层级、覆盖起止位置和父子包含关系，再把问题与表头树对齐。采用推理时流程，无需训练生成底座 | 当前 helper 不恢复多级或合并表头；可先在**固定来源本来保留 HTML/LaTeX 单元格跨度**的材料上构建带原文位置的 header path。不能从已有扁平段落凭空恢复消失的 colspan。上游默认多次模型调用增加时延，且部分提示假设问题必有答案，必须改成允许证据不足。README 声明代码 MIT、数据 CC BY-NC 4.0，但该固定树未发现独立 LICENSE 文件；本次只核查思想，尚未复制代码 |
| **SMART，Findings ACL 2026**；[正式论文 §3.3–3.5、Appendix B/C、Limitations](https://aclanthology.org/2026.findings-acl.1474.pdf) | 将表头语义展开与操作执行分开；使用 SELECT、COMPUTE、SORT、INFER、ANSWER 的结构化步骤，外部计算器/排序器验证后提供修订反馈。正式 PDF 未确认可固定的公开源码/许可证，因此为原理适配入口 | 值得复用的是“先指定取哪些源单元格，再核算操作”，不是复制输出格式。可为操作数绑定现有 ref、row/value span，并检查单位；无需改底座权重。其正式实验使用 70B/72B、4096 输出预算，不能继承为当前 7B 的效果保证。论文明确验证器主要处理算术/排序，**不验证高层逻辑或文本关系**；算对 1+10 仍不能证明附件是否包含在某个文档上限内 |
| **When LLMs Read Tables Carelessly，2026-06**；[作者论文 §3–5](https://arxiv.org/html/2606.32029v1)、[Amazon 作者页](https://www.amazon.science/publications/when-llms-read-tables-carelessly-measuring-and-reducing-data-referencing-errors) | 将错单元格/错列/伪造值与遗漏必要行分别评估，再用 critic 过滤候选或触发重新采样。可借鉴其“来源取值正确性独立于最终答案正确性”的分解 | 最贴近本例的是**逐 claim 的对象–列–值绑定及相关行覆盖检查**。论文不少过滤收益使用 Sonnet-3.7 加 gold answer 的上界评审与多候选采样；另有训练小 critic 的方案。没有将这些设置作为无训练在线模块直接复用，也未确认可下载官方 critic 权重。当前可用原表 tuples + 现有本地模型做不含 gold 的有限诊断，须单独检验误报和漏报 |
| **Structured Self-Correction，SURGeLLM 2026**；[正式论文 §3、消融和附录](https://aclanthology.org/2026.surgellm-1.19.pdf)；作者仓库固定 `dd169866bf066b6935cd850b7a82d72f791545da` | [`prompts.py`](https://github.com/FarseenSh/table-self-correction/blob/dd169866bf066b6935cd850b7a82d72f791545da/experiments/src/prompts.py)分离单元格、计算、逻辑、完整性检查；[`evaluator.py`](https://github.com/FarseenSh/table-self-correction/blob/dd169866bf066b6935cd850b7a82d72f791545da/experiments/src/evaluator.py)分别统计发现错误、修复成功、误报和把正确答案改坏 | 可复用评测分解及单次选择性复查协议，避免只统计修改次数。论文观察到模型/任务相关的收益与过度纠正；不能从论文中推导当前 7B 必获益。README 的部分数字与最终论文不同，以正式论文为准，不借用其阈值作本项目门槛。README 声明 MIT，但固定树未见独立 LICENSE；当前未复制模板或评分代码，其短答案匹配也不能直接用于本项目开放式回答 |

还核查了 [Table-Critic（ACL 2025）](https://aclanthology.org/2025.acl-long.853/) 的 Judge/Critic/Refiner/Curator 分工，以及 [StructRAG（ICLR 2025）](https://proceedings.iclr.cc/paper_files/paper/2025/file/5975754c7650dfee0682e06e1fec0522-Paper-Conference.pdf) 的动态结构路由。前者[固定仓库](https://github.com/Peiying-Yu/Table-Critic/tree/032cc7eb21513eeddcbebb10ed58f799a373929e)中只有部分第三方目录附许可，不能把它扩展成整个项目的许可；后者完整方案包含 DPO 路由训练。两者仍有机制利用价值，但已明确知道是一个短限额表时，先测试单次取值/范围检查比直接引入四代理循环或训练路由更能定位当前错误。它们不是被永久排除的研究路线。

## 当前不足与优先实施路线

literal helper 只解决输入表示的一部分：它与当前问题无关，不选择焦点行；只支持简单明确格式，缺少层次表头和跨段条件；不会证明产品/版本/限制对象相同；也不检查生成答案是否忠实使用这些行。模型仍可能照着正确表格写出错误的包含关系，普通 NLI 也可能只看到了相同数值。注释还会新增 tokens，APPEND 下每轮重复注释的代价应进入 B 的完整成本，不能假定它天然更省。

**优先一：来源绑定的取值与关系核对。** 在已有完整 audit 上增加一个独立离线诊断入口：输入当前问题、真实历史、完整可见证据、初稿和 literal tuples；输出每个可检查的数值 claim 对应的原文 row/value/条件跨度、选取的对象、操作类型、结果及“不足以判定”的原因。不是只让模型说对/错。模型提出的跨度必须回指实际来源，复制值和显式计算可确定性复核；对象同义、隐含包含关系、跨版本适用性仍作为单独待验证判断，不能因为数字在表中出现就判支持。允许“表支持数值，但不支持新增范围/组合”这一中间结论。已有 MiniCheck 可作对照信号，不能用同一模型自评宣告修订有效。

在冻结初稿上量化诊断后，再研究一次选择性修订：反馈具体的错取行、缺失行或无据范围，保存初稿与修订稿，仍保留完整 QA/证据，并允许选择不修订。线上检查不接收 reference 或未来轮答案。无需立刻增加新的 critic 权重；可以先接现有本地模型，同时将调用、等待、输出及提交耗时计入总响应。该入口是 DRE/SMART/SSC 的组合适配假设，尚未实现，不能称为这些论文的完整代码复现。

**优先二：CRAFT 式问题相关行焦点。** 对当前 required 中的表格，复用已准备的 BGE 编码或 reranker，根据当前用户问题和已批准的 user-only query view 给行排序；返回原 span 与得分，只调整新增注释，不删除源段落、不改变原 top5 和完整 QA。必要脚注与显式同对象条件随焦点一起呈现。先比较所有行注释、相关行注释、相关行加显式条件闭包；在三行小表上不预设收益，长表与相似对象混淆才是该机制的主要适用场景。

**优先三：层次表头与可执行数值操作。** 在来源保有结构的材料上引入 TreeThinker 式 header path，将父级对象、版本、指标和单位随单元格带入派生表示；再借鉴 SMART，仅对明确要求计算或排序的问题使用带原单元格引用的操作。不能把“可计算”当作“有权合并这些对象”。来源只剩无分隔文本时，保留当前不猜测行为，后续通过同版本原始材料或公开元数据补结构，而非把今天网页状态回填固定旧快照。

这三条路线可以分别消融：原文、literal 注释、问题相关行、取值/范围诊断、选择性修订及其组合。报告错→对、对→错、未修复、无法判断，按对象/条件/算术/遗漏/历史来源污染分类；同时报告回答完整性、拒答、引用支持、额外 tokens、完整响应与会话时间。开发集可继续用于机制探索和对照，最终 test 保持独立。原始开关对照无论正负都保留，它用来判断下一处瓶颈，不是结束 A 模块研究的门槛。
