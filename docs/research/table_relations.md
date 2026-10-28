# 原文表格行关系注释

`adapters/table_relations.py` 实现可选、无模型的表格辅助表示。它保留证据正文、版本和稳定引用，另输出对象、原列名、原值、显式单位及字符位置，用于帮助模型区分同一表格中不同对象的属性与条件。

## 复用依据和适配边界

[TabRAG 原论文 §2.2–2.3](https://arxiv.org/html/2511.06582v1)先把单元格与行、列关联，再由语言模型产生供检索使用的描述。其[作者实现 `VLMPrompts.vlm_table_prompt`](https://github.com/jacobyhsi/TabRAG/blob/c3aba4190f141737f1a52ed1bacce95bec337667/src/prompts.py)明确使用 row/column/value/units，固定在 `c3aba4190f141737f1a52ed1bacce95bec337667`。本适配借鉴这些字段之间的结构关系，独立实现确定性文本解析，在阅读输入中附加行关系。上游[仓库许可证](https://github.com/jacobyhsi/TabRAG/blob/c3aba4190f141737f1a52ed1bacce95bec337667/LICENSE)为 CC BY-NC 4.0；论文页面另标 CC BY-NC-SA 4.0。

解析器同时支持Markdown表格与带编号表题的扁平文本。开发样本`ibmcld_00558-23465-25360`和`ibmcld_12904-23555-25450`包含`Limit Maximum Size`表头及三行对象数量；测试同时覆盖这些原表摘录和其他对象、数值与单位。

## 接口与来源位置

```python
from adapters.table_relations import extract_table_relations, render_table_relations

tables = extract_table_relations(evidence.text)
supplement = render_table_relations(existing_reference_id, tables)
```

两者均为纯函数，仅从传入正文提取关系；无表时分别返回空元组、空字符串。调用者决定是否在当前输入附加 `supplement`，并对新增 tokens 使用现有候选预算和审计。补充表示沿用原证据身份，`EvidenceBlock.text` 保持原样。

通过`context.table_relations: true`启用，默认关闭；原生CLI使用`--table-relations`，matrix两条执行路径继承共同的base config。每轮对required证据提取一次，供APPEND/REBUILD共用。APPEND在当前消息附加本轮注释，新增长度计入共享预算。集成检查覆盖默认行为、QA/正文跨度、Native输入一致性及身份计算复用。

新增的 `context.table_annotation_style` 默认为 `all_rows`，保留原全行输出；设为 `focus_refs` 才使用问题相关行和原引用样式，而且仍须 `table_relations: true`。该轻量样式直接复用 CRAFT 的行文本构造函数，词法选择复用现有 BM25S 与纠正线索，所需依赖已属于 `contextkv[retrieval]`；默认关闭和全行样式不加载此可选依赖。完整 MIT 归属见[CRAFT notice](../third_party/CRAFT_NOTICE.md)。

`context.table_structure_mode: explicit`支持无编号数量表头，并从原文Markdown标题和缩进提取父路径；默认`legacy`支持下述两种格式。实现见`adapters/table_structure.py`。

`SourceSpan(text, start, end)`使用原段落的Python Unicode字符偏移，左闭右开，满足`original[start:end] == text`。`TableRelations`保存表题、表头、整表跨度、关系与原注释；`TableRelation`保存`object/column/value/unit/row_span/notes/parent_path`，默认父路径为空。二者为不可变dataclass，可通过`dataclasses.asdict`序列化。正文格式保持原样，单元格提取时去掉两端格式空白。

支持两种明确结构：

- `markdown`：单行表头、标准横线分隔行、列数一致、首列行名非空。每个值单独关联其原行名和列名。缺失行名、多级/合并/不等列数的表不作猜测。
- `captioned_numeric_rows`：编号`Table`表题、单行非数值表头、至少两行连续的“对象＋末尾数量和显式单位”。支持常见大小、时间、速率、频率、质量、长度、百分比和计数单位。列边界不明确时保留完整`flattened_header`；未知单位或不规则行返回空注释，原文照常提供。

数值与单位按原文保留。Markdown单元格有唯一显式数量时取其单位；缺少数量单位时，从列名的`(unit)`或`[unit]`后缀提取。多个数量的单元格保持原文，源表中的合计行照录。

相邻的`Note:`、`Remark:`、中文注释或脚注以`source note`保留。行与注释开头具有相同的`†`、`‡`、单星或`[^id]`标记时，建立行级`notes`关联；其余注释保留在表级。遇到普通叙述段落时停止搜集。

相应 CPU 合同覆盖于 `tests/test_table_*.py`；已有局部生成检查，整体质量与成本收益尚待验证。
