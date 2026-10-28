# 原文表格行关系注释

`adapters/table_relations.py` 实现可选、无模型的表格辅助表示。它保留证据正文、版本和稳定引用，另输出对象、原列名、原值、显式单位及字符位置，用于帮助模型区分同一表格中不同对象的属性与条件。

## 复用依据和适配边界

[TabRAG 原论文 §2.2–2.3](https://arxiv.org/html/2511.06582v1)先把单元格与行、列关联，再由语言模型产生供检索使用的描述。其[作者实现 `VLMPrompts.vlm_table_prompt`](https://github.com/jacobyhsi/TabRAG/blob/c3aba4190f141737f1a52ed1bacce95bec337667/src/prompts.py)明确使用 row/column/value/units，固定在 `c3aba4190f141737f1a52ed1bacce95bec337667`。本适配借鉴这些字段之间的结构关系，独立实现确定性文本解析，在阅读输入中附加行关系。上游[仓库许可证](https://github.com/jacobyhsi/TabRAG/blob/c3aba4190f141737f1a52ed1bacce95bec337667/LICENSE)为 CC BY-NC 4.0；论文页面另标 CC BY-NC-SA 4.0。

真实开发反例的正文已经是扁平文本，不能只实现 Markdown 解析：`ibmcld_00558-23465-25360` 和 `ibmcld_12904-23555-25450` 具有编号表题、`Limit Maximum Size` 表头和三行不同对象的数量。实现没有写入这些产品名、对象名或具体数值；测试保留原表摘录用于回归，并使用不同对象、值和单位验证通用行为。

## 接口与来源位置

```python
from adapters.table_relations import extract_table_relations, render_table_relations

tables = extract_table_relations(evidence.text)
supplement = render_table_relations(existing_reference_id, tables)
```

两者均为纯函数，仅从传入正文提取关系；无表时分别返回空元组、空字符串。调用者决定是否在当前输入附加 `supplement`，并对新增 tokens 使用现有候选预算和审计。补充表示沿用原证据身份，`EvidenceBlock.text` 保持原样。

共享候选入口已通过 `context.table_relations: true` 接通；原生 CLI 使用 `--table-relations`，matrix 两条执行路径继承共同的 base config。默认关闭。每轮对 required 证据提取一次并供两个候选共用，APPEND 即使复用已有正文，也会在当前消息附加该轮注释；新增长度仍受原共享预算约束。专项集成验证了默认逐字兼容、QA/正文跨度不变、Native 输入一致和哈希调用次数不增加。已有局部验证，尚无默认开启的整体收益结论。

新增的 `context.table_annotation_style` 默认为 `all_rows`，保留原全行输出；设为 `focus_refs` 才使用问题相关行和原引用样式，而且仍须 `table_relations: true`。该轻量样式直接复用 CRAFT 的行文本构造函数，词法选择复用现有 BM25S 与纠正线索，所需依赖已属于 `contextkv[retrieval]`；默认关闭和全行样式不加载此可选依赖。完整 MIT 归属见[CRAFT notice](../third_party/CRAFT_NOTICE.md)。

可选 `context.table_structure_mode: explicit` 已补无编号明确数量表头和有实际 Markdown 标题/缩进的父路径；默认 `legacy` 保留本节原两种格式。它没有推断真实扁平列表中丢失的缩进。源码与合同见 `adapters/table_structure.py` 和相应测试。

`SourceSpan(text, start, end)` 使用传入原段落的 Python Unicode 字符偏移，左闭右开，保证 `original[start:end] == text`，不是 UTF-8 字节偏移。`TableRelations` 保存表题、表头、整表跨度、关系与原注释；`TableRelation` 保存 `object/column/value/unit/row_span/notes/parent_path`，默认父路径为空。全部是不可变 dataclass，可用 `dataclasses.asdict` 留存审计。正文内空白、Markdown 转义、大小写和数值写法不作归一化；仅排除单元格两端的格式空白。

支持两种明确结构：

- `markdown`：单行表头、标准横线分隔行、列数一致、首列行名非空。每个值单独关联其原行名和列名。缺失行名、多级/合并/不等列数的表不作猜测。
- `captioned_numeric_rows`：编号 `Table` 表题、单行非数值表头、至少两行连续的“对象＋末尾数量和显式单位”。末尾单位只识别明确的常见大小、时间、速率、频率、质量、长度、百分比和计数写法。不能恢复的列边界保留为完整 `flattened_header`，绝不把 `Limit Maximum Size` 强拆成两个猜测列。未知单位或混合不规则行回退为空注释，原文仍然存在。

数值和单位不换算，也不相加。Markdown 单元格有唯一显式数量时保留其单位；没有数量单位时，仅从列名明确的 `(unit)` 或 `[unit]` 后缀取单位。单元格中存在多个数量时不推断一个统一单位。源表本身若有明确的合计对象行，会照录该行，不采用“一律禁止合计”的规则。

相邻、显式标记的 `Note:`、`Remark:`、中文注释或脚注只作为 `source note` 原文保留。只有行中与注释开头存在完全相同的 `†`、`‡`、单星或 `[^id]` 标记时，才建立该行的 `notes` 关联。没有标记的注释不变成所有行的共同限制条件。遇到普通叙述段落就停止搜集，不跨越章节追找脚注，因此真实例中远处 Frankfurt 的地域脚注不会被接到大小限制上。

相应 CPU 合同覆盖于 `tests/test_table_*.py`；已有局部生成检查，整体质量与成本收益尚待验证。
