# 可选原文结构扩展：CPU 实现与全 41 轮覆盖

2026-10-24。本轮实现 `table_structure_mode: explicit`：在既有来源跨度、CRAFT 行文本与词法 focus 上增加无编号表和明确父路径。默认仍是 `legacy`；未运行新生成或 judge，没有质量改善结论。此实现借鉴[既有研究记录](table_relations.md)中层次表头的表示思想，不复制 TreeThinker 的模型生成树，也不从丢失结构的正文重建 colspan。

## 接线与解析边界

```yaml
context:
  table_relations: true
  table_annotation_style: focus_refs
  table_structure_mode: explicit
```

原生 CLI 对应 `--table-relations --table-annotation-style focus_refs --table-structure-mode explicit`；matrix 的 Native/controller 共用完整 context 配置。直接调用为 `extract_table_relations(text, mode="explicit")`。默认调用和 `mode="legacy"` 的原注释、消息、tokens 保持相同。完整 required 正文、原 QA、稳定引用不变；只增加当前轮的可选注释，沿用现有 token 预算，不新增内容 hash、模型依赖或在线检索。

`adapters/table_structure.py` 支持如下窄结构：

- **`unnumbered_numeric_rows`**：单独的短标题、单独的数量属性表头（如 size、duration、rate）、连续至少两条“对象＋末尾单一数值和已知单位”。表头不拆列，标题和整表均保留原字符位置。混合/不规则行、普通句子作表头、未知单位、数值多列、代码块不被修补成表。
- **父标题路径**：仅取原文 ATX Markdown `#`–`######` 祖先，同级或上级新标题关闭旧路径。普通相邻段落不自动成为条件；路径保留标题原文，不把标题解释成新的逻辑结论。
- **`nested_numeric_list`**：实际带缩进的列表，父项以冒号结尾，子项有明确 `对象: 数量单位`，按原缩进建立父路径。没有缩进的同级 bullet 不被猜成嵌套；无名值、未知单位和没有明确父项的数字列表保持原样。每项仍带原 Evidence ID 与各段 `SourceSpan`。

`TableRelation.parent_path` 是原文跨度元组。它进入 CRAFT 的派生行文本和辅助注释，不覆盖正文；非结构文字继续只在原文中出现。既有脚注仍仅按相同显式行标记绑定。父路径说明来源层次，不证明跨行包含、合计或性能保证。

## 冻结全量开发扫描

复用原扫描的 **off 全部 41 轮**，没有按新增材料或成功例抽样：

`runs/remote-h200-table/runs/p3-table-relations-v1/off/audit-9f471d3cf04445da932b579b57902f6d.jsonl`

既有 audit SHA 为 `8a59a351b14f504dacb185771b1e57d310cf6a0826eb16ba753ca987bfb07639`。脚本核对固定 dev 会话集合，使用本地已缓存的固定 Qwen tokenizer 重建所有 41 个原 `input_ids`，均完全相同；变体保持同一完整历史与所有正文跨度。派生结果和可重跑脚本在：

- `runs/table-structure-dev-review/recompute.py`
- `runs/table-structure-dev-review/summary.json`
- `runs/table-structure-dev-review/turns.jsonl`：保留每一轮的两个模式、选择轨迹、原文结构和拒绝候选。

| 同一 41 轮材料，均使用 focus_refs | legacy | explicit |
|---|---:|---:|
| 有可解析表的轮次 | 1 | 2 |
| 表出现次数 | 2 | 4 |
| 原文行关系数 | 6 | 12 |
| 带实际父路径的关系 | 0 | 0 |
| 相对无注释的额外 tokens 合计 | 110 | 365 |
| 单轮最多额外 tokens | 110 | 255 |
| 各轮解析＋focus CPU 中位数的平均 | 0.327 ms | 0.670 ms |
| 最慢一轮的解析＋focus CPU 中位数 | 0.618 ms | 1.229 ms |

CPU 是本机 Mac 上每轮 7 次的局部观测，排除首次可选库导入和整个 prompt 分词，不能作为服务器响应开销。额外 tokens 由同一真实 tokenizer 完整渲染计算。

仅 `d5b1e735a040853ed361a3dfde1b8ef0<::>4` 的注释改变，其余 40 轮注释一致。新增来源为：

1. `ibmcld_06968-16615-18789 / E_8fac10f0de` 的 **Maximum document size**。保留 `Plan File size per document` 整表头和四行套餐/上传文档大小，未给它们附加附近 crawled documents 的套餐条件。
2. `ibmcld_06968-18330-20453 / E_c07431a963` 的 **Maximum field sizes**。保留 `Field type Maximum allowed size per document` 整表头及 html field / Sum of all other fields 两个不同对象。后方普通段落中的版本条件没有被移植为表中每一行的条件。

该轮问题只是 “Is there a limit on the file size?”，现有词法 focus 对两张表内的行分别打平，因此保留全部六行。**这不证明六行都必要**，也可能增加 field/file 混用风险；下一次同材料生成对照需要检查这一点，不能把解析数量增加当答案改进。

## 保留下来的真实拒绝与缺口

同一 passage 的 “Premium plans only:” / Box 等列表已经在抓取时丢失缩进。六条带数值的 bullet 都是同级文本，无法从实际结构确认哪些项属于该父级；当前明确记录 `numeric_list_without_explicit_indented_parent`，没有推断 Premium 的作用范围。另一段开头只有孤立 `Plus (includes Trial) 10 MB` 行，记录 `fewer_than_two_contiguous_numeric_rows`，不凭另一个 passage 自动续表。

这些拒绝是候选解析记录，**不是完整表格召回率标注**。本 cohort 的真实父路径激活仍为零，通用缩进/标题测试只能确认实现合同。后续若要覆盖丢缩进列表，应取得同版本结构化来源或固定的来源元数据，另做来源一致性验证；不能用今天网页回填旧快照，也不能只凭附近 “Premium” 字样制造祖先。

## 验证和下一步

`tests/test_table_structure.py` 覆盖非产品特定的单位/对象/数值、Unicode/CRLF 精确跨度、标题关闭、真正嵌套与同级列表、失去缩进、混合列表表格、未知单位、代码、原文/历史保持和策略入口。独立审查还发现父列表内的缩进代码可能被误识为子列表；已按父内容列的四空格代码边界修正，并加入原反例。原表格、focus、Native/controller 输入一致、matrix 继承和默认字节测试继续通过：本轮定向共 **95 tests＋6 subtests**；Ruff 通过。

下一步可由统一 GPU 调度对新增轮冻结相同历史/证据/Q0，比较 legacy focus 和 explicit focus，记录六行被如何使用、对象/套餐/范围/引用错误、额外 tokens 与响应成本。父条件效果需要另一个确有原始层次的来源场景；当前 41 轮不能证实它。本轮不改默认，不声称原子 judge 或结构提取等于事实验证。
