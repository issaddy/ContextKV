# 可选原文结构扩展：全 41 轮覆盖与单轮 GPU 对照

2026-10-24。本轮实现 `table_structure_mode: explicit`：在既有来源跨度、CRAFT 行文本与词法 focus 上增加无编号表和明确父路径。默认仍是 `legacy`。全41轮CPU扫描后，已对唯一新增轮做6次冻结GPU生成：表内数值使用改善，但没有解决当前产品/问题的适用性，也没有补出引用，不能称完整回答修复或总体质量提升。此实现借鉴[既有研究记录](table_relations.md)中层次表头的表示思想，不复制 TreeThinker 的模型生成树，也不从丢失结构的正文重建 colspan。

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

下面的真实对照已执行上述唯一新增轮，父条件效果仍需要确有原始层次的来源场景；当前41轮不能证实它。本轮不改默认，不声称原子judge或结构提取等于事实验证。

## 单轮冻结历史的6次真实GPU生成

完整原件保存在 `runs/remote-h200-table-structure/runs/table-structure-frozen-v1/`，执行 run ID 为 `ae9b6b33ed094966a510e17a0b36a320`。`cases.json` 与运行前本地冻结文件逐字相同，SHA `82e69551f2f9472b7b303ffaf1106a7b9a37d36686922a2c1844ce1861db6825`；原结果未改写。只读复核脚本及派生摘要为 `runs/table-structure-gpu-review/{analyze.py,summary.json}`。

范围是已见开发任务 `d5b1e735a040853ed361a3dfde1b8ef0<::>4`，**同一个历史与问题，不是6个独立样本**。Q0、完整6条历史消息、原query末尾空格和全部5段来源一致。温度0、上限1024、正常EOS；未检索、未调用judge、未给模型当轮原答案或评审判据。顺序为：

1. off → legacy_focus → explicit_focus。
2. legacy_focus → explicit_focus → off。

这两个循环改变了位置，但不是三位置完全平衡设计。所复用的是既有冻结对照执行器，不经过完整P3检索/控制器闭环。

独立复核确认：

- 三臂完整消息重新分词后与冻结IDs相同；off与legacy_focus的全部消息、IDs、hash完全相同，并匹配旧off输入receipt。每个臂的原历史与原证据字符跨度保持不变。
- 6/6成功，全部 `stop` 且receipt完整，无1024截断；每条实际receipt的input hash、attempt、engine session、generation和restart身份都与对应prepare/retire一致，实际轮换顺序符合计划。
- 每次cold reset后GPU/host占用和active/queued均为0；实际device hit和host load均0，prefill分别完整覆盖4737或4992输入tokens。池容量为GPU121920/host279040 tokens、page64，本实验不是有限容量下的恢复收益对照。
- 6次retire均ack，但物理free计数均0；后续cold reset清掉残留缓存。不能把retire确认描述为本轮释放了全部KV。
- 四个off/legacy输出的文字和IDs完全相同，而且文字与旧off答案一致；两个explicit输出也完全相同。**6/6都没有Evidence ID引用**，内联引用注释在本例未转化为答案引用。

| 臂，各2次 | 输入 tokens | 输出 tokens（两次相同） | client首token均值 | guarded generate完成均值 | client TPOT均值 |
|---|---:|---:|---:|---:|---:|
| off | 4737 | 412 | 140.03 ms | 2056.88 ms | 4.655 ms |
| legacy_focus | 4737 | 412 | 114.30 ms | 2031.38 ms | 4.655 ms |
| explicit_focus | 4992 | 140 | 117.93 ms | 770.97 ms | 4.669 ms |

client TPOT按实际首/末chunk与输出token数计算；server TPOT另存原结果。冷重置/prepare耗时分别保留，未混入表中guarded generate。首个off的client TTFT为165.46ms，第二次为114.61ms，说明小样本/位置仍影响时延；没有为比较删掉第一条。explicit输入多255 tokens，但答案少272 tokens，完成更快主要伴随生成长度变化。**答案内容和完整性未等价，不能报同质量加速或P3总体收益**。

### 按原来源核对的有限改善与剩余错误

以下是assistant逐源审查，没有自动填入人评分，也不形成正式五维质量gate。

先明确产品范围：三条先前用户问题从“IBM是否有文档数据库”推进到“任意JSON”及“image/PDF与document一起存储”，用户从未直接说Cloudant；历史assistant将话题具体化到Cloudant。第一轮真实来源支持Cloudant是JSON文档数据库，但不能据此把后续assistant的错误数值或代码当事实。当前五段材料是Akamai CDN、两段Discovery文档/字段限制，以及两段File Storage卷/inode说明，**没有Cloudant单附件上限的直接依据**。本6次保持这些材料，不额外补源。

| 原来源/对象 | off与legacy_focus的四次相同回答 | explicit_focus的两次相同回答 |
|---|---|---|
| `E_8fac10f0de`，Discovery **上传文档**的套餐表 | 把Discovery计划冠名Cloudant；Cloud Pak for Data50MB正确，但Premium错写10MB（原表50MB），Enterprise10MB正确，未列Plus | 四行原表值均正确：Cloud Pak for Data/Premium50MB，Enterprise/Plus10MB；没有显式标明Discovery或“上传”适用范围 |
| `E_c07431a963`，**索引field**上限 | 未列这两行 | 正确保留HTML field5MB、Sum of all other fields1MB。没有观察到把1MB直接称Cloudant附件上限；这项预先担忧未在本输出发生 |
| 两段File Storage/inode说明 | 将7.8TB达到最大inode数的**卷容量门槛**改称单个文件大小上限，错误 | 未再输出该错误；这是删除无关/错误细节，不证明已答对当前附件问题 |
| CDN默认/优化条件 | 未使用 | 未使用；没有凭其不同产品的数值补Cloudant限制 |
| 当前对话的产品与所问对象 | 明确把Discovery的上传/爬取限额当Cloudant，错误 | 开头改说“documents in IBM Cloud Pak for Data”，却未说明为何适用于前文Cloudant的image/PDF附件；没有承认当前证据不足以确定Cloudant附件限制 |
| 引用及额外操作建议 | 无Evidence ID引用 | 无Evidence ID引用；结尾建议外部保存文件并在document引用，重复了历史建议，但当前5sources没有提供该产品操作的支持 |

explicit的六个表值和行标签匹配原文，Premium错误得到局部纠正，原7.8TB错误未再出现，不能把这些变化抹成“毫无改善”。与此同时，它把回答焦点转为Cloud Pak for Data，并未完成当前对话中的Cloudant附件问题，也没有给出源ID或证据不足说明。**Discovery数字更准确，不等于Cloudant答案正确**。

这次结果支持继续研究“结构表示帮助源内取值”的机制，但表明只加行关系不能解决产品适用性和检索主题断裂。下一项独立分支可检查来源约束的对话主题解析：历史assistant只辅助指代/意图，产品实体必须回溯当时实际证据，事实依据仍来自当前明确来源；简单放宽用户历史的命名实体regex无法补出从未由用户写出的Cloudant。本次不同时改检索或提示，不把这6次开发重复当独立质量验证。
