# 支持单元：原文位置、显式依赖与适用边界

2026-10-24，设计阶段。本文补充[来源绑定生成方案](source_bound_generation_design.md)，回答“支持单元从哪里来、闭包能保证什么”，不重复模型选择与解码综述。**完整 QA 和全部原正文继续保留时，选择较少单元不会直接减少正文 prefill。** 它可能改变证据关注、出处绑定和回答长度，也会增加计划/辅助输入；收益必须按质量与完整成本验证，不能用单元压缩率替代 KV 节省。

## 完整开发材料实际提供了什么

只读扫描了固定 MTRAG 开发集的 **22 会话/159 轮、795 个当前 top5 位置、605 个独特片段**。先按 dev conversation ID 过滤，再取当前 user 和证据白名单；未使用官方 assistant、targets、answerability、context.reference 或 qrels，未分析 heldout 正文。这不是 P3 新检索结果，也不含 APPEND 额外历史证据。

| 域 | 会话/轮 | 证据位置/独特片段 | 空行段落候选 | 已识别表格/关系 | 触发表格轮 |
|---|---:|---:|---:|---:|---:|
| ClapNQ | 6/47 | 235/175 | 175 | 0/0 | 0 |
| FiQA | 5/32 | 160/139 | 139 | 0/0 | 0 |
| Govt | 6/39 | 195/148 | 2227 | 0/0 | 0 |
| Cloud | 5/41 | 205/143 | 1451 | 5/13 | 2 |
| 合计 | **22/159** | **795/605** | **3992** | **5/13** | **2/159** |

表格计数来自现有 `table_relations` / `table_structure`，按独特片段计；按重复位置则为7张/19关系。没有新增 parser 规则。真实 parent_path、标记脚注链接、source_notes 均未激活；7条拒绝记录保留。**识别为零不等于原材料没有表格或条件**，只能说明现 parser 的覆盖。

所有795个原 context 都没有显式 version、dependency、header/condition dependency 或 api_source 字段。Cloud11个独特片段出现 def/HTTP 等线索，但原材料中**经固定 SDK 源验证的 API 单元为零**。语料 snapshot 不是产品版本；代码词面也不是签名、认证与初始化合同。

所有605个片段都能提供整段字符锚点，但这不是100%语义支持。Govt 的2227个空行候选中有1301个不足30字符，包括菜单、语言和编号；ClapNQ/FiQA则全部是单段片段。不能把空行或句号普遍解释为独立事实边界。复算脚本、全分母与精确 span 检查见[扫描记录](../runs/support-unit-design/README.md)和[summary.json](../runs/support-unit-design/summary.json)。

## 原理与已有实现的复用

[AFTG](https://aclanthology.org/2024.acl-long.182/)提供先选择片段、再联合规划/生成的分解；[ReClaim](https://aclanthology.org/2025.findings-naacl.55/)提供合法原文序列约束；CAGE进一步显式表示答案与支持关系。它们不证明任意选中片段足以支持转述。固定作者版本、许可及自动重绑来源等移植问题已列在[来源绑定方案](source_bound_generation_design.md)，此处不重新实现整套 pipeline。

优先直接复用以下已有结构，不另建抽取模型或全局图：

| 现有入口 | 可携带的确定信息 | 不能推断的内容 |
|---|---|---|
| `EvidenceBlock` / `EvidenceKey` / 引用 manifest | 原文、来源身份、语料版本、已有字符范围与稳定 ref | 产品版本、来源真实性、当前问题适用性 |
| `table_relations` / `table_structure` | 原行/对象/列/值、同表表头、同标记脚注、明确父路径 | 邻接 prose 的全部条件作用域、跨行包含/合计关系 |
| `api_evidence` 的固定 source_spec/AST sidecar | 声明原文、签名、docstring、公开符号与显式依赖 | 动态路径、继承/绑定行为、所有调用前置条件、代码实际可运行 |
| `CorpusIndex.resolve` | 精确来源/版本的显式 block 依赖闭包 | 相似来源替代、隐式语义依赖、缺失材料的自动补全 |

已有扩展 SDK 包可验证闭包接口，例如 `post_document → Document → Attachment/Document.to_dict`；它不属于上述795个原 context，不能混入覆盖比例。API sidecar 的签名 offset 相对**完整源码**，表格 span 相对 **EvidenceBlock.text**：已有 `post_document` 源39568:42735、签名39572:39775，映入 block 必须转为4:207。[独立闭包与坐标核查](../runs/support-unit-design/api-closure-example.json)

## 最小通用模块草案

单位称为“支持候选”，不称已验证事实。建议仅有构建与闭包两个纯入口，尚未新增生产代码：

```text
SpanRef = (EvidenceKey, start, end)       # 统一相对 EvidenceBlock.text
SupportCandidate = (
    local_id, kind, primary_spans, context_spans,
    required_units, required_blocks, scope_status, fallback_passage_id
)
build_support_catalog(blocks, table_mode, matching_api_sidecars) -> candidates
close_support_selection(ids, candidates, visible_keys) -> closed_ids, missing_keys, unknown_scope_ids
```

`kind`只区分整段、prose span、表格行、API声明；局部ID供当前计划枚举，不替换稳定 E_ID、不新增 hash。文本从原 blocks 按 span 读取，重复文字的不同来源保持独立身份。构建时一次检查坐标/原文对应，两候选可共用不可变 catalog；不做每 token 扫描、持续重hash或后台图存储。

闭包仅纳入可审计边：行到其表头/对象、同标记脚注、明确父路径，API到已核声明/显式spec或原注解指向的符号，以及已有 `EvidenceBlock.dependencies`。一般邻近段落、if词、产品名相似度不自动产生“必需条件”边。局部单位用有限集合遍历，block版本闭包复用 `CorpusIndex.resolve`；不要求另建DAG服务。

默认 **visible-only**：依赖不在当前允许材料里就返回 missing，不静默从别的版本/语料补入。若独立实验允许来源扩展，先通过既有语料适配把闭包纳入 required 并记录材料变化，再构建单元；不能仍称原 top5 同材料对照。

## fallback 与不能保证的语义

无可靠细边界、指代/条件跨句且无明确结构、导航正文混杂、不规则表格、截断代码/公式、版本不明或缺依赖时，默认保留整段 fallback 和 unknown 原因。它不能找回片段外的上文，也不把结构闭合升级为语义充分。

真实原文说明了必要边界，精确 source/task/rank/offset 见[九个示例](../runs/support-unit-design/examples.json)：

- Cloud 同轮来源分别写 Request Body 11MB 与 Total request 10MB，缺少可排序的适用版本，不能自动取最大值/多数票；另一个表的版本例外在表格之后，单闭合 header+row 仍会漏掉它。
- ClapNQ 的“16台这些机器”承接前句改型设备，并以 oxen 为条件；FiQA 的 YTM 陈述带持有/支付/再投资假设，公式还丢失上标。只复制数字句不能保证对象、条件或公式正确。
- Govt 的片段同时含多个法律标题/日期，或混入导航和其他 FAQ；空行、日期及 `.08` 的句号不能直接决定语义范围。
- Cloud 的 `sdcclient` Python 代码片段不能因出现 `def` 就成为 Cloudant 的版本化 SDK 声明。即便某 getter 的显式闭包只有自身，也不能推导它不需要初始化/认证。

可机械保证的是身份、坐标、字面复制和显式依赖未丢；不能保证问题对象解析、相关性、来源现行性、全部条件、联合推理、OR/AND、逐项/总量、普通转述蕴含、API可执行或回答完整性。**`引用属于选中组` 与 `原文蕴含回答` 是不同合同。**

## 下一步假设与强对照

先完成[已规划的 Qwen3 8B/32B 能力参考](source_bound_generation_design.md#验证顺序与失败标准)，避免在单一弱生成器上堆结构规则。随后在相同完整159轮材料/历史与模型上比较：A0原Q0单次回答；A1普通notes/提纲再回答；A2支持单元计划、显式闭包、回答实现与宿主引用。A1/A2采用同阶段预算（拟384+1024），A0最终cap1024；普通两段也可见相同单位表，控制多一次调用和辅助输入收益。预算在运行前冻结，保留全部失败/length/未覆盖，不以官方答案选择单位。

从A2同一计划直接渲染原文支持，另作无需第二次模型调用的 extractive 强简单对照。若它同样有效且更便宜，复杂转述没有获益证据；若出处绑定正确但范围/回答性不改善，瓶颈仍在选择或语义实现，不能报格式成功。A2优于A1也只是组合机制结果，后续固定计划/文本才能拆出引用携带与条件保持的增量。

逐项报告任务覆盖、对象/量词/条件/版本、cited与visible支持、机械出处、代码必要性、完整调用/输入/输出与时间成本。按会话聚类并分域；所有22会话仍是dev，官方top5非oracle。之后再做自有历史闭环与共同P3，不把结构扫描、合法引用或同一模型自评分当作质量提升。
