# 原子评审开发验证

本轮验证冻结的 `bench.atomic_judge`（v1 源码 commit `dc6b8170d0a8e2619b5f187f21d26d3a05410c92`），不改 `native_v1` 的五维协议，不回填既有分数，也不构成人工验收或 test 非劣结论。v1 的 29 条真实本地模型调用结果已导入：42 次 HTTP 全部正常结束，但 21 条 ledger 为 partial。既有接口类型问题，也有抽取改变范围、历史污染和正确短答误伤，不能把问题归结为 JSON 格式。方法来源与实现边界见 [原子评审提案](atomic_judge_proposal.md)，原生评审对照见 [开源 judge 验证](open_judge_validation.md)。

## 输入与冻结边界

执行输入为 `runs/atomic-development-inputs-v1/{audit,labels}.jsonl`，共 29 条：26 个 assistant 显式构造的合成世界探针，以及三个已完成的真实冻结答案。期望标签、方法臂名及生成来源仅在 sidecar，提取与 checker 的白名单输入不含这些标签。

合成输入标记 `measurement_kind=simulation`，没有伪造生成器 input IDs、output IDs 或 receipt。13 组正反对照覆盖对象范围、逐项/总量、单位换算、版本与角色条件、否定、脚注例外、联合来源、历史答案污染、引用子集、当前问题、正确拒答、错误拒答和代码结构。其中 24 条属于 prose 检查；两条 JSON 结构对照走已有静态语法诊断，不能计入 prose 语义正确率。所有期望是合成定义，标签来源不是人类评审。

三个真实输入来自 `runs/table-frozen-pairs-v2/cases.json` 和 `h200-order-pairs/measurements.jsonl`：`old_error_history`、repeat 0、`off / all_rows / focus_refs`。独立构造器 `runs/atomic-development-review/prepare_inputs.py` 已逐项核对实际 input IDs、既有 token hash、receipt 与输出，且三者的 query、actual history、可见来源及原文相同。执行侧使用的原始生成长度分别为 277、218、114 tokens，均未截断；这些长度只作为离线来源信息，不是评审标签。

导入时三条真实生成的 `unvalidated_hardware_run` 标记被标准化成 `hardware`，差异记录在 `source-marker-note.json`，未改变模型输入。分析保留这个来源限定：这些生成并未因此成为已验证的性能或质量测量。原 audit 和响应均未回填修改。

Selene 的模型为 `AtlaAI/Selene-1-Mini-Llama-3.1-8B`，revision `427792f1c3e2073cb7da216924fd884b1ba496e0`；本轮提前固定 temperature 0、单调用输出预算 4096、上下文窗口 32768，并以固定本地 tokenizer 完整计数。输入不会为了容纳模型而截掉来源。Atomic 使用自己的提取与检查 prompt；配置中的 native protocol 名称只来自复用的运输配置，不表示本轮使用五维 Result 评分模板。

## 结果分析口径

只读脚本：

```bash
python runs/atomic-development-review/analyze.py \
  --audit runs/remote-h200-atomic/runs/atomic-development-v1/audit.jsonl \
  --labels runs/remote-h200-atomic/runs/atomic-development-v1/labels.jsonl \
  --results runs/remote-h200-atomic/runs/atomic-development-v1/ledger.jsonl \
  --output runs/atomic-development-review/analysis-v1-rerun.json
```

脚本保留全部 29 条输入分母，缺失、截断、schema 错误、定位失败和 unresolved 都不删除；无结果预检已确认 29 条全部保留，24 条 prose 全为 unknown。原始响应、finish reason 和 usage 保留在实际结果文件，派生分析引用该文件及 SHA。完整派生结果为 `runs/atomic-development-review/analysis-v1-reviewed.json`；最初生成的 `analysis-v1.json` 也保留。两者语义标签统计相同，前者细分了绑定与 evidence/verdict 自相矛盾的错误，并带上来源 marker 说明。

分析分为四层，不能用下层成功替代上层正确：

1. 运输与结构：HTTP、完成标志、上下文预算、JSON、必需槽位是否成立。
2. 原文绑定：claim 的 answer quote 是否定位，checker 的 quote 是否属于本次 visible/cited 来源子集。定位只证明来源，不能证明蕴含。
3. 抽取覆盖：原文单元是否未处理、声明不可验证、需代码/表格另审，是否有无法绑定的引用。已有单元被覆盖，也不能证明所有限定词都保留。
4. 模型 ledger 与预期的对应：正例误伤、反例漏检、拒答的来源可答性，以及相同事实在全部可见来源和被引用子集上的差异。failed-slot 原字符串、联合来源数量作为定位诊断单列，不能因字段同义名称不同把正确语义判断强改为错误。

完整但判错与未能评估分别统计。引用缺失不会变成事实错误；相同 visible/cited 集合复用的检查只计一次 HTTP 调用。`matched` 仅表示当前 ledger 标签对应合成预期，仍需检查抽取是否丢失作用域或条件，不能称语义验证通过。开发例已见、规模小，也不提供总体准确率或置信保证。

## 三个真实答案的逐源预审（assistant audit）

五个可见段落中，`E_53079fa73a` 与 `E_fe06da00dd` 是正文相同的限额资料：表格分别列出 Individual Document Size、Single Attachment Size、Request Body Size，对应 1、10、11 MB；正文建议把二进制附件或大 JSON 放到对象存储，并说明超限时不复制。两段重复正文不能算独立的证据确认。

`E_8312203498` 讨论文档保持小尺寸及文档增长带来的性能问题，没有在 1 MB 后写“包含附件”；`E_bd58806373` 介绍附件和内联 BASE64，提醒附件大小、数量影响性能；另一个 Discovery 段落讨论文档切分，与 Cloudant 附件上限无关。旧 assistant 历史含错误概括，只能用于理解用户纠正的对象，不能提升当前 claim 的事实支持度。

| 实际答案 | 待验证的具体边界 | 引用边界 |
|---|---|---|
| `off` | 单附件 10 MB 有直接支持；重复出现的“文档包含附件总计 1 MB”额外扩大了表格对象范围，不能因为 1 MB 数字出现而判完整支持。keep-small 和对象存储建议有来源，其他性能概括仍需逐项核对。 | 没有原始 Evidence ID 引用；这与事实判定分开。 |
| `all_rows` | 三项限额分别列出更贴近表格，但“组合文档和附件总量”仍是关系解释而非表格原句；末尾“不影响整体性能”的保证超出了来源的提醒。不能把整篇自动视为完美正例。 | 没有原始 Evidence ID 引用。 |
| `focus_refs` | 核心答案只指单附件 10 MB，与表格直接相符，是识别正确短答误伤的重要对照。 | ID 出现在独立的 Evidence 行，引用组与后续句子的绑定可能仍 unresolved；不能把字符串中出现 ID 当成逐 claim 绑定完成。 |

以上是 assistant 对原始来源的检查记录，未填任何 human 分数，也没有把三个答案设置成模型可见的正负标签。

## v1 实测结果

29 条都有结果：8 complete、21 partial；42 次独立 HTTP 均为 `stop`，包括 29 次提取与 13 次 checker，复用的 cited/visible 检查不重复计数。合计输入 36,662、输出 5,643 tokens，实际调用输入为 508–5,171 tokens，未碰到 32,768 窗口与 4,096 输出预算的边界。没有把失败归因于上下文截断。

| 合成 prose 分组 | 冻结输入数 | ledger 标签对应预期 | 对应错误 | 未完成所需判定 |
|---|---:|---:|---:|---:|
| 正事实对照 | 12 | 3 | 0 | 9 |
| 负事实对照 | 10 | 3 | 1 | 6 |
| 正确/错误拒答 | 2 | 0 | 0 | 2 |
| 合计 | 24 | 6 | 1 | 17 |

这不是“6/7 准确”：17 个 unknown 仍在分母中，三个真实例全部 partial，且完整的单元记账不等于语义可靠。两条代码控制的静态 JSON 检查分别识别合法 JSON 和尾逗号错误；模型试图把代码单元作为 claim，协议拒绝，不能把静态检查的成功加到 prose 结果中。

各类别没有被筛掉：object_scope、version_role 两组正反例均得到预期 ledger verdict；per_item_aggregate 为一个 unknown、一个 mismatched；unit_equivalence、joint_evidence、history_is_not_evidence、citation_subset、current_question、两类 abstention 均未完成该组完整判定；negation、footnote_exception 各只有一条完成所需判定。此处的 unknown 可同时含正确的局部 verdict 与未完成的忠实性/覆盖检查，不应擅自补为正确分。

结构与定位问题：

- 23 个 claim 被 `value_raw` 的 string-or-null 限制拒绝，实际模型返回的是 7、10 等 JSON 数值；这属于表示不兼容，不是事实错误。另有一项缺少 `kind`。
- `focus_refs` 的两个 claim 从其他单元带入引用 ID，被 v1 整体拒收，导致本可单独检查的可见事实丢失。引用关联问题应独立记录，不能扩展 cited 来源集合，也不必丢掉语义 claim。
- `all_rows` 的五个 statement 原样复制了 schema 示例字符串，而非真实回答。数值类型修复不能自行证明这些抽取忠实。
- 17 条出现未记账单元；两个代码控制都尝试抽取结构单元。覆盖不足不只来自校验器，也来自模型漏抽、漏写 uncovered。

以下语义失败保留原始结果，没有用规则修正分数：

| 探针 | 可复核问题 | 与工程失败的区别 |
|---|---|---|
| `p004` 四 worker 总量 | 原回答说“四个 worker 合计最多 7”，抽取改成“一个 worker 最多 7”；checker 仍标忠实且 insufficient，忽略来源明确的合计 28。 | complete 也会错；范围保留与忠实性判断均失败。 |
| `p010`、`p020` | 原始断言分别保留了错误否定关系/错误单位，checker 给出正确的 contradicted，却把“与来源矛盾”当作 extraction_faithful=false。 | 抽取忠实与事实真实被混淆，协议保持 unresolved，不能自动改回 true。 |
| `p015` 历史污染 | 当前正确答案及来源均为 80；checker 理由转而引用历史 assistant 的 400，并给 insufficient/不忠实。 | 即使 prompt 说历史非证据，模型仍会混淆对象与消息角色。 |
| `p021` 正确拒答 | 被抽取成 assertion，返回 insufficient，没有来源可答性判定。 | complete 不代表拒答维度已验证。 |
| `p024` 错误拒答 | 已识别来源 14 天与拒答矛盾，但 source_answers_current_question 仍为 null。 | 必需的拒答合同未完成，不能靠局部 verdict 补分。 |
| `real-003` 正确短答 | c3 对“单附件 10 MB”返回 contradicted，同时引用“Single Attachment Size 10 MB”、关系 supports，理由也复述同一值。 | verdict 与证据关系自相矛盾；宿主保留失败，没有用 quote 命中把它改为 entailed。 |

## v2 最小修正与待验证边界

在完整保存 v1 后，新的独立协议升为 `contextkv-atomic-evidence-v2`：

- `value_raw` 接受字符串、有限 JSON 数值或 null，原样传递与保留，不强制转成字符串；boolean、非有限数与容器仍在外部边界拒绝。
- 模型声明的额外引用不再丢弃语义 claim。原声明、原单元 ID 与额外 ID 分开记录；checker 的 citation_ids 和 cited 来源集合始终由宿主从原单元解析。额外 ID 不产生支持证据，仍让整体 ledger 保持 partial。
- 提取和检查模板改成字段说明，去掉容易被复读的示例答案值；保留条件、scope、模态、精确原文定位、抽取忠实性与证据支持要求。没有依据本轮错误修改已有分数或自动修答案。

本地 atomic/native 75 项针对测试通过，覆盖数值原样保留、非法结构、借入邻单元 ID、可见来源与引用子集分离、quote 精确性及旧有泄漏边界。v2 仍需对同一 29 条完整重跑，并分别报告新增可评估项和新增语义错误；格式可用性改善不能预先声称已解决历史污染、范围错改或正确例误伤。
