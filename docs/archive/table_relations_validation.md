# 表格行关系辅助表示：真实 P3 开关对照

> 归档记录：保留当时的设计、结果与复用依据；其中“下一步”和执行计划不代表当前任务。当前工作仅以[实现清单](../implementation_plan.md)为准。

2026-10-24 的这组新运行验证了功能接线和完整来源保留，尚未证明质量或时延收益。旧运行曾出现“附件也包含在文档 1MB 总额内”的错误；本次 fresh off 与 fresh on 都没有重现该错误，不能将它写成 on 修复成功。

原始产物为 `runs/remote-h200-table/runs/p3-table-relations-v1/`，off run `9f471d3cf04445da932b579b57902f6d`，on run `9baa1eb4c90947e1ae628222a7201b2a`。独立派生脚本和逐轮结果为 `runs/table-review/recompute.py`、`summary.json`、`turns.jsonl`，两个 `*-code-audit.jsonl` 只检查语法，未执行命令或调用 judge。

```bash
.venv/bin/python runs/table-review/recompute.py
```

脚本使用本地固定 tokenizer 重建所有实际输入并逐 token 比较；同时核对回答、receipt 的输入身份和原文 span。它不读取 reference answers，不生成质量分。

## 可比条件与真实触发

两组均为相同 Cloud dev 的 5 会话、41 轮，P3、normal_rag、Q0、c4、1024 输出预算；均 41/41 成功、全部 REBUILD。配置差异只有 `context.table_relations` 和输出 state_dir。模型/tokenizer/template 指纹、controller source、dataset/split、workload/runtime 一致；retrieval manifest 唯一差异是初始化用时。两组相同 correction_aware + BGE hybrid/reranker，实际 top5 文本/分数/顺序、可见 evidence manifest 与完整 audit bodies 均 **41/41 相同**。cold 记录确认各次运行前 GPU/host 使用量均为零。执行顺序为 off 后 on，各一次。

本地依据原文、历史、引用、共享模板重建 **82/82** 实际 input IDs，均完全相同。on 的注释仅出现在 `d5b1e735a040853ed361a3dfde1b8ef0<::>5`，包含两份限额表各三行；off 全部不含注释。两份来源的数值行、对象和位置均可回查，未附加远处的地域脚注。其他 40 轮没有表格辅助表示。

## 附件目标轮的来源核查

本节是 assistant 对保存答案及完整可见来源的逐条核查，不是 human blind review。

- 两版都回答 single attachment 最大 10MB；on 也说明超过该大小可以使用 object storage 并在文档保存链接。当前问题只是纠正“问附件”，on 未重复复制细节和其他操作步骤，不据此认定完整性变差。
- off 为 365 输出 tokens，on 为 193；实际输入分别 4865 和 5173。这个输入差额混合了先前历史差异和新增注释，**不能作为注释自身的 token 代价**。
- 两版都混入 `E_8312203498`（`ibmcld_00510-5537-7566`）来支持 10MB 附件限制，而该段只包含文档 1MB、建模及附件使用建议，未写附件 10MB。off 同时引用了正确限额表 `E_fe06da00dd`；on 同时引用两份正确表 `E_53079fa73a` / `E_fe06da00dd`。因此数值有可见支持，但仍有额外来源错绑，不能称引用完全修复。
- 该轮之前两组 Cloudant 会话的 turn2、turn3、turn4 回答已不同，目标输入的历史并未冻结。fresh off 已经没有旧反例，不能将 on 的更短回答当成旧错误的成对修复证据。

## 全部变化与未触发轮回归

| 对齐项 | 相同轮数 / 41 |
|---|---:|
| 当前问题、检索材料、可见正文与引用映射 | 41 |
| 完整 input IDs、实际历史 | 11 |
| output IDs、答案原文 | 6 |

11 个输入相同的轮中仍有 5 轮输出不同，分别为 SQL turn1、电话/语音 turn2、CocoaPods turn1、VA turn5、Cloudant turn2。它们没有表格注释；在 c4 服务端运行中，不能假定 greedy 就保证逐 token 重现。这也说明其他会话的变化不能归因于注释文本。40 个未触发轮中，有 29 个 input IDs、34 个 output IDs 不同；自身生成历史继续传播。

保留的具体问题包括：

- **Cloudant turn3，早于注释轮：** off 的完整 JSON 虽能解析，却把附件放入 `image`，没有按可见正文的 `_attachments` 结构表示；on 使用 `_attachments`，但 JSON 中插入 `//` 注释，且把 PNG 字节示例标成 PDF。后者的 JSON 语法失败可确定性复核。没有运行 API；这既不是完整代码可用性验证，也不能算表格注释造成的回归。
- **Cloudant turn9，注释后的历史分支：** 两版都把可见 Discovery 链接称作 Cloudant Python SDK 文档。当前五段来源包括 Discovery、toolchain 和 SDK 通用说明，不能为这些 Cloudant 专用代码/链接提供相应依据。on 新增更多说明仍未解决产品适用性；保留为来源范围问题，不能只因答案有链接就判正确。
- **非表格会话的语法问题：** on 的 SQL turn8 有两个完整 curl JSON payload 无法解析，电话/语音 turn4 有五个标成 JSON 的对象成员片段；片段不等于五个完整请求损坏。两者均未触发表格注释，不能归为注释文本的直接损害。

静态检查合计 off 0、on 8 个不可直接解析的 JSON 片段/payload，来自上述三个任务（含 Cloudant turn3）。其中 5 个是对象成员片段，1 个含 JSON 注释，2 个是完整 curl 数据候选。这是语法与呈现诊断，不是 8 个总体语义错误，也未覆盖 Python/API 调用正确性。完整逐轮材料已保留，未只挑目标轮展示。

## 时间和长度：描述性结果

| 指标 | off | on |
|---|---:|---:|
| pipeline arrival → commit 均值 s | 3.764 | 3.590 |
| 等会话均值 s | 3.796 | 3.618 |
| 完整响应 P95 s | 5.760 | 5.075 |
| pipeline TTFT 均值 / P95 ms | 904.5 / 2174.5 | 865.4 / 2040.5 |
| 检索等待及执行均值 ms | 741.0 | 706.3 |
| 输入 tokens 总量 | 214661 | 210742 |
| 输出 tokens 总量 | 21325 | 20237 |
| 截断 / 41 | 2 | 1 |

完整响应使用与 `bench.report` 一致的 `t_commit - pipeline t_arrive`；检索排队没有扣除。新增注释却出现总输入更短，来自不同生成历史，不能解释为辅助表示压缩了输入。只有一次 off→on，输出长度、调度与历史均变化，不以这张表宣布吞吐或质量非劣。

下面的冻结历史单轮对照将旧错误样本和 fresh off 目标各自固定，原 off 输入 IDs 完整保留，仅在最后 Question 前加入来源注释，交替执行 off/on 与 on/off。它检验当前干预的局部效应；结果与上述闭环结果分开记录，并继续保留原始错误及未触发轮问题。

## 冻结历史的 8 次冷隔离运行

原始结果位于 `runs/table-frozen-pairs/h200-order-pairs/`，输入为同目录上一级的 `cases.json`；`review.py` 重新核验后生成 `review.json`。在**旧错误历史这个固定 fixture** 上，on 两次都去除了“文档连附件合计不能超过 1MB”的错绑，而 off 两次都重现它。这是一个可重复的局部改进；同时，on 并未提供引用，部分额外解释仍不能由当前来源充分核验，不能把该结果扩展成完整答案质量通过。

```bash
.venv/bin/python runs/table-frozen-pairs/review.py
```

审核确认实际输入包 SHA 和脚本 SHA 与运行 manifest 一致，模型/模板指纹一致；每个 fixture 的两臂保持前 9 条消息完全相同，分别有 1327 / 1275 个相同历史 tokens。源段落、跨度、引用和当前问题都保留，唯一文本改动是在当前 Question 前插入 helper 输出，分别为 4917→5334 与 4865→5282 input tokens，**净增均为 417 tokens**。没有重放之前的生成，更没有把原目标轮答案送回模型。

两 fixture 各执行 off→on、on→off，共 8 次，全部成功并以 `stop` 完成，没有输出截断。各臂执行前 GPU/host 缓存为空且无活动/排队请求；容量为 121920 / 279040 tokens。全部 receipt 的 GPU 命中、host 加载为零，prefill 与完整输入长度一致；epoch 均确认退役。每个 fixture/arm 的两次 **output IDs 和答案文本完全一致**。因此此次输出差异不再由此前回答分叉造成，但只验证了两个已选开发历史与当前 c1 冷隔离设置。

| 固定历史 | arm | 两次输出 tokens | guarded generate 墙钟均值 ms | 逐条来源核查 |
|---|---|---:|---:|---|
| old_error_history | off | 277 / 277 | 1433.6 | 两次仍将附件算进文档 1MB；单附件 10MB 本身有据；没有 Evidence ID |
| old_error_history | on | 218 / 218 | 1141.6 | 两次区分单文档 1MB、单附件 10MB、请求体 11MB，去除已知 1MB 组合错绑；没有 Evidence ID |
| fresh_off_history | off | 366 / 366 | 1816.3 | 两次原本就答对附件 10MB；正确表引用和额外错绑同时存在 |
| fresh_off_history | on | 365 / 365 | 1825.8 | 两次保持附件 10MB，未重新引入 1MB 组合错误；没有 Evidence ID；新增示例有 MIME/实际字节不一致 |

上述为对全部 8 份输出的 assistant 来源审查，未使用人工评分或模型 judge。具体剩余问题不能因答案更短或去掉引用就忽略：

- **old/on 的范围恢复是局部的。** 表中三个对象和数值确实分别存在，已知 1MB 错绑消失。新增的 `document + attachments` 表述不是当前表格直接给出的原始字节求和规则，尚未核验所有编码/上传方式的适用性；不能仅凭 11MB 行就宣称这层扩展解释已证明。末句“这些限制确保不影响整体性能”也超出来源，来源明确提示附件可能影响性能。
- **fresh/off 的额外错引仍在。** `E_fe06da00dd` 的原表支持 10MB，但并列的 `E_8312203498` 不给出该附件上限；它还被用于复制限制等不在本段的内容。其他可见段落有支持不等于这些引用逐条正确。
- **fresh/on 没有修复引用映射。** 它删除了所有 Evidence ID，包括正确和错误的 ID，仍不能满足来源可核查要求。新增 JSON 可解析，但 `.jpg` / `image/jpeg` 配上以 PNG 签名 `89 50 4e 47 0d 0a 1a 0a` 开始的 Base64 数据；两次输出同一示例，属于一个重复出现的确定性内容不一致。没有执行 API，JSON 语法正确也不代表整段操作可用。

时间只覆盖 guarded generate，冷置与 epoch prepare 在原始行中另记，不包含真实 P3 的检索、排队及控制器开销。旧错误历史 on 少输出 59 tokens；fresh on 几乎等长且均值略慢。不能把一组更快解释成新增注释降低了 prefill，实际两组都多预填了 417 tokens，也不能据两个开发 fixture 形成总体质量非劣或吞吐结论。

这个结果支持继续发展 A：原文字段关系可以改善特定历史下的对象混淆，同时暴露来源标注、范围解释和示例一致性的下一层缺口。接续以保存的初稿和完整来源检验行/值/条件绑定、引用核对和选择性修订，保留不修订与普通 NLI 对照；不把“已知错句消失”单独当作整答验收。
