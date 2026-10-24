# Granite BYOC：旧29项回归与16项结构变体

> 归档记录：保留当时的设计、结果与复用依据；其中“下一步”和执行计划不代表当前任务。当前工作仅以[实现清单](../implementation_plan.md)为准。

2026-10-24。两个冻结BYOC标准均已执行。**新增标准没有表现为稳定的整体改进**：`relation_scope_v1`修正了一条恰当拒答，却新增正确联合换算和事实/引用维度混淆的误报，真实长答案的对象范围错误仍未检出；`citation_binding_v1`在新变体中识别了错引和未知ID，但仍漏掉旧 `p018`。原生默认保持不变。所有结果是开发诊断，不是人工评分、五维验收或正式test非劣结论。

## 固定输入、执行与独立核验

原始结果位于 `runs/remote-h200-granite-byoc/runs/`：`granite-byoc-development-v1/`为原29项，`granite-byoc-variants-development-v1/`为新16项；每组都有三个profile的独立ledger。53个导入文件的archive SHA为 `350bbff47b3c3503f34fe81fc95e0d3155eb68cca6cb8f35c6252a669c5e3b42`。源码部署 `a2cfff7645b8e69c9e8931c5bc5cf7954a5eda5a` 的273项tracked文件已逐一与导入的 `source-deployment.json`核对。原始数据、criterion及预设期望均未修改。

独立脚本和派生结果为 `runs/granite-byoc-review/analyze.py`、`review.json`，可复跑：

```bash
.venv/bin/python runs/granite-byoc-review/analyze.py \
  --root runs/remote-h200-granite-byoc/runs \
  --tokenizer-path /path/to/fixed/granite-guardian-4.1-8b \
  --output runs/granite-byoc-review/review-rerun.json
```

全部135行都保留：126次实际HTTP、9行引用协议未覆盖，0失败、0输入/输出截断。模型固定 `ibm-granite/granite-guardian-4.1-8b@ab01ccca5dcfb80246369a086a4a87a29198f5af`，BF16、TP1、部署窗口32,768、temperature 0、no-think、64输出上限。所有实际输出都为12 tokens的空think加一个yes/no，finish均为stop。

独立核对了135行的完整visible正文、reference ID、实际history角色与文本、当前query/answer，以及profile/version与冻结标准。对126个已执行输入，以固定官方tokenizer重新渲染，再与ledger、请求和服务端回显逐token比较，全部一致；原输出IDs和usage也相符。原29默认输入及输出IDs与[先前v2](granite_judge_validation.md)的29项逐一相同。引用模式的9行未覆盖仍保留完整消息/来源，risk与score均为null，没有伪造IDs或HTTP。

旧29各profile的audit/labels字节相同，且与上轮相同；新16各profile也保持相同audit/labels，逐项匹配运行前固定的 `tests/fixtures/granite_byoc_variants.json`。标签只在输入/执行核对之后离线关联，未进入模型消息。新16是预先构造的合成结构变体，**不是独立人工meta-test**，也没有为原生ungroundedness另行预设一套标签。

## 覆盖、预算与成本边界

| 输入集 / profile | ledger行 | 实际评分 | 未覆盖 | 输入tokens总数 / 最大 | 输出tokens总数 |
|---|---:|---:|---:|---:|---:|
| 旧29 / ungroundedness | 29 | 29 | 0 | 20,503 / 4,092 | 348 |
| 旧29 / relation_scope_v1 | 29 | 29 | 0 | 24,476 / 4,229 | 348 |
| 旧29 / citation_binding_v1 | 29 | 23 | 6 | 13,021 / 3,999 | 276 |
| 新16 / ungroundedness | 16 | 16 | 0 | 5,239 / 345 | 192 |
| 新16 / relation_scope_v1 | 16 | 16 | 0 | 7,431 / 482 | 192 |
| 新16 / citation_binding_v1 | 16 | 13 | 3 | 5,361 / 426 | 156 |

总输入76,031、输出1,512 tokens，所有完整输入加64预算均满足窗口。旧引用协议未覆盖 `p021/p024/p025/p026/real-001/real-002`；新协议未覆盖两种无引用拒答及只有 `[1]` 的普通数字引用。这是解析/协议覆盖边界，不是这些答案被判正确。包含可解析未知ID的 `unknown-in-group` 确实进入模型，规则层另保留未知ID。

六组按原生、relation、citation顺序串行执行，旧29先于新16。Qwen大池共驻但idle、Granite服务用于本次诊断，没有同时进行benchmark。各组HTTP累计时间约2.555、2.418、1.874、1.331、1.322、1.055秒；存在共享prefix复用，且引用组调用数更少，不能将较小总时长解释成标准的因果加速。输入更长的新标准也没有被计为“免费优化”。本次没有独立加载/冷态重复或统一推理输出长度的速度实验。

## 旧29：一项拒答修正伴随回归

原生默认的22个事实控制仍全部吻合；relation同22项仅20项吻合。这里沿用既有sidecar：entailed对应风险no，contradicted/insufficient对应yes；并未修改预期以适配新输出。relation相对默认只改变三条，其他26条原生yes/no相同。

| 样本与原文关系 | 默认 | relation | citation | 解释 |
|---|---|---|---|---|
| `p013`：来源分别给96小时和24小时/天，答4天并引用两个来源 | no | **yes** | **yes** | 正确联合推导，两新标准均误报。关系标准明确允许来源给定的换算，不能以“不逐字出现4天”替它辩解。 |
| `p018`：端口7443由E_01支持，答案只引用给timeout的E_02 | no | **yes** | **no** | visible事实仍受支持，relation的yes不能算新增事实能力；专门引用标准反而漏检错绑。无推理文本，不能声称relation内部成功识别了错引。 |
| `p021`：来源仅提CSV，答案说明未给snapshot retention | yes | **no** | 未覆盖 | relation修正这条恰当拒答误报；未覆盖不是citation正确评分。 |
| `p024`：来源给14天，答案却说没给 | yes | yes | 未覆盖 | 两事实标准均指向风险，不能用它抵消p021旧误报。 |
| `p025/p026`：有效JSON和尾逗号无效JSON | no / no | no / no | 未覆盖 | 不构成代码语法正确性验证，静态诊断继续单列。 |

旧sidecar只有 `p017/p018` 显式定义了citation_verdict，两者在引用profile中分别正确、漏检，故该显式引用对子仅1/2吻合。其余visible事实标签不能自动被改造成一套完整引用金标。即使直接审查可确认 `p013` 的引用支持，仍将其作为具名反例报告，不回写原期望。

引用漏检不来自模板丢失：`p018`的完整documents含 `doc_id=E_01` 的端口正文及 `doc_id=E_02` 的timeout正文；固定tokenizer重渲染与native请求/回显一致，实际解码也能定位两个ID和正文。这里不能通过补一个不存在的“ID缺失修复”来解释模型行为。

## 新16：全部预设结果与维度分离

下表true/false均表示**相应标准的风险**，不是五维质量分。原生列只报告输出；它没有独立预设期望，不借relation标签给它计算正式准确率。relation为13/16吻合，citation为12/13吻合、另3未覆盖；分母和目标不同，不能把这两数直接当标准排名。

| 结构变体 | relation / citation预设风险 | 默认 | relation | citation |
|---|---|---|---|---|
| table-object-supported | false / false | no | no | no |
| table-object-inclusion | true / true | yes | yes | yes |
| aggregate-supported | false / false | no | no | no |
| aggregate-scope-changed | true / true | yes | yes | yes |
| explicit-footnote-supported | false / false | no | no | no |
| explicit-footnote-lost | true / true | yes | yes | yes |
| derived-unit-supported | false / false | **yes** | **yes** | **yes** |
| derived-unit-wrong | true / true | yes | yes | yes |
| history-not-source | false / false | no | no | no |
| history-copied | true / true | yes | yes | yes |
| justified-abstention | false / 未覆盖 | no | no | 未覆盖 |
| unsupported-abstention | true / 未覆盖 | yes | yes | 未覆盖 |
| support-only-elsewhere | false / true | yes | **yes** | yes |
| unknown-in-group | false / true | yes | **yes** | yes |
| ordinary-number-reference | false / 未覆盖 | no | no | 未覆盖 |
| joint-citation-supported | false / false | no | no | no |

relation与原生的16个输出完全相同，引用profile覆盖的13条也与它们同向。因此，新变体未显示预设新标准在原生输出基础上的额外改善；不同profile的期望和覆盖差异仍有诊断价值。

三个关键限制均可直接依据原文复核：

- `derived-unit-supported`给exactly3072 KiB及1 MiB=1024 KiB，答案3 MiB引用同一来源，换算和对象均正确。relation/citation预设false但均给yes，是实际误报。与旧 `p013` 一起说明换算支持不稳定；不能因新 `joint-citation-supported` 的3小时×60→180分钟通过就宣称能力已改善。
- `support-only-elsewhere`的端口7221由可见E_a明确支持，但答案错引仅给timeout35秒的E_b。relation相对既定“全可见事实支持”预期是误报；citation给yes符合错引风险。不能为了让relation看起来更好，把引用错误改成事实不受支持。
- `unknown-in-group`的事实由E_a支持，但引用组多出不存在的E_f00d。规则正确记录未知ID，原生模型仍实际执行；citation yes符合预期，relation yes未保持此次预设的事实/引用维度分离。没有think解释，不能断言这一输出的内部原因。

新 `support-only-elsewhere` 中citation能检出错引，而语义结构相近的旧 `p018` 仍漏检；来源ID、措辞或数字变化的影响没有被隔离，不能将这两个例子的差异提升为泛化证明。

## 三条真实回答与后续选择

`real-001/real-002/real-003`在默认和relation下均为no，保持了先前的关键局限：第一条把分列的document限额扩成“包含附件”的范围，第二条加入限额保证不影响整体性能的断言，均无相应原文支持；第三条核心单附件限额有直接来源。citation只覆盖有明确引用的第三条并给no，前两条未覆盖。当前答案均不包含50MB，那个错误在此前assistant历史中，不能拿它替代最后assistant的评估目标。详细来源在[原生真实审查](granite_judge_validation.md#三个真实回答逐源审查)。

当前保留默认标准，两个BYOC作为显式诊断选项；没有按本次错误修改措辞、标签或阈值，也不据三个真实例局部调prompt。测量能力之外，A的父条件/来源绑定和C的写入准入仍分别推进，避免以增加judge调用代替机制改进。

一个明确的后续候选是官方think模式：保持三个criterion原文和相同29项输入，事先固定更大的单次输出预算（例如统一2,048），完整运行而不是只重试失败例，分开记录语义变化、截断与额外成本；若运行，no-think结果保持原样。官方模式会在score前输出解释，但解释存在并不证明判定或引文正确，仍需核对原文。**此候选尚未执行，本报告不预测其收益。** [官方固定think/no-think规范与示例预算](https://huggingface.co/ibm-granite/granite-guardian-4.1-8b/blob/ab01ccca5dcfb80246369a086a4a87a29198f5af/README.md)。
