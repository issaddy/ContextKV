# A：固定SDK证据包的12次真实生成对照

> 归档记录：保留当时的设计、结果与复用依据；其中“下一步”和执行计划不代表当前任务。当前工作仅以[实现清单](../implementation_plan.md)为准。

2026-10-24。三个开发案例、原材料/追加SDK材料两臂、两次反向顺序均已完成：12次成功、0次截断，六个“案例×臂”的重复输出ID与文本分别完全一致。**新增材料修正了一个实际参数错误，但仍出现初始化合同错误、无谓代码扩写和缺少引用；不能据此认定整体质量改善。** 这是源证据扩展的局部开发probe，不是正式五维评分、质量非劣或同材料缓存策略比较。

## 输入、执行及绑定

原始结果位于 `runs/remote-h200-api-frozen/runs/api-frozen-pairs/h200-order-pairs/`。执行源码部署为 `20fef314241c60530775a41f07fdba15c40752fd`，部署记录含293个tracked文件；本地独立核对其Git blob清单与该Git树一致，实际执行的ignored driver SHA也与本地文件一致。固定SGLang为 `3f36fc08331f64cd63c4fbbdf39cc402f38bb38b`、补丁0ed；Qwen2.5-7B-Instruct为 `a09a35458c702b33eeacc393d103063234e8bc28`，BF16/TP1/page64，GPU121920/host279040，write_through、kernel/layer_first、TMA关闭。请求串行运行，服务本身允许最多8个running requests。

每例按 `original → sdk_sources`、再 `sdk_sources → original` 执行。每次请求前独立cold，GPU/host实际占用和活动/排队请求均为0；没有生成预热。Q0、greedy、1024输出上限、正常EOS、32768上下文窗口相同。记录完整冷态、epoch准备、输入hash、raw answer/output IDs/finish reason、receipt、退休ACK和独立准备/生成耗时。

独立分析核对了12个输入身份、请求/attempt/epoch/generation/restart关联、零命中后的Prefill计数、`device + host + prefill = input`、输出长度、退休关联与固定运行顺序，未发现不一致。原query、全部实际前序QA和原可见证据保持原文；三个original输入逐token等于保存的P3 audit，SDK臂只在末轮Question前追加版本适用声明及原源码块。新增块沿用CorpusIndex版本闭包，完整原文span和旧ref不变。

| 案例 | 输入tokens 原/SDK | 净增输入 | SDK块数 | 输出tokens 原/SDK（两次各自相同） |
|---|---:|---:|---:|---:|
| d5…第3轮：图片/PDF附件 | 3823 / 5872 | 2049 | 3 | 567 / 451 |
| d5…第9轮：Python SDK位置 | 6291 / 11417 | 5126 | 6 | 466 / 600 |
| b461…第4轮：customization ID含义 | 5041 / 8707 | 3666 | 4 | 504 / 461 |

全部输入无需截断，预留1024输出后最小余量20327。SDK版本是明确声明的来源适用版本，用户没有指定唯一release：Cloudant0.0.27 / `c98dd5b…`，Watson7.0.0 / `c373366…`。未把旧语料替换成当前网页，未执行SDK或答案代码。

一个必要的对照边界：**本次三个original答案都不等于此前P3保存答案**，尽管冻结输入ID精确相同。因此以下结论以这次新original臂为对照，不把旧回答中消失的错误都归因SDK扩展。没有进一步实验区分跨运行差异的原因。

## 六个唯一答案的来源审查

以下是助手对完整实际答案与来源的逐项比较，不是人工盲审标签。两次重复的答案完全相同，六份审查对应全部12个实际输出。可复核答案字符span、原证据key/ref/body span和额外固定源码位置，见 `runs/api-frozen-pairs/review-h200-v2/source-review.json`。

### 附件：本次两臂已有正确字段，但示例仍有缺陷

两臂均回答可以将文件作为附件，且都使用 `_attachments`。该结构、BASE64表示以及附件过多/过大的性能注意事项，原材料 `ibmcld_00556-7-1696` 与 `ibmcld_00556-10831-12062` 已明确提供。SDK臂新增的“PUT携带最近 `_rev`”也已在原00556-7段中。因此本次不能声称SDK包修复了先前普通 `image` 字段冒充原生附件的问题：新original本来就正确使用该字段。

两臂的JSON围栏均含 `//` 注释，现有 `bench.code_audit.json_diagnostic` 判定为非法JSON。两份示例又将JPEG文件名/`image/jpeg`与 `iVBOR…` PNG样式前缀放在一起；原00556-10831段本身已有同类MIME/缩略数据不一致，不能全部归因于模型。没有解码验证这些完整示例数据，也没有把解析失败当成整个答案都无事实支持。

original另给外部BLOB存储/URL方案，其一般建议有原文依据。SDK臂更短且没有这段扩展，不足以直接判“更好”或“更差”；它的核心附件回答仍完整。新增 `application/pdf` 是合理MIME形式的陈述，但这些visible块没有明示该字面值；这只是来源覆盖说明，不是将其判为事实错误。两臂均没有Evidence ID引用。

### SDK位置：doc_id改正，但初始化和范围出现新问题

original提供安装命令和PyPI资源链接，满足“在哪里找SDK”的主要请求；同时扩写CRUD代码。其 `get_document(db=..., document=...)` 与固定0.0.27签名不符：原签名要求 `doc_id`，`**kwargs`不能补足缺失的必需参数。该签名在original臂不可见，属于额外固定版本的离线合同诊断。[固定get_document L1721](https://github.com/IBM/cloudant-python-sdk/blob/c98dd5b640538978790c777049f3c2e9df8e816a/ibmcloudant/cloudant_v1.py#L1721)。

SDK臂实际改成了 `get_document(db=..., doc_id=...)`，`post_document(db, document)`也与新增可见签名一致，这是本次配对可确认的局部改正。但它又两次生成 `CloudantV1(url=..., iam_apikey=...)`。同一固定源码的构造器只声明 `authenticator`，没有这些关键字或 `**kwargs`；该调用不符合本次声明适用的0.0.27版本。[固定构造器L61](https://github.com/IBM/cloudant-python-sdk/blob/c98dd5b640538978790c777049f3c2e9df8e816a/ibmcloudant/cloudant_v1.py#L61)。

**构造器不在本次新增包中。** 离线审查读取的是此前已保存的同版本完整SDK文件，没有将它倒填为模型曾见证据。这显示仅补操作方法与schema还不是完整的最小用法合同。original的 `authenticator=`初始化至少与这份构造器声明相符，但本轮没有验证外部sdk-core认证类或实际网络行为，也不能因此宣布其整段可运行。

SDK臂从466变为600输出tokens，提供多段重复用法，去掉original的明确资源链接；安装命令也没有说明代码的0.0.27适用范围。它仍指出正确包名，不能说完全没有回答，但更多代码不能算位置问题更完整。此前P3的 `post_database`错误在本次两臂都没有重现，不能据此给新包记一次修复。两臂仍无Evidence ID引用。

### customization ID：定义有据，新Python教程仍未形成完整合同

两臂的GUID定义均有原 `ibmcld_13396-1754-3852` 支持。original的language-model WebSocket参数与原13361/13384例子对应；SDK臂的acoustic参数示例与原13263段对应。新增的 `list_language_models().get_result()` 及 `response['customizations']`表面用法有可见的固定方法和 `LanguageModels`说明支持，回答也将列表结果明确称为language model IDs。不能仅因另一段出现acoustic示例，就断言它实际把language ID传给了acoustic参数。

不过，SDK臂新Python段使用 `SpeechToTextV1(iam_apikey=..., url=...)`；固定7.0.0构造器声明的是 `authenticator`、`service_name`，没有这两个关键字或 `**kwargs`。[固定构造器L75](https://github.com/watson-developer-cloud/python-sdk/blob/c373366ef573c553fa05b260660c2aa88e1b1449/ibm_watson/speech_to_text_v1.py#L75)。此构造器同样未在新增包内；新增listing资料诱导了超出定义问题的教程，而初始化仍缺依据。

两臂还将使用custom model的条件说得过宽，例如笼统称建立WebSocket连接需要该ID。原文讲的是指定该custom model时的用法，不足以支持一般连接都必须带customization ID。原先P3的错误GET范围/versions对象数组未在本次original中复现，不作为SDK臂改善计数。所有代码均未执行，两臂无Evidence ID引用。

## 引用、成本与可复算入口

共享 `evidence-id-groups-v2` parser检查全部12答，包括组合引用；结果均为0个Evidence ID引用。没有unknown ID，不等于引用合格或引用支持正确。答案中的普通资源URL另行按来源审查，不冒充Evidence ID。

两次的平均 `generate_wall_ms` 仅作描述：附件2756/2243、SDK位置2330/3158、ID定义2461/2379（原/SDK）。新增输入与输出长度均变化，只有两个反序重复；这些值不提供总体加速、显著性或P3端到端收益结论。

产物与复跑：

```sh
# 新目录，读取原始结果；不调用模型或执行答案。
.venv/bin/python runs/api-frozen-pairs/analyze.py \
  --results runs/remote-h200-api-frozen/runs/api-frozen-pairs/h200-order-pairs \
  --output runs/api-frozen-pairs/review-recheck

# 固化已完成的六份助手来源审查及精确原文位置。
.venv/bin/python runs/api-frozen-pairs/source_review.py
```

`review-h200-v2/summary.json`保存12条机械绑定、顺序、失败/截断分母和重复一致性；`review-cases.jsonl`保留完整原QA、原visible材料、新包、实际回答和旧P3回答，并把五项语义审查栏保持独立；`source-review.json`才是本次人工阅读式助手结论，明确不是human-blind评分；`source-binding.json`记录源码/部署绑定。旧raw、原SDK包、原preview及早期准备报告全部保留。

## 有限的后续工程闭环

1. **把操作签名包补成最小可用的版本合同。** 复用同固定SDK的官方初始化/认证示例或公开factory，显式关联操作与客户端建立方式；若需要外部sdk-core，要另外固定其适用来源，不从包最低版本猜唯一release。公开类的构造器是API使用合同，可作为受限入口处理，不必开放任意私有属性或实现通用代码执行器。先对同三例做“当前包/补初始化合同”对照，仍保留全部失败和不必要教程；不预设添更多代码一定有收益。
2. **复用已有静态诊断检查整个代码块，而非只检查一个方法名。** 当前JSON注释问题已能被 `bench.code_audit`识别；下一步可将已明确绑定包/类/版本的构造器与方法关键字作联合离线检查，给原span与缺合同清单。MIME/示例数据也须区分源例自身不一致与新生成错误。诊断不改写原答案，更不把AST/JSON通过当可执行性或答案质量证明。
3. **分开A效果选择和B6最终验收。** 这三个probe不能填入 `quality_eligible=true`。待全41轮共同A输入配置确定后，再以相同材料、输出预算和资源比较强简单候选（已有固定阈值/周期/REBUILD及合理原生写策略），按同调参预算选B6；对完整实际答案做有输入绑定的独立五维评价/盲审，并保留版本、接口和问答范围错误。随后冻结方法、成本artifact、评价协议与test配置。现有judge负例不能用本次助手审查替代正式非劣，新增SDK材料也不能只给Ours而不给对照。

这一轮说明来源适配已能实际进入生成并改变局部API用法，也暴露了“方法签名正确、整段用法仍错误”的下一处工程缺口。该结果支持继续完善证据包及验证合同，尚不支持将其设为默认检索增强或宣布整体质量贡献成立。
