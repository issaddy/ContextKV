# 来源支持主题选择器 v1：41轮真实开发验证

2026-10-24。两种模式各覆盖同一固定41轮，其中各36轮实际调用Qwen、各5个首轮直接保留问题。共82条ledger、72次HTTP，全部原始completion已保存。**来源资格检查能限制无源名称，但不能保证话题选择正确；目标d5t4中，来源约束模式漏选Cloudant，无约束对照反而选出了Cloudant＋Attachments。** 本轮没有执行新检索或答案生成，不能据此宣布召回、回答质量或整体性能收益。

## 数据与执行范围

原始目录：`runs/remote-h200-source-topic/runs/source-topic-selector-v1/`。导入包SHA为`97233f01303859805ceaeff87da36ad650b2f6dbdfb6ee942fd90cedd9fc990a`；原frozen cases亦随包保存，与本地逐字相同，SHA为`b8920637f40aa9f1730ca8766c0c7e04fd3bf88e9957c1fa016153bf5d156ad0`。部署源码为`58a0a08`，保留实际driver及部署记录。只读派生分析在`runs/source-topic-review-v1/{summary.json,details.jsonl,failures.json}`，复跑脚本为`runs/source-topic-preparation/analyze_selector.py`及`runs/source-topic-review-v1/classify_failures.py`。

两个mode共享当前问题、旧off实际历史、先前可见来源、三轮selector语境窗口、模型、temperature0、输入24576/context32768/输出256预算。仅source_topic过滤无资格assistant实体；`assistant_topic_unverified`是明确未溯源对照。每个实体给模型一个明示来源摘录，完整原文/offset留在宿主；模型没有读全部历史证据。完整生成QA没有被改动。对照只涉及5个已见dev会话、41个固定问题，72次调用不是72个独立语义样本。

runtime-before确认GPU0 H200、Qwen2.5-7B-Instruct固定revision `a09a35458c702b33eeacc393d103063234e8bc28`、BF16、context32768、XGrammar、`allow_auto_truncate=false`；HTTP served-name使用该本地模型路径的显式别名。`chat_template_kwargs.enable_thinking=false`在本地计数与HTTP一致。服务实际GPU池121920/host279040 tokens，起始为空，但**没有逐调用清空KV**。两个mode按任务交替顺序，后续可能有原生prefix复用，不能称均等冷态或孤立model latency。

## 输入、来源与记录合同

- 82/82期望行均存在，无重复、额外或缺失。72/72 HTTP均有completion，没有HTTP错误或输入超限。
- 72/72服务端`usage.prompt_tokens`与本地计数、冻结计数一致；source输入827–19638，unverified827–20813 tokens。这验证计数对齐，**不是远端逐token IDs证明**。
- 全部source inventory scope与冻结值相同；本旧off cohort逐轮visible=当前retrieved5的等价前提成立。该结论不扩展到含额外visible材料的任意APPEND运行。
- 已选实体的ID与冻结model目录对应，宿主记录的来源span都能回指严格更早的可见原文；检查未发现绑定损坏。**“词在源里出现”不等于这个词是适当实体，也不等于它应成为当前话题。**
- 所有11失败都保留原query及原始模型输出；没有把失败清洗、删行或回填为成功。没有qrels、人工分数或quality gate。

| 每mode共41条，其中36次模型调用 | source_topic | assistant_topic_unverified |
|---|---:|---:|
| 首轮跳过/原问题 | 5 | 5 |
| `resolved`，宿主构造了新query | 32 | 28 |
| `ambiguous`，保留原query | 0 | 1 |
| `failed`，保留原query | 4 | 7 |
| completion正常stop / length | 35 / 1 | 33 / 3 |
| 模型主动`unchanged` | 0 | 0 |

`resolved`是格式与ID检查后的状态，不是“语义正确”的标注。全41轮两mode最终query有17轮相同、24轮不同，其中首轮跳过与共同失败也会产生相同query。27轮两者均resolved，不能据此推断27轮都改善。

## 全部11失败：语法合法不代表ID合法

4个`length`输出均达到256 tokens且JSON未完整闭合；7个`stop`输出都是可解析JSON，但包含目录外ID。后者中6个把用户ID写成带前导`]`的字符串，如`"]u7"`；另外1个source输出包含未提供的实体`e2`。v1 schema只约束ID字段为string，故这类字符串在JSON语法上合法，宿主拒绝是正确行为。

下面用会话前缀表示任务，完整task ID和raw text保存在`failures.json`。

| task | mode | 原始失败内容/原因 |
|---|---|---|
| 1c041…t5 | unverified | length；user_turn_ids字符串中反复生成`]}}…`直到256 |
| 1c041…t7 | source | stop；`entity_ids=["e2","e92"]`，其中**e2未在该mode目录，e92=Data Lake是合法候选**。没有仅保留e92而伪造成功 |
| 1c041…t7 | unverified | length；从e0开始大段枚举实体ID，输出未完成 |
| 1c041…t8 | unverified | stop；用户ID为`"]u5"` |
| 1c041…t8 | source | length；user_turn_ids退化为长重复字符串 |
| b4614…t8 | unverified | stop；用户ID为`"]u7"` |
| b4614…t8 | source | stop；用户ID为`"]u7"` |
| c01c8…t5 | unverified | stop；用户ID为`"]u2"` |
| c6c3b…t9 | source | stop；用户ID为`"]u6"` |
| d5b1e…t8 | unverified | length；大量实体ID后进入包含`<tool_call>`的无效长字符串 |
| d5b1e…t9 | unverified | stop；用户ID为`"]u8"` |

这些不是HTTP故障，也不能一概称为模型不懂话题。反之，即使修复合法ID约束，下面已通过格式检查的语义问题仍存在。

## 目标d5t4与同会话的正反结果

当前问题为`Is there a limit on the file size? `，前一用户问题询问image/PDF与document一起存储。用户没有直接命名Cloudant，但历史assistant与第一轮真实来源均提供了这一产品线索。来源模式的目录实际包含`IBM Cloudant`（7个支持条目）、`Cloudant`（8个）和`Attachments`（3个），所以这次漏选**不是来源门控将Cloudant排除**。

| mode | 实际结果 |
|---|---|
| source_topic | `Conversation topic: Attachments`，再追加u3的image/PDF原用户问题；没有Cloudant |
| assistant_topic_unverified | `Conversation topic: Cloudant; Attachments`；没有另加用户问题 |

source恢复了附件对象语境，但没有补足产品；unverified在这次解析上保留了更完整的产品＋对象组合。当前没有下游检索，不能断言前者必定检索失败或后者必定正确回答，更不能把来源约束说成全面优于baseline。

同会话还显示两方向的变化：

- t2/t3两模式都能补出Cloudant；这是词法user-only视角无法直接取得的历史assistant实体，证明工程链能利用有意义的语境。
- t6当前已明确IBM Cloudant及“get started”。unverified却追加Attachments、IBM File Storage、Attachment Size Limit，把先前的不同产品/问题带回；source只加Attachments，虽然没有错加File Storage，仍偏向旧附件话题。
- t7问Lite/Standard计划，source选Inline/Attachments，unverified选Storing Large Files，都没有抓住计划比较所需的产品主题。
- t9问Python SDK，source选IBM Cloudant和Cloudant两个字面别名，至少恢复产品；unverified格式失败而使用原query。别名重复尚未合并为语义实体，不能把两个名称算两个有效主题。

## 其他会话：来源出现不等于当前意图需要

以下为assistant逐原文/历史的只读语义诊断，不是人类标注，不给总体正确率；当前问题始终保留，问题在于追加锚点可能偏移检索方向。非d5反例另经独立agent复核。

| task / mode | 当前明确对象与实际选择 | 诊断 |
|---|---|---|
| 1c041…t6 / source | 问`what is Data lakes`，选Knowledge Studio＋Admin | 来源`ibmcld_16551-0-1579`讲Knowledge Studio模型/版本存储，能证明词出现，不能证明定义Data lakes应绑定该产品。baseline此处返回ambiguous |
| c01c8…t5 / source | 问用Carthage安装，选CocoaPods＋u4“不用CocoaPods” | 同一目录里Carthage存在且有6个support，仍回到用户正在绕开的旧方式；不可仅解释为候选缺失 |
| c6c3b…t6 / 两者 | 问创建encrypted image，选`By`＋旧v3/v4比较问题 | By是源内常见连接词，不是产品/对象实体；name-span召回和source字符串资格都不能替代实体/关系判断 |
| c6c3b…t7 / 两者 | 当前已明确IBM Cloud Container Registry，选Vulnerability Advisor＋旧u4/u5/u6 | 将当前澄清主体重新拖向旧扫描产品及三个旧问题，说明“总是补历史”的偏向 |
| c6c3b…t8 / 两者 | 问Container Registry Grant Access；unverified选Vulnerability Advisor，source选IBM Cloud Container Registry＋u7 | 来源模式此处对象对齐更好，但仍需下游检索验证 |
| b4614…t7 / 两者 | 问更多电话号码及限制，均选Force Secure Trunking/SIP Authentication | 选择的是先前安全设置，未直接消解当前号码数量需求 |

source还选出了Run、Go、In、By等非实体片段；unverified也有Run、Go、After等。候选召回本来允许这些词进入以避免漏实体，真实结果说明选择器未充分去掉它们。不能通过在本已见数据上手填一张停用词表就宣称通用问题已解决。

## 成本仅是本执行顺序的开发观测

全部36次调用（含失败）均计入每mode时长：

| 指标 | source_topic | assistant_topic_unverified |
|---|---:|---:|
| completion elapsed均值 / P50 | 350.59 / 292.17 ms | 537.29 / 316.90 ms |
| completion elapsed P90 / 最大 | 517.77 / 1748.61 ms | 1216.81 / 3692.55 ms |
| wrapper wall均值 / P90 | 402.00 / 603.15 ms | 590.21 / 1357.38 ms |
| 36条wrapper wall合计 | 14.47 s | 21.25 s |

completion elapsed包括本地计数和HTTP；wrapper还包括候选/来源构造及冻结输入复核。后者为了验证会重复计数，不能直接代入生产实现成本。共享服务没有逐call清KV，候选数、输入长度、生成长度和失败分布也不同；因此这不是source模式的因果加速结论，更不是完整P3响应耗时。应把这一额外步骤与后续真实检索/生成共同计量。

## 下一步：先修输出空间，再验证语义机制

**优先的窄工程改动是显式`enum_v2`实验协议。** 按每次实际实体/用户候选ID构造JSON Schema enum，空集合只允许空array；保持prompt/messages、候选提取、3轮窗口、256预算和模型不变。不清洗`"]u7"`、不部分挽救失败、不改v1分数，v1仍为默认。需要实际XGrammar编译检查，且按同41全部重跑，统计raw变化、格式失败与合法但偏题的选择。动态grammar编译成本要记录；合法ID不保证正确主题，也不以任意实体数量上限掩盖枚举问题。

**语义改进应针对当前意图与产品/对象关系，而非仅合法词袋。** 本次最有价值的负例是“Cloudant和Attachments都可选却只取对象”、明确Carthage仍选CocoaPods、创建镜像却选By。接续可比较先定位相关历史轮/对象、再选择来源支持的产品—对象组合，并把“无需补全”作为实际可选决策；来源资格保持为依据，不取代意图解析。它必须在同41正反例和额外开发材料中检验，不能只调d5t4。

同时保留作者固定ConvSearch-R1 3B学习式CQR基线，使用其full-history/prompt/stop格式并单列窗口、模型和成本差异，避免只围绕自制selector反复调提示。随后应将v1合法选择及失败回退完整送入实际相同预算检索，才能判断这些新增查询的价值。v1分析没有读取qrels、修改生产或重写冻结输入；随后新增的可选v2实现与独立冻结如下。

## enum_v2：已实现并冻结，尚无实际v2模型结果

可选接口为`TopicResolver(..., schema_mode="enum_v2")`，检索配置为`selector_schema_mode: enum_v2`。默认`static_v1`沿用原静态schema、消息和trace；v2单独记录协议`source-topic-selector-enum-v2`，每次调用trace保留完整实际`response_format`。仅候选ID集合进入grammar，不把“应该选谁”、结果标签或正确产品放入约束。非空集合不增加数量上限，也不使用`uniqueItems`；重复ID仍按既有宿主规则保序去重。`resolved`必须非空、另外两状态必须空的语义合同仍在宿主检查，没有增加条件schema分支。

固定XGrammar源码对非空enum生成字符串分支；空enum被明确拒绝，因此空catalog使用`items: {type: string}, maxItems: 0`，其array生成器直接返回空数组。[固定enum实现](https://github.com/mlc-ai/xgrammar/blob/82505d0d987c36a4209fb3d8571cf6b0f28b5acd/cpp/json_schema_converter.cc#L1359-L1370)、[空array实现](https://github.com/mlc-ai/xgrammar/blob/82505d0d987c36a4209fb3d8571cf6b0f28b5acd/cpp/json_schema_converter.cc#L2278-L2294)。这些特征不在固定SGLang XGrammar拒绝列表中，但源码支持不等于当次服务器成功编译。[固定检查器](https://github.com/sgl-project/sglang/blob/3f36fc08331f64cd63c4fbbdf39cc402f38bb38b/python/sglang/srt/constrained/json_schema_validation.py#L154-L185)

`runs/source-topic-preparation/freeze_enum.py`从原v1 cases派生`frozen-v2`，不运行tokenizer、模型或检索。逐项验证41轮的原query、完整history/source、候选及顺序、72组messages和输入计数保持相同，仅response_format改变。新cases SHA为`963537420e1628070e88ac6dcbb66c1a5c85c03b90a05a1e7a908be37f037d0b`，旧v1文件未变。manifest显式保留temperature0、top_p未指定/请求省略、enable_thinking=false与原预算（避免把服务端sampling_defaults=model改成客户端top_p1）；execute仍重新核对本地tokens，服务端usage将用于实际对齐。全41的预算覆盖不变：24576输入预算无超限，8192敏感性仍是source14/36、unverified18/36。

复用现有driver，以下命令默认只plan；由统一GPU操作者另行决定`--execute`：

```bash
python runs/source-topic-preparation/probe_enum.py \
  --inputs runs/source-topic-preparation/frozen-v2 \
  --tokenizer-path models/Qwen2.5-7B-Instruct
python runs/source-topic-preparation/run_selector.py \
  --inputs runs/source-topic-preparation/frozen-v2 \
  --output runs/source-topic-selector-v2 \
  --tokenizer-path models/Qwen2.5-7B-Instruct
```

短probe在当次服务Python环境内记录XGrammar/SGLang版本及runtime，CPU编译非空/全空/仅user三种schema，并检查非法用户ID无法匹配；随后仅发首个冻结真实输入和明示合成空目录两次HTTP，保存raw、finish、失败及schema检验，不回退、不重试。这两次是协议可运行性检查，不并入82条正式开发ledger或语义评分。主v2仍完整82行、72次预期调用；动态grammar编译、共享KV及输入输出差异都需进入后续成本解释。当前只完成CPU合同测试和plan，没有宣称格式或语义已改善。
