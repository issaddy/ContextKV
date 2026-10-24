# A：版本化API证据包的最小实施方案

> 归档记录：保留当时的设计、结果与复用依据；其中“下一步”和执行计划不代表当前任务。当前工作仅以[实现清单](../implementation_plan.md)为准。

2026-10-24。依据[三项机制优先级](ac_extension_priorities.md)第3项，已完成保存答案的离线扫描和三个具体来源复核。结果支持把“缺接口材料”“材料已有但字段用错”“题目不需要而额外扩写”分开处理。[独立离线adapter与CLI](../guides/api_evidence.md)已实现，固定真实源码10个片段与原预览逐字/逐span一致；随后[12次真实生成对照](api_evidence_validation.md)完成，保留局部参数修正、初始化合同错误与额外教程，未建立总体质量收益。

## 扫描范围与可复核产物

扫描两臂 `p3-current-topic-fusion-v1` 和三臂 `p3-table-focus-v2` 的全部205份实际回答，即同41个dev任务的五份轨迹。110份包含代码围栏，8份输出截断仍保留。只从每轮实际 `visible_evidence_manifest` 取完整可见正文，不以当前top5代替全部可见证据，也不使用归档但不可见的块、test、reference答案或未来历史。

本地忽略产物位于 `runs/api-evidence-review/`：

- `scan.py`、`scan.jsonl`、`summary.json`：全部回答、原围栏偏移、Python AST调用及关键字、端点片段、完整可见正文中的字面位置；五个原audit文件SHA留在summary。
- `reviewed_cases.json`：以下三例的原任务ID、原EvidenceKey/ref、正文与答案字符偏移、具体复核理由。范围均为从0开始的半开字符区间。
- `official_sources.json`、`official/`：固定官方源码与完整上游LICENSE。失败的示例文件请求也保留为404，未虚构已取得官方完整用例。
- `preview_packets.py`、`packet_preview.json`：从两份固定SDK文件静态抽取原始连续源码区间、签名与类说明，保留版本、符号、行号和原文；不导入SDK，不执行答案代码。

复跑：`.venv/bin/python runs/api-evidence-review/scan.py`，随后运行同目录 `preview_packets.py`。39次Python属性调用中29次未在可见正文找到同名字面串，只是复核线索；名称出现也不等于签名受支持。24个JSON围栏无法直接解析，其中可能含片段、HTTP文本或省略号，不能直接当24个错误。205回答不是205个独立任务。这些统计均不生成质量总分或人工标签。

## 三个有原文位置的样本

三例均来自融合P3的 `current_topic_rrf` 臂。原材料版本统一为 `mtrag-53fef6f5917c7e10c436f697e69a955dbc795d81`；完整任务ID和audit路径在上述JSON中。

### 1. d5…第9轮：正确库链接后扩写了错误Python调用

用户问“where can I find python SDK for pushing and fetching data?”。已取回 `ibmcld_00462-1307-2903`（`E_dfd5c4ff08`）中的安装说明 `ibmcloudant>=0.0.27`，正文[271,290)，并有官方SDK链接。可见材料没有相应Python CRUD签名；出现的部分API用例属于Node.js。

回答却加入 `service.post_database(database=database_name)`，答案[823,868)，并使用 `post_document(database=...)`、`get_document(database=..., document=...)`。因此现有源不足以证明这些Python调用；仅给正确链接与安装入口已经能回答本轮主要请求，额外教程需要另有依据。

为形成可检验版本，固定官方 `IBM/cloudant-python-sdk` **v0.0.27 / c98dd5b640538978790c777049f3c2e9df8e816a**。该文件直接定义创建入口 `put_database(db, ...)`，写文档 `post_document(db, document, ...)`，读文档 `get_document(db, doc_id, ...)`，未定义 `CloudantV1.post_database`。答案中的参数不能满足这些必需参数；签名的 `**kwargs` 不能补足缺失的 `db`/`doc_id`。这是静态合同判断，不是实际网络执行结果。[创建数据库L419](https://github.com/IBM/cloudant-python-sdk/blob/c98dd5b640538978790c777049f3c2e9df8e816a/ibmcloudant/cloudant_v1.py#L419)、[写文档L898](https://github.com/IBM/cloudant-python-sdk/blob/c98dd5b640538978790c777049f3c2e9df8e816a/ibmcloudant/cloudant_v1.py#L898)、[读文档L1721](https://github.com/IBM/cloudant-python-sdk/blob/c98dd5b640538978790c777049f3c2e9df8e816a/ibmcloudant/cloudant_v1.py#L1721)。

**版本边界：** 冻结材料的 `>=0.0.27` 只是最低版本要求，不能据此认定用户唯一目标就是0.0.27。该版本是明确声明的开发证据包版本；上线需由项目依赖/用户约束指定目标，无法确认时标明适用版本，不自动改成今天最新版本。

### 2. d5…第3轮：附件结构已有，却改成普通JSON字段

用户问把图片或PDF随文档保存。原材料 `ibmcld_00556-7-1696`（`E_bd58806373`）正文[347,359)明确 `_attachments`；`ibmcld_00556-10831-12062`（`E_b4c7fd3838`）[405,419)给出附件名称到 `content_type`/`data` 的嵌套结构。回答却将普通 `"image": { ... }` 字段（答案[939,949)）称为内联附件。普通JSON当然可以保存图像文本，但不会因此自动获得原生附件语义。

本例首先是**使用已有结构的错误**，不能简单归因为检索没找到API。固定SDK的 `Attachment` 和 `Document.to_dict` 为后续证据包提供额外静态映射：`attachments` 序列化为 `_attachments`。这仍是新增来源，不倒填成当时生成已见材料。[Attachment L8700](https://github.com/IBM/cloudant-python-sdk/blob/c98dd5b640538978790c777049f3c2e9df8e816a/ibmcloudant/cloudant_v1.py#L8700)、[Document字段映射L10879](https://github.com/IBM/cloudant-python-sdk/blob/c98dd5b640538978790c777049f3c2e9df8e816a/ibmcloudant/cloudant_v1.py#L10879)。

还有一个源质量问题：原00556-10831段的示例本身把 `image/jpeg` 与 `iVBORw0KGgoAA...` 缩略数据并列，后者是PNG样式前缀。因此不能把这段当干净的MIME示例标准，也不能把所有类似混用归因于模型。证据包应保存这一不一致及原位置，不能补写“已验证正确”的用例；本轮不解码、执行或修写该答案。

### 3. b461…第4轮：GUID定义有源，列表范围和响应结构缺源

用户只问“what is the customization ID here?”。`ibmcld_13396-1754-3852`（`E_dc1dfc091b`）正文[78,82)的GUID支持基本定义；`ibmcld_13374-8726-10599`（`E_acf3d23973`）[18,36)包含 `/v1/customizations`，但原句是 **POST到corpora子路径**。它不能直接支持回答新增的 `GET /v1/customizations`（答案[1251,1273)）、“all custom models”范围和完整JSON结构。端点字符串相同并不足以证明HTTP操作相同。

固定官方 `watson-developer-cloud/python-sdk` **v7.0.0 / c373366ef573c553fa05b260660c2aa88e1b1449** 后可以进一步确认：该GET路径确实用于列出**custom language models**；acoustic models有单独的 `/v1/acoustic_customizations`。`LanguageModel.versions` 类型为 `List[str]`，描述base-model版本；原回答构造 `[{"version":"1.0"}]`，另用 `base_model`，与此版本的 `base_model_name` 不同。[language列表L1525](https://github.com/watson-developer-cloud/python-sdk/blob/c373366ef573c553fa05b260660c2aa88e1b1449/ibm_watson/speech_to_text_v1.py#L1525)、[acoustic列表L3030](https://github.com/watson-developer-cloud/python-sdk/blob/c373366ef573c553fa05b260660c2aa88e1b1449/ibm_watson/speech_to_text_v1.py#L3030)、[响应字段L6343](https://github.com/watson-developer-cloud/python-sdk/blob/c373366ef573c553fa05b260660c2aa88e1b1449/ibm_watson/speech_to_text_v1.py#L6343)。

这纠正了“源中没有GET所以GET不存在”的过强判断，也说明证据包需要绑定**方法＋路径＋模型类别＋响应字段**。SDK7.0.0是额外固定试验版本，原问题未唯一指定；路径的API v1与SDK发布版本不可混为一谈。

## 最小adapter与复用边界

已实现独立离线 `adapters/api_evidence.py`，核心为 `extract_sdk_records(source_path, source_spec)`，输出现有 `CorpusIndex.from_records` 能直接读取的记录及manifest。初版支持本任务已核的Python SDK原文；有实际需求再接OpenAPI，不先做多语言训练框架。

1. `source_spec` 显式给repository、固定commit/tag、源文件、语言/包名、目标版本状态和选择的公开符号。SDK代码用AST定位类、方法签名及docstring，HTTP方法/路径只提取源码中可确定的字面值。遇到动态路径或继承外部实现就保留原文与未知范围，不推断缺失事实。
2. 每条记录的 `text` 是连续原文区间，`span` 对应整份源文件字符偏移，`source_uri` 指向固定commit和行号，`version` 用该固定源版本，`anchors` 保存原符号；许可和source-spec放离线manifest/provenance。不同版本或语言是不同记录，不重写旧MTRAG EvidenceKey/正文。包名与版本只用于限定来源，不是答案标签。
3. 一个小包由签名/说明、确有声明的参数或响应类型、适用条件组成。原方法明确返回或接收的schema才作为依赖，通过已有 `CorpusIndex.resolve` 条件闭包加入；手工source-spec须公开其选择依据。原始连续文本仍是事实依据，辅助的对象/字段关系都回指其span，不能生成看似官方的修正教程。用例只在固定源确实存在且适用时纳入，缺失就记录缺失。
4. 检索继续复用现有BM25/BGE与相同公共预算；先以显式包/语言/版本筛源，再按当前用户操作及允许的历史语境检索。补充证据由普通EvidenceBlock和当前required处理，保留完整QA与旧正文，不新增在线哈希、独立图存储或judge服务。静态合同诊断可列必需参数缺失/当前类未声明的符号，但不把外部动态属性判定为运行时必然不存在。

DocPrompting的成熟能力是“依据自然语言意图取相关文档，再以文档生成”，这里复用这个分工；不复制其FiD训练、CoNaLa/tldr实验或声称其全部方法重现。作者固定仓库及LICENSE均为Apache-2.0。两份IBM固定源也有Apache-2.0文件头和LICENSE，产物保存原许可；初始实现只用标准库AST及项目已有检索/证据接口。[DocPrompting固定README](https://github.com/shuyanzhou/docprompting/blob/b7e0e626d47a0649090465bb15c91192fee8777f/README.md)、[LICENSE](https://github.com/shuyanzhou/docprompting/blob/b7e0e626d47a0649090465bb15c91192fee8777f/LICENSE)。

本项目可检验的扩展在于：将包/版本/操作范围与原文span、完整QA、统一required和实际缓存成本一起维护，并核查生成是否正确使用这些合同。普通文档RAG、AST本身及官方SDK事实不作为首创贡献。

## 小规模验证与公平比较

先用以上三例的完整历史冻结开发probe，保留“只需链接/定义”与“确需操作步骤”两类问题，比较原材料、同版本说明包、说明＋签名/schema包。单独记录固定证据包的新增token预算、检索/解析成本、答案是否完整、额外无源代码、显式调用/字段错误和引用绑定。第二例应另做“只对已有schema提供原span绑定”对照，否则无法区分补资料与正确使用已有资料的收益。原文例子的MIME不一致单独列出，不将它充作正确标签。

不同语言/版本、同名但不同模型类别的负对照使用固定来源，并在研究输入中明确其适用性；它们不能作为默许模型执行错误代码的测试。源代码文本足以完成这一轮静态合同验证，不需要执行答案、真实数据库写入或新模型训练。若以后测试可执行性，须另建隔离测试合同，不能把本次AST解析称为运行成功。

**这是新增证据扩展实验，不是同材料缓存对照。** A的原材料与补充包先在同一固定控制策略下比较；随后若要比较B0–B6缓存/上下文策略，所有策略共享相同已冻结的扩展包、问题、版本与预算。不得将新增SDK源码带来的答案变化或长短差异计成缓存策略质量收益。问题文本和真实前序历史可进入检索；review标签、参考答案和未来turn不可进入在线路径。三例probe只能检验机制，不能替代全dev回归和冻结test验收。
