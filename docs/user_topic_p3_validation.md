# 用户主题检索的 P3 闭环开发审查

`correction_aware` 和 `user_topic` 两组均完成 Cloud dev 的 5 个会话、41 轮，共 82 轮成功，无 unfinished 或超时。用户主题检索在套餐与 SDK 等续问中取回了更合适的产品来源，但回答仍出现产品、平台、参数和版本范围混用。**本轮没有人工质量评分、质量等价结论或自动准入**。

原始输入、回答、可见文档、实际历史、事件及 manifest 位于 `runs/remote-h200-user-topic-p3/runs/p3-user-topic-v1/`。主分析在 `runs/user-topic-p3-review/`；本审查另用 `independent.py` 从原始文件重算 `independent.json`，保留文件 SHA、每 task 配对、设置差异和不执行代码的语法诊断。已扫描全部 82 份回答；下列实质问题逐条对照当前 `audit_evidence` 及实际历史，不是完整的逐 claim 金标。没有读取 test qrels，也没有用当前外部 IBM 知识替代这份固定语料的证据。

## 比较范围

两组使用同一 controller source SHA `18ad9c560c9b9f1c2db27567d6572ae17f75a5814da36cdb391133a82f427643`、模型指纹、Q0、P3/c4/seed42、输出上限 1024、`normal_rag` 和 `focus_refs`。配置除 state_dir 外相同；retriever resolved 配置除 query_mode 外相同。BM25s+BGE-M3+RRF 的候选与 rerank 预算均为 50，返回 top5，reranker 上限 1024。两份 workload 只差 retriever_config 文件路径。冷准备记录均显示 GPU/host unique 为 0；真实容量是 **121920/279040**，不能把 YAML 中 mock 用的 65536/131072 当作实际池大小。

这是先 correction_aware、后 user_topic 的单次有序闭环比较。匹配 41 个相同 task/query，但只有 11/41 的实际 input IDs 与历史相同、6/41 的 output IDs/答案相同。36/41 的可见文档序列相同；检索直接改变的 5 轮为 CocoaPods 2/3、Cloudant 7/9、WebSocket 4。即使当前文档相同，前轮生成历史及当前采样执行也可能不同；不能将其余回答变化全部归因于该轮检索。

| 描述量 | correction_aware | user_topic |
|---|---:|---:|
| 成功 / 计划轮数 | 41 / 41 | 41 / 41 |
| 等会话平均 pipeline 完成时间 s | 3.863 | 3.749 |
| 全轮平均 / P95 pipeline 完成时间 s | 3.818 / 5.896 | 3.717 / 5.389 |
| 平均检索时间 s | 0.810 | 0.755 |
| 平均 / P95 control 时间 ms | 78.92 / 102.52 | 81.25 / 109.72 |
| 输入 / 输出 tokens | 211192 / 21240 | 213063 / 20970 |
| 原生 actual prefill / host loaded tokens | 161656 / 0 | 161735 / 0 |
| 截断轮数 | 1 | 1 |

Pipeline 完成时间定义为 `max(t_response_done, t_commit) - benchmark arrival`，包含检索。本批 82/82 轮均有 `t_commit >= t_response_done`，所以先前仅用 commit 的独立计算与该定义数值相同；分析脚本已统一使用 max。两组截断均为 `1c041ce47a81941c26899fdf08bde961<::>8`，1024 输出预算下回答末尾未完，不能把 success 理解成内容完整。全部动作为 REBUILD（5 initial、36 normal_rag）。数据不证明可推广的时延改进、质量改善或显著性。

## 五个检索变化点及其实际回答

以下任务简称使用完整 conversation ID 的前缀，后面的数字是轮号。

| 任务 | 可复核来源与回答变化 | 尚存问题 |
|---|---|---|
| `d5b1e735a040853ed361a3dfde1b8ef0<::>7`，Cloudant 套餐 | correction_aware 混入 Event Streams `ibmcld_01183-7-2154`，回答把三可用区架构借给 Cloudant，并说 Standard 无存储上限。user_topic 换为 Cloudant `ibmcld_00612-7-2163`，正确引用 Standard 最小 100 reads / 50 writes / 5 queries；Lite 的 20/10/5、1GB 和次日复查来自 `ibmcld_00558-1499-3456`、`ibmcld_12904-1535-3460`。 | 更合适的当前取材可直接确认；不因此认证整个套餐回答，也不将固定语料中的价格/版本当作当前产品建议。 |
| `d5…<::>9`，Python SDK | correction_aware 用 Discovery `ibmcld_07128-4280-5113` 的 namespace/API 链接拼出未经当前来源支持的 `watson-developer-cloud/cloudant-sdk-python`，并附未闭合 f-string。user_topic 取回 `ibmcld_00462-1307-2903` 和 `ibmcld_00462-7-1647`，安装命令 `ibmcloudant>=0.0.27` 有直接依据。 | user_topic 又扩写完整 CRUD Python 示例；可见资料只给安装/库链接，`ibmcld_00646-3898-5632` 的操作例属于 Node.js。其 Python 语法能解析，不表示 API 参数、更新 revision 或行为已验证。原问“在哪里找 SDK”不需要用未证实代码填满回答。 |
| `b4614daadceeecb400f7baf0aa48dcb8<::>4`，customization ID | 原组把 `ibmcld_13652-4903-6514`、`…6121-7661` 的**示例** GUID 当作这里的确定 ID。user_topic 改为 Speech-to-Text 的连接参数材料：`ibmcld_13263-1646-2958`、`13361-1589-2935`、`13384-1568-2859` 等。 | user_topic 将 language 与 acoustic ID 都解释为 custom acoustic model。声称 `ibmcld_13322-7081-7811` 给出一段 WebSocket 代码，但原文是 HTTP cURL 与传参时机说明；不能把自行改写的示例称作来源原例。 |
| `c01c8cf11437e6bb3bc93efac26528c2<::>2`，CocoaPods issues | 替换一个非关键片段，双方仍取回 Functions 的警告及安装资料 `ibmcld_10817-1342-3184`、`…7-1802`。警告本身已被找到。 | user_topic 继续把 App ID 的 `IBMCloudAppID`、keychain sharing（`ibmcld_02772-4213-5899`）并入 Functions 安装建议；“文中列了配置步骤”不等于这些产品共用同一要求。 |
| `c01…<::>3`，明确询问 warning | 两组都知道 `SWIFT_VERSION` 警告。来源 `ibmcld_10817-1342-3184` 明确是 Pods 中的 IBM Cloud Functions target。 | 两组均改成 `IBMCloudAppID` target；user_topic 另外加入 **Tools > Android > Sync Project with Gradle Files**，来自 `ibmcld_02772-4213-5899` 前半 Android 段。对 iOS CocoaPods 的解决步骤形成新的跨平台拼接。 |

## 全回答扫描中的其它实质发现

下表区分“当前来源能支持的局部变化”和“仍需修正的内容”，不将行数或问题数当作方法质量胜率。对没有在可见片段中找到的步骤，只称未获当前证据支持，不推断整个外部官方文档也没有它。

| 任务 / 范围 | 对照事实与诊断 |
|---|---|
| `d5…` 2、3、4，两组共有 | 第 2 轮把 SQL Query 的 newline-delimited JSON 与 TEXT 输入的 `regexp_extract/get_json_object`（`ibmcld_13498-32548-34436`）当作 Cloudant 文档查询说明；第 3 轮来源 `ibmcld_00556-7-1696` 的 attachment PUT 路径是 `/$DATABASE/$DOCUMENT_ID/$ATTACHMENT`，回答换成 `_attachments` endpoint，且标为 JSON 的示例含注释；第 4 轮把 Discovery 的 50/10/32MB uploaded/crawled document 限制（`ibmcld_06968-16615-18789`）归给 Cloudant。相同错误不能算作新 query mode 的改善。 |
| `d5…` 5，两组共有 | 明确澄清 attachments 后，双方都回答 10MB；`ibmcld_00558-23465-25360` / `ibmcld_12904-23555-25450` 的表明确为 Individual Document 1MB、Single Attachment 10MB、Request Body 11MB。该轮两组 input/output 完全相同，且都使用 focus_refs，不能归因为 user_topic 独有收益。 |
| `d5…` 8 | 原组把 Db2 Lite 的 45 天 reactivation / 60 天删除（`ibmcld_01025-7-2061`、`ibmcld_07578-494591-496583`）套到 Cloudant。user_topic 不再带入这段限制，但“Cloudant Lite 可无限期使用”的绝对断言未由当前 Cloudant 片段明确给出；两组本轮文档相同。 |
| `c01…` 4、5 | user_topic 保留 `ibmcld_10817-1342-3184` 的 OpenWhisk Cartfile、carthage update 和 framework 流程；未重复原组额外的 Cartfile post-checkout/sed 结构。第 5 轮原组在 OpenWhisk 导入之后接上 Event Notifications 的 ENPush 初始化/注册（`ibmcld_07551-14062-16080`、`…15747-17355`），user_topic 去掉这段跨 SDK 拼接。两组当前文档相同，这些只是闭环输出的局部差异。 |
| `c01…` 6、7 | 两组承认当前材料没有 Carthage 最新版本，却继续给 Homebrew 安装等未获当前来源支持的步骤；原组额外猜测 0.41.0，user_topic 没有该猜测。第 7 轮“去 Google”取回的五段均无 Carthage 指南，仍延续过去助手的做法；历史有说过不等于外部事实已获证实。 |
| `c6c3b02ca32795af64c903dd76700517` 3、5 | 第 3 轮同时可见旧 release note 的“不必做任何事”与更新通知，答案没有解释时间适用条件。第 5 轮 user_topic 把 v3 end of support 定为 2023-06-19，而 `ibmcld_01442-1679-3832` 明确当时 **尚无 end-of-support 日期**；弃用默认版本与结束支持不是同一事件。 |
| `c6…` 6 | `ibmcld_01412-7637-8525` 支持 Key Protect wrap/unwrap 和 Image Key Synchronizer 链接。user_topic 改用这些步骤，比原组的 cryptsetup/GPG 说法更贴近证据，但又补出可见材料没有给出的 add-on 安装命令，且没有给出真正加密镜像的完整步骤。原组示例将 `docker save` 输出经 GPG 写到文件后，却仍 tag/push 原镜像；生成该文件本身不证明推送的镜像已加密。 |
| `c6…` 8、9 | user_topic 将 `actions:container-registry/...` 当作 `--roles` 值；原组也编出 `container-registry-reader/editor/manager`。`ibmcld_01477-5499-7319` 明确配置 namespace 需要 account-level Manager 和 resource-group Viewer，查看需要 Reader/Manager；`ibmcld_01377-13470-15034` 则区分 action 与 role。步骤不能只保留相似名称而替换这些类型及作用域。 |
| `b461…` 2、3、5、6 | 两组给 WebSocket 操作示例时把 Assistant phone integration 的 configure envelope 与 STT/TTS 接口混用。user_topic 第 5/6 轮额外构造 `/v1/phone` WebSocket URL，并把 SIP username/password 放入 query；当前 `ibmcld_16288-1733-3996` / `03158-8929-11062` 仅支持 UI 的 secure trunking、SIP digest/TLS 配置，未给这套 HTTP/WebSocket API。相同概念词不构成端点及参数的证据。 |
| `b461…` 7、8 | user_topic 第 7 轮补充 Enterprise 1000 concurrent calls，有 `ibmcld_16287-7751-9832` 表格支持，并明确是并发通话而非电话号码个数。第 8 轮两组都有对应 SIP provider / failure settings 来源，但不能把服务商专属操作或示例配置默认为所有 provider 通用。 |
| `1c041ce47a81941c26899fdf08bde961` 1–4 | 首轮两组将 Spark/COS SELECT 配置与 Cloudant `_find` 连成一个排障流程。第 3 轮取材已变为 Discovery，同一建议却被 user_topic 总结成 IBM Cloud SQL Query。第 4 轮 user_topic 生成 `"dql": "SELECT * FROM documents..."`，而 `ibmcld_07045-4959-6113`、`07094-7-1956` 等是 Discovery Query API 参数；当前来源没有支持该 SQL-shaped payload。 |
| `1c…` 5–8 | 第 5 轮在未明确存储服务的情况下直接选 VMware NFS 操作，不能将它当作对之前数据库存储问题的通用解法。第 6 轮 user_topic 去掉原组“用 Discovery JSON 查询 Netezza Parquet”的例子，保留当前 `ibmcld_16728-*` 的 Netezza 说明。第 7 轮按所取 Watsonx.data 文档回答，但仍需确认用户所指产品。第 8 轮两组都截断，并给出当前来源未支持的 `_find` group-by 结构和不能解析的 cURL JSON；`ibmcld_00623-7-1781` 支持的是声明式 JSON 查询/建索引，不能因此推出任意聚合接口。 |

## 对下一步实现的约束

检索阶段应继续改善主题锚定与候选覆盖；生成阶段需要独立保持**产品 / 版本 / 平台 / API 参数类型**的绑定。本文中的失败不支持用更多相关段简单堆叠替代这种绑定：CocoaPods 同一段同时包含 Android 和 iOS，Cloudant SDK 的正确库链接也未阻止模型自行扩写 CRUD。

可执行的窄改进是：保留 passage 中的局部标题/代码对象范围；回答“在哪里找”时优先返回证据给出的链接与安装命令，不能把完整例程当作必须补齐的内容；对发生在当前回答中的代码做独立语法/API 证据诊断；将历史用于消歧而不是默认当作外部事实。后续再用固定历史的输入对照、同预算重复及冻结质量协议区分 retrieval、history propagation 与 generation 的贡献。这里没有据个别案例手写产品答案或调整质量门槛。

此次两组各 405 条事件均无 `quote_observed`，所以无法顺带验证普通 `ignore_eos=False` 的新 native ratio 预算公式；独立结果明确记为未观测。新观测补丁的功能结论仍限于 [受控 probe](native_observation_validation.md)，不会借 P3 成功生成补齐缺失的调度覆盖。
