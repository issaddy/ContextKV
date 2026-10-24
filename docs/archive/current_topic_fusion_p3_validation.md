# Current/topic 融合的 P3 闭环开发验证

> 归档记录：保留当时的设计、结果与复用依据；其中“下一步”和执行计划不代表当前任务。当前工作仅以[实现清单](../implementation_plan.md)为准。

本轮 `single` 与 `current_topic_rrf` 两臂均完成 Cloud dev 的 5 会话、41 轮，共 82/82 成功，无 unfinished 或超时。融合后的 customization ID 定义更贴近新取回的来源；附件和 SDK 两轮没有直接照抄新增的跨产品/复制材料。不过，回答仍有未经来源支持的 API 示例，后续还出现版本、容量与产品范围混用。**本报告不提供人工分、自动质量胜率或质量准入。**

原始文件位于 `runs/remote-h200-fusion-p3/runs/p3-current-topic-fusion-v1/`。主分析与独立复算保存在 `runs/current-topic-fusion-p3-review/`；`independent.py` 从原始 benchmark、audit、manifest、cold 和 events 生成 `independent.json`，未执行模型生成的代码。全部 82 份回答已扫描，重点及新增问题对照实际可见片段和历史核查；这仍是 agent 开发诊断，不是完整的逐 claim 人工金标，也没有读取 test qrels。

## 设置、预算及输入完整性

两臂的 controller 配置除 state_dir 外完全相同，retriever resolved 配置只差 `query_fusion`，共同使用 `user_topic`、Q0、focus_refs、normal_rag、P3/c4/seed42、32768 context / 1024 output。workload 除 retriever 文件路径外相同。实验所记 11 个组件/driver SHA 与本地副本一致；参考实现 commit 为 `681b998100acc38a13e9ec8e427e6b531697af81`，controller source SHA 两臂均为 `18ad9c560c9b9f1c2db27567d6572ae17f75a5814da36cdb391133a82f427643`。

生成模型为固定 Qwen2.5-7B-Instruct revision，检索仍用 BGE-M3 `5617a9f…` 与 reranker `953dc6f…`，BF16、batch16，encoder/reranker token 上限为 8192/1024。模型身份沿用 operator-supplied 本地 snapshot 元数据，不声称本轮重新证明了全部权重字节。BM25/dense 每查询各取 50，RRF 常数 60，最终 rerank 总预算固定 50、top5。single 共 41 次查询编码，融合共 45 次；两臂各 rerank 2050 对。只有 4 轮 current 与 topic 文本不同，融合多做一路检索，**不能称为总检索算量相同**。

两次 cold 记录均为 GPU/host unique=0，实际池为 121920/279040 tokens、page64；两臂只做 KV reset，没有重启 Qwen 服务或执行生成 warmup。顺序固定 single→fusion，尚未随机化或重复交叉。每臂 controller/retriever 子进程各自加载检索模型；引擎进程的 kernel/allocator 状态不因 KV reset 重置。这个顺序和资源边界限制了性能归因。

独立机械核查确认：82 轮 audit answer 与 benchmark 一致，输入/输出 token 数相符，可见正文 key 顺序与 manifest 一致；每一轮 `actual_history` 都恰好包含同臂前面所有真实 user/assistant 对，无漏轮或用另一臂答案替换。另一个独立检索绑定检查 `runs/current-topic-fusion-review/p3-retrieval-binding.json` 确认 82 个实际输入重建及 receipt 匹配，在线 trace/top5 正文与离线排名相符。

跨臂只有 9/41 实际 input IDs 与历史相同、5/41 output IDs/答案相同。38/41 可见文档序列相同，真正改变生成材料的轮次只有 `b461…4`、`d5…5`、`d5…9`。当前材料相同的其它轮仍可能因先前生成历史和执行条件不同而产生不同答案，不能把其内容差异全部归因于该轮融合。

## 时间与工作量

Pipeline 完成时间统一为 `max(t_response_done, t_commit) - benchmark arrival`，包含检索；82 轮均为 commit 不早于 response_done。

| 描述量 | single | current_topic_rrf |
|---|---:|---:|
| 成功 / 计划轮数 | 41 / 41 | 41 / 41 |
| 等会话平均 pipeline 完成时间 s | 3.835 | 3.594 |
| 全轮平均 / P95 完成时间 s | 3.795 / 5.443 | 3.580 / 4.949 |
| 平均 / P95 检索时间 s | 0.768 / 2.184 | 0.798 / 1.949 |
| 平均 / P95 control 时间 ms | 78.57 / 113.06 | 79.07 / 102.49 |
| 输入 / 输出 tokens | 213437 / 21467 | 209541 / 19459 |
| actual prefill / host loaded tokens | 162109 / 0 | 160837 / 0 |
| 截断轮数 | 1 | 1 |

| 会话 | single 平均完成 s | 融合平均完成 s |
|---|---:|---:|
| `1c041…` | 4.031 | 3.911 |
| `b461…` | 4.710 | 4.286 |
| `c01c8…` | 4.112 | 3.378 |
| `c6c3…` | 3.329 | 3.270 |
| `d5b1…` | 2.991 | 3.124 |

融合的本次总体完成均值较低，检索均值略高、输出总量少 2008 tokens；有一个会话完成均值变高。输入、输出和历史均已分叉，不能将端到端差额当作融合自身的因果加速。两臂都在 `1c041ce47a81941c26899fdf08bde961<::>8`（Cloudant Query，非 CocoaPods）达到 1024 tokens 并截断；success 仅表示请求成功完成协议，不代表答案内容完备。

## 三个材料变化点的逐源核对

**`b4614daadceeecb400f7baf0aa48dcb8<::>4`：定义改善，获取步骤仍超出可见材料。** single 将 `language_customization_id` 与 `acoustic_customization_id` 都说成用于声学适配。融合新取回 `ibmcld_13396-1754-3852`，其中直接说明 customization_id 是模型 GUID，融合回答改为 unique identifier，核心定义有明确依据。该段还解释 created/updated/language/dialect/owner/name/description/base_model/versions/status 字段，并非全部字段凭空编造。

但融合随后增加 `GET /v1/customizations` 列举获取 ID，以及含 `customizations` 顶层和 `versions:[{"version":"1.0"}]` 的完整响应示例。这些端点/结构没有出现在本轮五份可见片段或此前历史里；另一新增段 `ibmcld_13374-8726-10599` 给出的是 **POST `/v1/customizations/{id}/corpora/{name}` 添加语料**，不能拿来证明 GET/list。这里仅判为当前证据不足，不借当前外部 API 知识断言该端点现实不存在。回答明确将虚构 GUID 标为示例，没有把它当作用户实际 ID；两个 bash 块却保留了游离的 `IBM Cloud` 部署标签，复制执行前仍需清理。

**`d5b1e735a040853ed361a3dfde1b8ef0<::>5`：新增 PDF 材料未污染这次答案。** 融合将 `ibmcld_07224-7650-9752` 换为 `ibmcld_06968-15099-17180`，后者讨论 Discovery 对加密 PDF、单页图像及压缩包的处理。它不支持 Cloudant attachment 限制。但融合回答仍绑定 `ibmcld_00558-23465-25360` / `ibmcld_12904-23555-25450` 的 **Single Attachment Size = 10MB**，没有说 Cloudant 禁止密码 PDF，也没有借用 Discovery 限制。两臂都答对了这一个表格关系；这是当前局部观察，不能证明今后对所有跨产品干扰都稳健。

**`d5…<::>9`：链接实用，完整 SDK 调用仍未获支持。** 融合新取回 `ibmcld_00580-52091-54223` 的复制介绍，内容涉及 source→target、持续复制、checkpoint 和 PouchDB。回答没有把这些机制塞进“在哪里找 Python SDK”的回答，而是保留 `ibmcld_00462-1307-2903` / `…7-1647` 的 `ibmcloudant>=0.0.27` 安装命令，并新增它们给出的官方 GitHub 链接；API 链接也在 `ibmcld_00646-3898-5632` 中。它比只给安装/代码、更少直接定位库的 single 回答更贴合问题的一部分。

两臂仍扩写了 Python CRUD 调用；当前片段没有提供 `post_database/post_document/get_document` 的 Python 调用签名，相关 `00646` 示例属于 Node.js。融合去掉 single 继承来的 JPEG 标签/PNG Base64 示例，并更换了参数名，但不能仅凭代码看似合理或 Python 可解析认证 API 正确。完整 SDK 例程也不是回答“在哪里找”的必要条件。

## 全部回答扫描发现的其它问题与局部变化

以下轮次本轮 top5 相同。它们说明生成和历史传播问题仍在，不能据此统计融合的因果质量胜率。

| 范围 | 当前回答与来源的关系 |
|---|---|
| `d5…` 2–4 | 两臂继续把 SQL Query 的 newline-delimited JSON / TEXT 提取函数（`ibmcld_13498-32548-34436`）套给 Cloudant。第 3 轮把普通 `image` JSON 字段称作原生 attachment，而 `ibmcld_00556-7-1696` 明确使用 `_attachments`；示例还将 PNG Base64 标为 image/jpeg。第 4 轮继续将 Discovery 文档限制归给 Cloudant，并把 File Storage **达到最大 inode 数所需的 7.8TB 卷大小**说成单文件最大尺寸（`ibmcld_07578-1268470-1270517` / `16727-1271119-1273166`）。 |
| `d5…` 7、8 | 融合第 7 轮再次写 Standard “no fixed storage limit”，而 Cloudant 片段只支持按用量计费与扩容；第 8 轮把 `ibmcld_01025-7-2061` / `07578-494591-496583` 的 **Db2 Lite 每 45 天 reactivation** 套到 Cloudant。single 没带该 45 天规则，但其“无限期”断言也缺当前 Cloudant 明确依据。 |
| `c01c8cf11437e6bb3bc93efac26528c2` 2、3 | single 仍把 Functions 的 Swift warning 处理与 `IBMCloudAppID` Podfile 合并。融合第 3 轮不再指定 App ID target，保留设置 No / SWIFT_VERSION 的核心内容，但拓展到 Swift5 并不能由当前 Swift4 片段直接证明；第 2 轮甚至用“遇到 CocoaPods 问题就切换到 CocoaPods”作结论。相关来源仍为 `ibmcld_02772-4213-5899` 与 `10817-1342-3184` / `…7-1802`，须保留产品边界。 |
| `c01…` 4–7 | Fusion 第 5 轮的 Carthage/OpenWhisk 基本步骤有 `ibmcld_10817-1342-3184` 支持，但“保证不再遇到 Swift warning”过强。**第 6 轮新增明确问题：将 0.38.0 下载包、tar 解压和 `sudo make install` 作为 latest 安装步骤**；五份当前来源及此前历史都没有 0.38.0。single 承认没有最新版本，不给这一版本号。两臂仍有“Carthage 是 CocoaPods 的依赖管理器”等不当关系，以及缺证据时继续给安装长答的倾向。 |
| `b461…` 2、3、5、6 | single 把 phone integration 参数与 Assistant `/v1/recognize`、STT/TTS 内容拼成 WebSocket 示例。融合第 5/6 轮去掉这些拼装代码，UI 的 SRTP、SIP digest/TLS、搜索响应少于 5000 字符可由 `ibmcld_16288-1733-3996` / `16288-6287-8401` 等支持。第 5 轮总结却把来源的“可选择一个或两个安全选项”改成两者都必须启用，并作无数据泄露保证；第 2/3 轮仍未清楚区分 STT/TTS 与 Assistant configure envelope。 |
| `c6c3b02ca32795af64c903dd76700517` 3、5–9 | 两臂不再像此前某些回答那样把 v3 end-of-support 固定为 June19，但“继续v3不用做事”仍未处理旧 release note 与迁移通知的时点。加密镜像回答仍不能用 wrap key 代替完整镜像加密流程。single 又生成 `namespace-create/image-push` 等当前片段无据命令；fusion 第 9 轮正确保留了 `ibmcld_01477-5499-7319` 的 **account-level Manager + resource-group Viewer** 和查看需 Reader/Manager，属于可确认的局部改进；第 8 轮仍混用平台与服务 role 示例并输出空 action 表。 |
| `1c041…` 1–8 | 仍在 SQL/Data Engine/Discovery/Watsonx.data 等产品之间拼接。第 2/3 轮将 Discovery DQL/NLQ 功能归给 SQL Query；第 4 轮两臂都给 SQL-shaped DQL，融合还添加当前源未给出的 `facets` / `aggregations` payload；第 5 轮把 VMware NFS 操作和 Knowledge Studio model 清理并列作为未指定存储的解法。第 8 轮两臂都截断，且把来源中“按年份统计书籍”的用例自行实现成未获当前材料支持的 `_find` group 语法。 |

自动代码语法检查仅是辅证：例如 JSON fence 中可能包含 HTTP 报文头或合法的字段片段，解析失败不能一律当作语义错误或总分扣分。相反，语法成功也不能证明产品对象、参数名或调用行为正确。这里没有执行任何生成命令。

## 历史参照及结论边界

此前 [user_topic P3](user_topic_p3_validation.md) 的单路等会话均值为 3.749s，本次 single 为 3.835s；两批单路 41/41 的可见文档相同，但只有 8/41 输入和 4/41 输出相同。这只说明历史传播与运行变异实际存在，不能把旧单路并入本次当作额外随机重复，也不能用旧质量判断代替新回答审查。

本轮建立了完整运行、相同 rerank/生成预算设置、检索绑定与闭环历史的工程证据；在三个直接材料变化点看到有用定义/链接和未发生的特定干扰，同时保留新增无据步骤及其它回答中的回归。整体质量仍未确定，默认 `query_fusion` 保持 `single`。后续应将检索覆盖、来源范围绑定、稳定 API/代码与版本依据、回答质量分别验证，而不是继续增加材料便默认认为质量等价。本轮按计划结束，没有追加新的 GPU 实验。
