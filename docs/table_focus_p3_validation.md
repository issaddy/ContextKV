# 焦点行与行内来源：三组完整 P3 开发回归

2026-10-24，`off → focus_refs → all_rows` 各完成 Cloud dev 的五个会话、41 轮，共 **123/123 成功**。实际输入、完整 QA 链、证据正文、receipt 与输出记录均通过核对。焦点版在附件纠正轮正确保留单附件 10 MB 与对应原始引用；但后续仍出现条件标题矛盾、跨产品来源误用和 SDK 外推，**不能认定完整会话质量通过或总体非劣**。这是三次有序开发运行，不是随机重复的因果性能实验。

原始 27 文件经导入方 SHA 核验后保存在 `runs/remote-h200-table-focus/runs/p3-table-focus-v2/`。只读分析入口为 `runs/table-focus-p3-review/analyze.py`，派生 `summary.json`、`turns.jsonl`、`pairs.jsonl` 保留完整计数、答案、实际历史、引用映射和失败字段。本文语义观察来自 assistant 对目标及 d5 会话原始材料的逐项核查，不是人工评分或模型 judge 分数；123 轮自动合同核验不等于 123 轮全部语义评审。

```sh
.venv/bin/python runs/table-focus-p3-review/analyze.py
```

## 运行条件与输入核验

三组使用相同 Qwen2.5-7B-Instruct 固定 revision `a09a35458c702b33eeacc393d103063234e8bc28`、Q0 system prompt、32K 上下文、1024 输出上限、`normal_rag`/REBUILD、`stable_key`、c4、seed 42。检索为已冻结的 `correction_aware` + BGE 混合检索/重排配置；只含用户 query 视图，未把参考答案送入在线路径。线程设置为 RAYON 4、OMP/MKL/OpenBLAS 1，`TOKENIZERS_PARALLELISM=false`。

运行方记录工程源码基于 `b2c452f`，三份 manifest 的控制器 source SHA 均为 `05d9d30c8191753776242b8b20c4fcd43c407117befffe8d3bb358feb7f4f4c8`；experiment 中八个 A 集成文件 SHA 与本地对应文件逐项相符。三组 engine 仅 restart ID 不同，cache config hash 一致。检索 manifest 的唯一差异是初始化耗时，模型/索引/query 视图参数相同，实际每个 task 的检索顺序、分数、visible refs 和完整正文也均相同。

资源为 H200 上的 Qwen 大池，服务实际报告 **GPU 121920 / host 279040 tokens**、page size 64、write-through；配置文件中保留的 mock 容量值不替代此服务报告。三个 arm 各开始前 cold reset，GPU/host 已用量均为 0、活动与排队请求均为 0，cold 后 restart ID 与该组 manifest 对应。它们是**每组起点冷态**，不是每轮都冷置。运行方记录 judge 服务已经停止；此信息不由 Qwen 的空池遥测反推。该资源设置不能与 16K B/C 矩阵混比。

| 自动核对 | 结果 |
|---|---:|
| 保存 input IDs 由完整消息精确重建 | 123/123 |
| 完整原文 body spans、visible refs 与 body 覆盖 | 123/123 |
| 实际历史等于本组此前全部成功 QA | 123/123 |
| receipt input hash、答案与 output IDs 对齐 | 123/123 |
| 每组成功 / 失败 / 缺 audit | 41 / 0 / 0 |
| 任意两组 query、检索序列与材料相同 | 41/41 |
| 缺失或未完成会话 | 0 |

只在 `d5b1e735a040853ed361a3dfde1b8ef0<::>5` 解析到表格；focus 和 all_rows 各追加一次当前轮辅助段。其余 40 轮没有当前表格注释，但因完整实际历史继续传递，后续输入仍可能不同。三组均有两轮输出截断，未删除这些成功但截断的样本：off/all_rows 为 `1c04…` 的第 4、8 轮；focus 为 `1c04…` 第 4 轮和 `b461…` 第 8 轮。

## 目标轮：改善何在，哪些问题仍存在

当前用户是 “Not documents, I was asking for attachments?”。完整可见材料中的两份表来自 `ibmcld_00558-23465-25360`（`E_53079fa73a`）与 `ibmcld_12904-23555-25450`（`E_fe06da00dd`），分别明确列出 Individual Document Size 1 MB、Single Attachment Size 10 MB、Request Body Size 11 MB。两段正文相同，是两个来源标识，不是两份独立佐证。

| arm | 实际输入 / 输出 tokens | 核对原文后的观察 |
|---|---:|---|
| off | 4756 / 419 | 单附件 10 MB 本身正确，但又写 individual document “including all attachments” 为 1 MB，添加了原表未给出的包含关系；无规范 Evidence 引用。还把 PUT 附件地址笼统写成 `_attachments` endpoint，而可见附件原文给的是带附件名的具体路径，不能把这部分当成完整有据步骤 |
| focus_refs | 4975 / 101 | 回答单附件最大 10 MB，并引用上述两个真正支持该行的原始 ID。没有新增 1 MB 合计、请求体求和、性能保证或多余代码示例。针对本轮上限纠正，未观察到必须回答的信息因缩短而丢失；该判断不扩展为其他问题完整性结论 |
| all_rows | 5173 / 375 | 去掉已知 1 MB 包含错绑，并保留 10 MB，但没有 Evidence 引用。新增 JSON 示例含 `//` 注释，严格 JSON 解析失败；`application/pdf`/`.pdf` 与所给 Base64 的 PNG 签名不符，解码仅 70 bytes，却声称该示例为 10 MB。不是只有格式美观差异 |

all_rows 的问题由答案原文即可复核：保留原字符串，不执行代码；其 Base64 开头解码为十六进制 `89504e470d0a1a0a`。合法限额值并不能使同一回答的示例和额外说明自动得到支持。focus 的正面判断也不是只数合法 ID，而是核对其具体对象、最大值、引用所指的原始表行。

off 与 all_rows 在该会话第 1–4 轮的输入、输出和目标前历史完全一致；目标输入差 417 tokens 正是全行注释。focus 在尚未触发表格的第 2 轮已有生成差异，此后完整历史发生分叉，因此实际目标输入的 `4975−4756=219` **不是注释长度**。同一历史反事实渲染中 focus 仅增加 110 tokens；这次 focus 的目标不能当作与 off 同历史的纯干预。此前[冻结单轮正反顺序实验](table_focus_validation.md)提供的是另一种更局部的证据，两者分开解释。

## 目标之后：完整对话仍有实质缺口

以下检查覆盖三组 d5 第 6–9 轮全部答案及其当前完整可见正文；没有调用外部 SDK 或判定真实世界接口版本。第 5 轮以后没有再触发表格 helper。

| 轮次 | 三组共同材料与答案观察 | 判断边界 |
|---|---|---|
| 6：如何开始使用 Cloudant | 三组均围绕搜索服务、创建实例/凭据、打开 dashboard、创建数据库与文档作答。`ibmcld_00547-7-1654`、`00580-19349-21448`、`00616-7-1788` 提供相应操作依据；all_rows 添加的 Dickens 图书 JSON 来自 `00522-7-1725` | 没有观察到由目标附件值带来的新的核心对象错绑；不声称所有 UI 标签或 API 细节已实际操作验证。三组均缺规范 Evidence ID 引用 |
| 7：Lite 与 Standard 的差异 | 三组正确保留 Lite 20 reads/s、10 writes/s、5 global queries/s、1 GB，以及超限升级/删除后的处理。它们对 Standard 存储给出当前段落未直接证明的“无限/无固定上限”表述（off 又加了 up to the plan's storage capacity 限定）；focus 还写 Standard 无写限制。focus/all_rows 的三可用区描述在当前材料中来自 **Event Streams** 的 `ibmcld_01183-7-2154`，不能直接当成 Cloudant 所有部署的证据 | 可见 Cloudant 段落支持配置吞吐和按用量定价，以及解除 Lite 的特定存储配额写禁令；不等同于无限资源或所有写限制均不存在。产品、部署范围与解除条件仍需分别核查 |
| 8：探索时是否选 Lite | 三组均建议 Lite，保留主要 1 GB 与吞吐约束。focus 的第四项标题却是 **“No Write Limitation”**，正文又说明超额 402 与恢复写入条件；off/all_rows 没有这个标题矛盾 | 这是应保留的条件回归，不因 focus 更短便判更好。`ibmcld_00558-1499-3456` 与 `12904-1535-3460` 明确给出写限制。focus 正文仍写出了条件，故不是整段全部无据 |
| 9：Cloudant Python SDK | off/all_rows 将 Continuous Delivery Python SDK、Discovery 文档链接冠名为 Cloudant。focus 改用 Watson SDK，并构造 `from watson_developer_cloud import CloudantV1` 示例；当前五段材料未提供该 Cloudant 类与操作依据 | `ibmcld_00690-1043-2339` 明确讨论 toolchains/Tekton；`07128-4280-5113` 明确讨论 Discovery，其余是通用 SDK handbook。三组均未解决产品适用性，不能把“链接存在于材料”当成“支持 Cloudant SDK”。不凭此离线检查断言所有外部链接失效或执行结果 |

第 2、4 轮还保留 SQL Query / Discovery 资料被归给 Cloudant 的原有问题，它们发生在表格触发前。第 3 轮已存在示例错误；all_rows 第 5 轮又延续了 PDF/PNG 示例。后续变化包含历史传播，也包含未触发时已观察到的输出波动，不能全部归因于表格功能。

## 非触发轮与历史传播

| 配对 | 相同实际历史 / 相同输入 / 相同输出（41 轮） | 相同输入但输出不同 |
|---|---:|---:|
| off / focus | 8 / 8 / 3 | 5 |
| off / all_rows | 11 / 10 / 6 | 4 |
| focus / all_rows | 8 / 8 / 5 | 5 |

32 轮来自完全未触发表格的四个会话。off/focus 与 off/all_rows 各有 26 轮历史/输入不同、30 轮输出不同；这些会话不存在表格注释的直接干预。d5 第 6–9 轮在所有配对中历史、输入和输出均不同；其中 off/all_rows 在触发前完全一致，而 focus 更早已分叉。配对脚本保留逐 task 分类，不把“同问题”误写为“同输入”。引用 parser 使用 `evidence-id-groups-v2`：规范已知引用出现轮数为 2/3/7，三组均无未知 ID；裸 `Evidence ID: E_…` 不计入规范引用。该计数不是事实支持或质量分。

## 完整时间与逐会话结果

完整响应时间为 `max(t_response_done,t_commit) − benchmark.t_arrive`，起点在检索前，已含检索和 controller；control 分项不再相加。它不含每个 CLI 进程开始时的模型/索引初始化。初始化约为 13.63/14.03/14.04 秒，另有进程级 elapsed 74.58/75.14/76.73 秒，不能与逐轮响应均值混用。

| 描述性指标 | off | focus_refs | all_rows |
|---|---:|---:|---:|
| 完整响应 mean / P95，秒 | 3.771 / 5.725 | 3.698 / 5.530 | 3.693 / 5.253 |
| 会话等权响应均值，秒 | 3.792 | 3.726 | 3.730 |
| 检索均值，秒 | 0.788 | 0.843 | 0.725 |
| controller mean / P95，ms | 79.67 / 112.20 | 79.19 / 102.86 | 79.36 / 107.98 |
| pipeline TTFT 均值，秒 | 0.954 | 1.004 | 0.884 |
| 总输入 / 总输出 tokens | 211474 / 20964 | 209051 / 20245 | 211343 / 20857 |
| 目标轮完整响应，秒 | 2.795 | 1.135 | 2.840 |

| 会话前缀（完整 ID 在 summary） | off 完成秒 | focus 完成秒 | all_rows 完成秒 |
|---|---:|---:|---:|
| 1c04 | 31.148 | 31.287 | 31.814 |
| b461 | 36.608 | 36.190 | 32.602 |
| c01c | 26.621 | 26.789 | 29.568 |
| c6c3 | 30.861 | 30.880 | 28.407 |
| d5b1 | 29.740 | 26.831 | 29.378 |

focus 的目标答复更短，目标及 d5 会话时间同时较低；但不同历史、其他轮生成差异、检索耗时及三组顺序都存在，不能从这一次运行计算孤立功能的因果加速或统计显著性。all_rows 的总均值也略低于 focus，但目标示例更差，这再次说明不能按时长选择质量方案。三组的截断均保留在时间与 token 统计中。

## 下一步由真实缺口驱动

当前结果支持继续研究可选 focus 的局部对象/引用改进，同时保留 all_rows/off 对照；不据此自动提升为默认或通过质量门。优先补条件路径、产品适用性与来源绑定，继续区分范围修复、引用支持、完整性及新错误，而不是追加一串通用提示约束。

SDK 轮提供一个具体可复用入口：**第 6 轮用户已明确说 IBM Cloudant，第 9 轮只说 Python SDK**，当前 `correction_aware` 在无纠正词时回到当前问题，丢失产品锚。可比较 current/现有纠正视图，与仅从允许的用户历史保留实体/主题的补全或 [Dawn 式意图视图](retrieval_research_next.md)；先在相同语料、BGE、候选预算和开发问题下比较检索材料，再做完整生成回归。历史助手可以保留在 QA 中，但其错误不能成为补全的外部事实来源；不读取未来轮或 targets。该扩展尚未修改在线 query，也没有用本轮答案反向改写已保存实验。
