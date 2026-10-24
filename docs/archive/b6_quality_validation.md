# B6 phase 1：完整开发质量评价与来源复核

> 归档记录：保留当时的设计、结果与复用依据；其中“下一步”和执行计划不代表当前任务。当前工作仅以[实现清单](../implementation_plan.md)为准。

**533 个答案的 2665 维评价全部完成并可解析；12 个候选均未通过预声明的五维联合非劣门槛，未产出 B6。** 独立审计 `issues=[]`，没有因格式或缺分阻止本次比较。60 项候选—B0 的逐维比较中，7 项为 `noninferior`、53 项为 `inconclusive`，没有 `inferior`。不确定不等于已经证明劣于 B0，也不等于非劣成立。

## 完整性与协议

质量原始文件来自一次完整的 [22 文件导出](../../runs/remote-h200-b6-quality/runs/b6-quality-dev-phase1-v1)，归档 SHA256 `ea23bb8301a811e322aa3abafdd557f903cb9355637d5f516c753658be487550`。生成仍是 [13 格 P2 实验](b6_joint_validation.md) 的五个开发会话、每格 41 轮、真实自有历史、完整可见正文；没有选取赢家或替换某个答案。

独立分析器 [analyze.py](../../runs/b6-quality-review/analyze.py) 复用原 strict parser、acceptance reader 与 session-cluster bootstrap，结果保存在 [summary.json](../../runs/b6-quality-review/h200-v1/summary.json)、[逐维原始记录](../../runs/b6-quality-review/h200-v1/slots.jsonl)、[全答案与来源](../../runs/b6-quality-review/h200-v1/answer-review.jsonl)、[重算质量报告](../../runs/b6-quality-review/h200-v1/quality-recomputed.json)。14 项 CPU 合同验证覆盖缺分、截断、响应字段错配、重复 ID、错误 backend 和改动门槛；测试评分仅在临时目录生成。

- 固定源码 `08931340c928687e8c5745c99734a869301ffebc` 的 14 个评分依赖、原 generation / audit / manifest / server record、准备程序、config、protocol、CPU preflight 的身份全部匹配。原始文件及其中路径不改写，acceptance 重算只做内存中的路径映射。
- 模型为 `AtlaAI/Selene-1-Mini-Llama-3.1-8B`，revision `427792f1c3e2073cb7da216924fd884b1ba496e0`，作者原生 `selene_absolute_v1` 模板、固定五维 rubric、无 reference targets、temperature 0、max output 1536、context 32768；生成器仍为 Qwen。模型独立性是声明的模型家族身份，不代表评价能力已获验证。
- 2665 条 `requested_regex` 与 `native_regex_v1` 一致，实际服务器 `grammar_backend=xgrammar`；固定 stock SGLang `3f36fc…`。逐条 raw HTTP content / usage / model / response ID / finish 与保存 grade 一致，全部 HTTP 200、`stop`、`scored`，响应 ID 唯一，无 NULL、长度失败或解析失败。没有重试或根据理由猜补分数。
- 实际 usage 的 prompt tokens 全部等于原 CPU 预检：输入范围 6236–14381，预留输出 1536 后均在窗口内；总输入 25309627 tokens，总输出 377598，单维最大输出 214。这里独立匹配了已绑定的预检和服务器 usage，没有在本地重新下载 tokenizer 或声称重新分词。
- 冻结 renderer 重建的 2665 个 messages 保留完整 history / visible evidence / answer。没有捕获完整请求 wire payload，不能把重建描述成独立抓包证明；原 HTTP 响应则全部保存并核对。

旧 [615 维无约束运行](dev_quality_validation.md) 的 6 个缺 Result 仍原样保留。本次新协议在新完整 cohort 上没有格式失败，说明该工程问题在本批次被消除；这不是同输入随机对照，不能由两个批次直接估计 regex 的因果改善率，更不能推定格式约束解决了事实或引用判断。

## 分数分布和非劣结果

下表为原生 1–5 分经 `(score−1)/4` 归一化后的逐维均值，每项分母均为 41。WT / SEL / WB 含义与生成报告一致。没有跨维平均成“总质量分”。

| 配置 | 正确性 | 依据充分性 | 可回答性 | 条件/版本 | 引用 | 生成 length / 41 |
| --- | --- | --- | --- | --- | --- | --- |
| Rebuild / WT | 0.915 | 0.884 | 0.963 | 0.902 | 0.689 | 1 |
| Rebuild / SEL | 0.927 | 0.933 | 0.957 | 0.909 | 0.677 | 1 |
| Rebuild / WB | 0.896 | 0.884 | 0.951 | 0.909 | 0.665 | 1 |
| Fixed4096 / WT | 0.933 | 0.902 | 0.951 | 0.884 | 0.689 | 0 |
| Fixed4096 / SEL | 0.927 | 0.927 | 0.970 | 0.902 | 0.646 | 0 |
| Fixed4096 / WB | 0.921 | 0.909 | 0.957 | 0.896 | 0.665 | 2 |
| Redundancy40 / WT | 0.909 | 0.909 | 0.939 | 0.890 | 0.671 | 1 |
| Redundancy40 / SEL | 0.921 | 0.927 | 0.963 | 0.921 | 0.695 | 1 |
| Redundancy40 / WB | 0.927 | 0.915 | 0.957 | 0.896 | 0.707 | 1 |
| B1 / WT | 0.921 | 0.909 | 0.945 | 0.896 | 0.677 | 0 |
| B1 / SEL | 0.915 | 0.896 | 0.957 | 0.933 | 0.683 | 3 |
| B1 / WB | 0.933 | 0.890 | 0.945 | 0.890 | 0.677 | 1 |
| B0 | 0.909 | 0.902 | 0.957 | 0.909 | 0.677 | 1 |

所有 2665 个原生分数的分布为：5 分 1503、4 分 965、3 分 196、2 分 1、1 分 0。这是评价器输出分布，不是正确答案比例。高分集中不说明任务已解决；下节的明确错绑同样得到满分。

主分析为配对 turn 均值差（候选减 B0），以会话聚类，5 个会话、5000 bootstrap、seed 42、95% 区间、预声明 epsilon=0.01。完整 60 个区间与会话等权敏感性结果均保留；后者也是 7 非劣 / 53 不确定，不替换主估计量。以下为通过的 7 个维度；每个候选至少还有一个维度不确定，所有引用与可回答性比较均不确定。

| 配置 | 维度 | 均值差 | 95% 会话簇区间 |
| --- | --- | --- | --- |
| Rebuild / SEL | groundedness | +0.0305 | [-0.0063, +0.0833] |
| Fixed4096 / WT | correctness | +0.0244 | [+0.0058, +0.0461] |
| Fixed4096 / SEL | correctness | +0.0183 | [+0.0058, +0.0321] |
| Redundancy40 / SEL | groundedness | +0.0244 | [+0.0058, +0.0449] |
| Redundancy40 / WB | correctness | +0.0183 | [+0.0063, +0.0291] |
| B1 / SEL | conditions_versions | +0.0244 | [+0.0122, +0.0321] |
| B1 / WB | correctness | +0.0244 | [+0.0000, +0.0526] |

例：Fixed4096 / WT 的条件维差为 −0.0244，区间 [−0.0417, −0.0063]；它既没证明差不超过 0.01，也没证明差超过 0.01，原结果保持不确定。Redundancy40 / WB 的引用均值最高，但区间 [−0.0682, +0.1486]，不能按点估计选胜者。12 个候选的 `acceptance_blockers` 均为空，失败原因是完整分数上的非劣判定，不是输入、协议或缺分故障。

远端在原 `0893134` 执行的 [freeze-outcome.json](../../runs/remote-h200-b6-quality/runs/b6-quality-dev-phase1-v1/freeze-outcome.json) 为 return code 1，保存 [readiness](../../runs/remote-h200-b6-quality/runs/b6-quality-dev-phase1-v1/b6-frozen.readiness.json)，`frozen_exists=false`。没有修改 epsilon、选择性删格或给最快格绕过质量约束。五个开发会话的探索比较不构成 heldout 或人工非劣证明。

## 预先指定附件问题：13 臂均复核

固定任务 `d5b1e735a040853ed361a3dfde1b8ef0<::>5` 的核心答案在 13 臂均为“单个附件 10 MB”，这由可见正文支持。复核依据是每臂实际 manifest 对应的完整来源，不是历史助手陈述、数值关键词或外部今日网页：

| reference ID | 原 passage | 实际支持范围 |
| --- | --- | --- |
| `E_53079fa73a` / `E_fe06da00dd` | `ibmcld_00558-23465-25360` / `ibmcld_12904-23555-25450` | 明列 document 1 MB、single attachment 10 MB、request body 11 MB；建议二进制附件放 object storage 并保存链接 |
| `E_8312203498` | `ibmcld_00510-5537-7566` | 文档建模、document 1 MB、附件功能与使用前注意事项；没有“Single Attachment Size 10 MB”或 object-storage 建议原句 |
| `E_6a86be8e02` | `ibmcld_12748-0-2435` | Security and Compliance Center：Attachment **50 per account**，并非 Cloudant 单附件大小 |
| `E_c07431a963` | `ibmcld_06968-18330-20453` | Discovery 的字段大小/摄入限制，不能替代 Cloudant 附件限额 |

Redundancy40 三臂另有两块历史证据 `00500…` / `00581…`；后者也支持 attachment 10 MB，但不改变上述各 ID 的真实绑定。全部原文、正文内精确字符 span 与 prompt manifest span 已保存到 [diagnostics.json](../../runs/b6-quality-review/h200-v1/diagnostics.json)。下面是助手逐源审查，不生成或替换人工评分；原生引用分仍按 1–5 原样记录。

| 配置 | 引用分 | 逐源发现 |
| --- | ---: | --- |
| Rebuild / WT | 4 | 8312 的附件描述与 5307/fe06 的 10 MB 原句分别绑定正确；品牌举例不由当前原文直接给出 |
| Rebuild / SEL | 5 | 列出相关 ID，10 MB 及 object-storage 建议可由 5307/fe06 支持；没有伪造逐条直接引文 |
| Rebuild / WB | 4 | 与 Rebuild / WT 答案相同，直接引文绑定正确 |
| Fixed4096 / WT | 5 | 与 Rebuild / WT 答案相同，直接引文绑定正确；此前实际历史不同 |
| Fixed4096 / SEL | 5 | 列出相关 ID，主结论有源；不等同每个列出的 ID 都有 10 MB 原句 |
| Fixed4096 / WB | 5 | 直接引文绑定正确，复制失败条件有源；品牌举例仍是额外扩写 |
| Redundancy40 / WT | 3 | 先说所给材料未明确限额，与实际可见表冲突；又称 8312/5307/fe06 均一致支持限额，其中 8312 不支持该数值 |
| Redundancy40 / SEL | 4 | 主结论有源、ID 列表可追溯；没有逐条伪引文；额外品牌/性能措辞不能仅因数值正确就全部认证 |
| Redundancy40 / WB | 4 | 与 Redundancy40 / SEL 答案相同，适用相同边界 |
| B1 / WT | 5 | 把 10 MB 原句直接引到 8312，错绑；5307 的对应引用正确 |
| B1 / SEL | 5 | 把 10 MB 原句直接引到 8312、6a86，两处错绑；fe06 正确 |
| B1 / WB | 5 | 把 object-storage 建议引到 8312、把 10 MB 引到 6a86，两处错绑；其他列出的引文有对应来源 |
| B0 | 5 | 与 B1 / SEL 答案相同，包含相同两处错绑 |

B0 和 B1 三臂在此任务的五个维度均获 5 分，引用理由还明确称 ID 正确支持其陈述，未发现上述错绑。另一方面，Rebuild / WT 与 WB 的条件理由要求明确“单个附件”，但答案已写 `Single Attachment Size` 和 `single attachment`；Redundancy40 / WT 的引用理由称没有明确引用，答案实际列出了三个 ID，却未指出 8312 的具体支持错误。评价器可能漏掉实质关系错误，同时用不准确的格式/表达理由扣分。

答案相同不等于完整 judge 输入相同。逐维重建核对显示：Rebuild WT↔WB、Redundancy40 SEL↔WB、B1 SEL↔B0 的完整 messages 相同，分数也相同；Rebuild WT↔Fixed4096 WT 的答案虽相同，实际历史和 messages 不同。因此这里**没有据此宣称重复评分不稳定**。确定的缺陷是“这份引用原文是否包含所引内容”被漏判，参照 B0 也有该缺陷；它不授权更换参照或改动既定门槛。

## 生成截断与未引用诊断

全部 **13 个 generation length** 都是 1024 tokens，覆盖四个 task：Query API 细节 `1c041…::4` 七次、电话连接/优化 `b461…::6` 四次、namespace `c6c3…::9` 一次、schema/table `1c041…::7` 一次；其 65 个 judge 维度完整保留。judge 的 2665 个 `stop` 不能消除被评答案本身的截断。

13 条均无 parser 可识别的 `[E_…]` 引用，4 条 Markdown code fence 未闭合。逐尾段检查可见 Fixed4096 / WB 电话例子停止在 `websocket.onmessage =`，同臂 Query API 例子截在 JSON 字段，B1 / WB 电话代码在语句中停止。这些代码没有被执行；围栏奇偶本身只作诊断，实际尾段给出了未完成的具体证据。部分长答已经回答了主要问题，不能把达到预算统一判成整答无效，也不能将其无条件排除在质量统计外。

全体 533 答案中 **419 条无可解析 E_ID，未知 E_ID 为 0**。没有 E_ID 可能仍含来源 URL，因此该计数不等于“没有任何引用”或“事实无依据”。这 419 条的原生引用分为：2 分 1、3 分 147、4 分 258、5 分 13。一个可确认的反例是 Rebuild / WB 的电话任务：答案无 E_ID 或 URL、在代码围栏中截断，citation 最终仍为 5，理由声称提供了有效引用；理由正文另提 4 分也没有被拿来回填，原 parser 只按末尾 Result=5 记录。

本轮完成了固定协议下完整开发质量评价，但格式可靠性、统计证据和语义可信度是三件事：格式本批次完整；统计尚未支持任何候选五维联合非劣；来源绑定和截断判断仍有已复核漏检。后续应把这些错误映射到生成/证据机制和固定来源核查，而不是继续换模型、宽松解析或调整 epsilon 直至通过。新的机制或评价协议需要单独冻结完整比较；当前数据、原分数和未产 B6 的结果继续保留。
