# 问题相关行与行内来源：冻结开发探针

> 归档记录：保留当时的设计、结果与复用依据；其中“下一步”和执行计划不代表当前任务。当前工作仅以[实现清单](../implementation_plan.md)为准。

2026-10-24，三臂、两个冻结历史、正反顺序共 12 次运行全部成功。`focus_refs` 在两个固定历史中均给出了有正确原始引用的附件上限答案，没有重现已知对象错绑或新增示例错误；每臂两次 output IDs 完全一致。这是**目标缺陷上的净局部改善**，支持继续可选集成和更广回归，不构成总体质量非劣结论。

输入与实验代码在 `runs/table-frozen-pairs-v2/`，原始结果在其 `h200-order-pairs/`，`review.py` 生成 `review.json`。它核验完整输入、来源和 receipt，并保留全部 12 份答案。语义结论来自 assistant 对原可见证据的逐条核查，不是 human review；没有调用 judge。

```bash
.venv/bin/python runs/table-frozen-pairs-v2/review.py
```

## 三个实验臂与复用范围

- `off`：原始输入。
- `all_rows`：上一版表格注释，完整消息/input IDs 与 v1 的 on 严格相同。
- `focus_refs`：根据当前用户问题选焦点行，每行直接附原 `[E_ref]`，加一条不从行关系推断跨行包含或性能保证的局部说明。完整正文和 QA 不删除。

实际复用了 [CRAFT 固定 MIT 源码](https://github.com/coral-lab-asu/CRAFT/blob/7c4be45bb292f4e050c4237b4141ccc9e91ebe30/src/craft_tabqa/core/text.py)的 `_clean` / `build_row_texts`，许可随实验保存。当前选择器是 BM25S 分词、简单复数归一和字段词重合，**不是 CRAFT dense reranker 的完整复现**。它复用既有 `correction_cue`，仅对明确分隔的纠正引导语取后续目标；无匹配、并列最大分、未解析否定或多行比较时保留全部或并列行。没有附件、产品、数值或任务 ID 的专用选择规则。

本臂同时改变焦点选择、引用呈现、辅助文字长度和局部范围指令，属于混合干预。效果不能单独归功于其中一项；必要时可再拆 focus-only / cite-only 对照。

## 输入与执行证据

case 文件和运行脚本 SHA 与 manifest 相符，冻结时记录的三个代码文件也通过核对。两 fixture 各保留 9 条先前消息，分别为 1327 / 1275 个完全相同的历史 tokens；所有原证据跨度与引用不变。唯一消息变化发生在当前 Question 前的辅助段。

| 固定历史 | off 输入 | all_rows 输入 | focus_refs 输入 |
|---|---:|---:|---:|
| old_error_history | 4917 | 5334 | 5027 |
| fresh_off_history | 4865 | 5282 | 4975 |

`focus_refs` 比 off 增加 110 tokens，all_rows 增加 417。它并未压缩或删除 required 正文，只缩短新增表示。

每个 arm 前独立冷置，12 次 GPU/host 使用量均为零、无活动或排队请求；receipt 均为零 GPU 命中、零 host 加载、完整输入长度的 prefill，epoch 退役均确认。全部 finish 为 `stop`，无截断。每个 fixture/arm 两次文本与 output IDs 相同；off 和 all_rows 的全部输出还逐 token 复现了 v1，因此旧问题没有被悄悄换成新的基线。

资源范围：Qwen 的 GPU/host pool 为 **121920 / 279040 tokens**，c1 冷隔离。按部署记录，同一 GPU0 上 Selene stock 模型已加载，但这 12 次期间未推理；这项共驻信息来自运行方记录，不是由 Qwen receipt 推断。不能与 16K 性能矩阵或真实 P3 并发吞吐混比。

## 对全部输出的来源核查

| 历史与实验臂 | 两次均观察到的行为 |
|---|---|
| old / off | 仍将文档连附件合计限制为 1MB；单附件 10MB 本身有据，但无引用 |
| old / all_rows | 移除已知 1MB 错绑并分别列出 1/10/11MB；仍无引用，且追加未充分验证的请求体组成解释和无性能影响保证 |
| old / focus_refs | 只回答本轮所问的单附件 10MB，并引用 `E_53079fa73a` 与 `E_fe06da00dd`；逐字引文 `Single Attachment Size 10 MB` 在两份原表均存在。没有跨行包含、请求体求和、性能保证或多余代码示例 |
| fresh / off | 单附件 10MB 正确，但在正确表引用外混入 `E_8312203498`，该段未提供附件上限或相应复制限制 |
| fresh / all_rows | 单附件上限仍正确，但删除了所有 Evidence ID；新增 JSON 的 JPEG MIME/后缀仍与 PNG Base64 签名不符 |
| fresh / focus_refs | 保持单附件 10MB，使用两份真正支持该行的原引用，没有 `E_8312203498`；未添加未问的包含关系或 MIME 示例 |

两个 focus 答案都不是仅“出现 10MB 和合法 ID”：已核对 ID 对应的完整原段落、原表头的最大值语义及具体对象行。两份表正文相同，它们是两个实际来源标识，不能称为两份独立佐证。本轮只问附件上限，省略其他安装/复制细节不是已证明的完整性损失。未观察到新的核心错误，但这些选择过的开发样本不覆盖其他对象、长表、跨版本、条件例外和问题类型。

## 时间：仅局部描述

| 历史 / arm | 输出 tokens（两次相同） | guarded generate 墙钟均值 ms |
|---|---:|---:|
| old / off | 277 | 1403.3 |
| old / all_rows | 218 | 1141.3 |
| old / focus_refs | 114 | 645.9 |
| fresh / off | 366 | 1818.3 |
| fresh / all_rows | 365 | 1823.5 |
| fresh / focus_refs | 101 | 585.0 |

更短输出与本次较低生成时延同时出现；不能据此单独推断查询焦点解析、缓存策略或整个 P3 的性能收益。这里不含检索、控制器和完整会话等待，焦点表示在冻结阶段预计算；共驻模型状态也与其他实验不同。冷置及 epoch prepare 时间另存原始记录，没有混入 generate。

## 集成判断与后续验证

已将此样式作为显式可选项接到共享候选入口，保留默认关闭、all_rows 对照、完整 QA/证据/引用以及既有预算检查。候选间共用一次当前问题的焦点选择，controller/native/matrix 使用同一配置。可选工程能力不等于默认质量已经通过；接续应验证全部开发会话、非触发轮、别的纠正/并列/否定与条件样本，并在完整 P3 时间内计入选择开销。保留 v1 的负面现象与全部基线输出，再决定是否拆分混合干预或引入现有 BGE 行评分。

## 已实现的可选工程入口

`adapters/table_focus.py` 与私有 `adapters/_craft_row_text.py` 已接入共享候选，完整 MIT 许可见[CRAFT notice](../third_party/CRAFT_NOTICE.md)。默认仍为 `table_relations: false`；样式默认为 `all_rows`，因此旧全行路径和默认关闭路径均保持原输出。只有显式启用并选择 focus 才使用新样式：

```yaml
context:
  table_relations: true
  table_annotation_style: focus_refs
```

原生 CLI 对应 `--table-relations --table-annotation-style focus_refs`；matrix 的两类执行路径继承共同 base config。该配置进入现有 policy/run provenance，不创建新的 hash 体系。`focus_refs` 使用已有 retrieval extra 的 BM25S；未启用时不加载它。

每轮 required 仍由同一证据策略决定，解析一次并共用焦点注释；不会因行选择从 required 删除正文。APPEND 复用旧正文时当前消息仍有焦点注释，旧消息保持原样。当前问题直接传给选择器，不使用历史助手文字来决定焦点。增加的输入继续使用统一预算，溢出返回 `shared_context_policy_overflow`，不静默裁 QA 或正文。

本地 **114 项相关测试与 6 个子测试通过**，Ruff 通过；覆盖通用词法/纠正线索、并列和未知逻辑回退、脚注、独立布尔开关与 style 校验、两种布局、完整 body/span/QA、原生与矩阵继承、默认逐字兼容及哈希调用次数不增加。另用固定真实 tokenizer 将生产 `build_candidates` 生成的两 fixture × 三样式与本实验冻结 input IDs/hash 比较，**6/6 完全一致**。这是接线等价验证，没有再发起 GPU 请求；它不代替完整 P3 回归或独立质量评审。
