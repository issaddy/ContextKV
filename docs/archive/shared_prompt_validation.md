# 共享提示词开发验证：Q0 / Q1 / Q2

> 归档记录：保留当时的设计、结果与复用依据；其中“下一步”和执行计划不代表当前任务。当前工作仅以[实现清单](../implementation_plan.md)为准。

截至 2026-10-24，**保留 Q0 默认，Q1/Q2 仅保留为可选实验配置**。Q1/Q2 在部分回答中改善引用或减少无据命令，也存在错引、跨产品回答及遗漏。Q2 输出更短不能据此视为质量保持下的性能收益。本轮没有外部 judge 调用，没有独立盲审评分，也没有建立语义质量非劣结论。

## 固定对象与可复核材料

基准来源为 `runs/remote-h200-native/b1-cloud-dev/native-B1-f7217274af984af9966ab39c910a9adb.json`。从该开发运行固定**全部 41 轮、5 个会话**，没有按答得好坏删选。集合文件为 `runs/remote-h200-quality/review-v2/q0-p2/cohort.json`，保存任务、问题、原检索上下文和运行条件。六次运行均为同一固定模型/tokenizer、原生 B1、并发 4、1024 输出上限。

Q1/Q2 P2 及 Q0/Q1/Q2 P1 原始结果位于 `runs/remote-h200-quality/quality-q{0,1,2}-b1-p{1,2}`，其中 Q0 P2 使用上述早期 B1 源文件。每个结果都提供真实 `visible_evidence_manifest` 与全部 `audit_evidence`。

独立脚本 `scripts/review_saved_answers.py` 已为六组生成：

- `runs/remote-h200-quality/review-v2/q{0,1,2}-p{1,2}/review.jsonl`：逐轮原回答、实际历史、完整证据、真实 ID 映射、自动检查及待填写的逐 claim 审查字段。
- 同目录 `review.md`：供阅读的完整审核包，证据正文不截断、不只保留当前 top-k。
- 同目录 `summary.json`：自动统计、解析器版本与运行条件差异。
- `comparison-p1.json` / `comparison-p2.json`：共同集合、证据集合和实际历史差异。
- `manual-focus.json`：本代理逐条阅读的 24 条定向论断记录，包含原回答偏移、证据 ID、完整来源正文和判断理由；不是独立人工标注或总体准确率样本。

原始 native JSON、原始 audit 文件和早期诊断均未改写。`review-v2` 为新规则下重算的独立产物。

## 自动指标与边界

| 协议 | 提示词 | 成功 / 固定轮数 | 含稳定引用（含组合） | 严格单 ID 格式 | 仅组合格式 | 截断 | 输出 token 均值 |
|---|---|---:|---:|---:|---:|---:|---:|
| P2 | Q0 | 41 / 41 | 10 | 10 | 0 | 3 | 564.34 |
| P2 | Q1 | 41 / 41 | 24 | 23 | 1 | 0 | 458.59 |
| P2 | Q2 | 41 / 41 | 24 | 19 | 5 | 0 | 266.71 |
| P1 | Q0 | 41 / 41 | 4 | 4 | 0 | 0 | 296.51 |
| P1 | Q1 | 41 / 41 | 13 | 13 | 0 | 0 | 287.78 |
| P1 | Q2 | 41 / 41 | 6 | 6 | 0 | 0 | 170.12 |

六组都是 41/41 完整可见证据可用、0 个未知 ID、0 个任务问题或原检索上下文变化。三组 P2 的可见证据 key 集合一致；Q1/Q2 各有 36 轮实际 QA 历史不同于 Q0，故需要按会话解释闭环变化。三组 P1 的实际历史与可见证据都相同，但 P1 是参考历史协议，不能替代 P2 的真实闭环表现。

`bench.judge.parse_citations` 使用版本 `evidence-id-groups-v2`，识别 `[E_ab12]` 与 `[E_ab12, E_cd34]`。未知 ID 在组合中也被检查。`strict_format_citation_count` 保留旧单 ID 格式计数；普通 `[1]`、裸 `E_ab12` 或 URL 不算引用。此次 Q1/Q2 确实产生组合格式，修复的是实际漏报，不是通过随意接受地址提高分数。

自动检查只能回答：任务是否齐全、证据正文是否齐全、ID 是否存在、格式、截断和长度。命令/URL/点分数字的字面出处只作为审查线索，**不能自动证明语义支持**；同样字面片段可能来自另一个产品或版本。上表不代表引用 precision、correctness、answerability 或 groundedness。较低 token 均值也不是质量约束的时延结论。

## 固定问题的逐论断复核

以下检查沿用同一组问题；其余论断保持待审状态，不外推错误率。完整定位与原文见 `manual-focus.json`。

| P2 问题 | Q0 | Q1 | Q2 |
|---|---|---|---|
| VA 扫描加密镜像，`c6c3…<::>1` | 核心否定正确；多引的 `E_630a704598` 没有加密扫描句 | 只引用直接支持的块 | 两个直接支持的块，去掉无关块 |
| CocoaPods 警告，`c01c8…<::>2` | post-install 修复实际在 `E_ad545fb062`，却反复挂在较早片段 `E_57a8237f7e` | 新增“CocoaPods 安装后到 Android Studio 同步 Gradle”错误范围拼接 | 正确使用后续修复块与 IAM namespace 限制；本案有所改善 |
| 获取 VA4，`c6c3…<::>4` | 补出可见材料没有的 `cr va scan` / `cr va --version`；历史发布条目被称为 latest | 不再给该 scan 命令，但把 CLI/plugin 默认输出变化归因 VA3→VA4 | 正确定位 Container Registry plug-in 1.0.0，省去无据具体命令；“or later”仍是推断 |
| 创建加密镜像，`c6c3…<::>6` | 无据扩展解密→push→扫描流程 | 相同无据流程后增加 ID | 更短，却建议改用未加密镜像或临时解密；材料仍未支持该推荐 |
| 文件大小，`d5b…<::>4` | 核心 1MB / 10MB 有来源，但称 `E_2bd6af736f` 也支持；该块实际谈 design document/view、64MB | 只回答文档 1MB，遗漏承接图片/PDF附件的 10MB；错误额外来源仍在组中 | 恢复附件/文档区分，但仍多引不支持该上限的 `E_2bd6af736f` |
| 明确问附件，`d5b…<::>5` | 错引 `E_6a86be8e02`：它是 Security and Compliance Center 的 Attachment 50/account；另一个来源才支持 Cloudant 10MB | 核心 10MB 绑定到正确两个来源 | 核心 10MB 绑定到正确两个来源 |
| Cloudant Python SDK，`d5b…<::>9` | 427 tokens 安装/代码/链接，全部可见材料没有提供 Cloudant SDK | 524 tokens 换为 `ibmcloudant` 代码，仍无可见来源支持；末尾承认 iaesdk 属于 Analytics Engine | 91 tokens，却把 `IBM/ibm-iae-python-sdk` 说成 Cloudant SDK，并挂 `E_f577ce10ca`；来源明确是 Analytics Engine API |

这里最有区分力的反例是 SDK：Q2 的第一句承认资料未直接提及 Cloudant SDK，随后仍把另一产品的仓库用于 Cloudant。因此“更短、带已知 ID、未知 ID 为零”可以与事实和来源错配同时出现。附件案例也说明，核心答案可以正确而引用绑定错误，不能把二者混成一个 presence 指标。

P1 的同一 SDK 问题同样没有通过来源约束：Q0 从证据外补 `ibmcloudant` 示例；Q1 明确将 Analytics Engine 的 `iaesdk` 用于 Cloudant 并补读写示例；Q2 补出可见资料没有的特定 SDK/core 安装步骤。故不能仅用 P2 历史传播解释问题。

本轮不再扩展提示词搜索。默认 Q0 保持，是避免将未经质量验收的候选推为新默认，并不表示 Q0 已达标。后续若要选默认，仍需预先固定的完整语义审核、正确拒答与过度拒答检查，并覆盖所有比较方法；不得只挑本表改善的案例。共享提示词和审核工具是可靠实验的工程条件，**不作为课题 A/B/C 的新方法贡献**。

## 复算与审查模板

```bash
# 仅首次固定Q0开发集合；新目录避免覆盖已有审查记录。
python -m scripts.review_saved_answers \
  --run runs/remote-h200-native/b1-cloud-dev/native-B1-f7217274af984af9966ab39c910a9adb.json \
  --output runs/review-repeat/q0-p2

# Q1/Q2使用完全相同cohort；RUN替换为对应真实native JSON。
python -m scripts.review_saved_answers \
  --run "$RUN" --cohort runs/review-repeat/q0-p2/cohort.json \
  --output runs/review-repeat/q1-p2

python -m scripts.review_saved_answers \
  --compare runs/review-repeat/q0-p2 runs/review-repeat/q1-p2 \
  --output runs/review-repeat/comparison
```

控制器 benchmark 若不内嵌完整证据，可额外传 `--audit` 对应运行的完整 audit JSONL。缺失或失败轮仍保留；缺失 audit 不降格为“已完成支持验证”，不会用 top-5 代替旧可见证据。

每条审查至少填写：原回答论断、它声称的引用、实际支持/冲突来源及原文、产品/版本/条件范围、支持/部分支持/错引/可见资料不足/未决。分别记录完整答案是否答题、是否遗漏关键条件和是否应拒答。未审内容保持 `pending`，不能自动补为通过。
