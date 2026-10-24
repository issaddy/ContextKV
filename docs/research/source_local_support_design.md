# 从段落计划转向指定来源的支持地址

2026-10-24。基于[12题真实诊断](../archive/source_plan_diagnostic_validation.md)，本轮先实现一个离线表示入口：**问题要求 → 指定来源内的逐字片段与显式条件片段**。它复用已有定位和身份机制，没有新增分句器、图服务、模型或GPU调用；尚未作为生产生成方法，也未建立质量收益。

## 为什么改变表示，而不继续扩大原实验

旧P将整段ID与自由applicability/uncovered文字关联，已经出现“段内有2007年数字却说没有”、债券time已有定义却说未解释、跨来源串接海平面数值、Functions问题选BMSCore材料等错误。这些不能全部归于段落太粗：两个时期可能就在同一句，正确段落也可能已选中。最终回答还可能忽略计划条件或不输出引用。

因此本次只解决一个可保证的环节：选择的文字究竟来自哪个原文位置，条件片段是否被明确携带。问题理解、语义适用性、条件是否找全及最终改写仍需分别判断；不能把位置成功升级成支持成功。

## 已有方法真正提供什么

| 复用来源 | 借鉴的机制 | 必须保留的差异 |
|---|---|---|
| [AFTG，ACL2024](https://aclanthology.org/2024.acl-long.182/)及[固定作者实现](https://github.com/lovodkin93/attribute-first-then-generate/tree/cc7b995dfb288da7982ae388994f182bef030b75) | 先选指定文档的连续highlight，再组织支持组和句子实现；可以跨句 | 句界不是原子事实证明。作者部分定位代码去标点、跨源改绑、取首位置及LCS修复不适合本项目；不能将10.5与105或同文异源合并 |
| [ReClaim，NAACL2025](https://aclanthology.org/2025.findings-naacl.55/)及[固定作者实现](https://github.com/pdxthree/ReClaim/tree/bc997dda1340eb6facaf1787294bad3377754914) | reference/claim交替，trie约束引用文字；完整方法有专门训练 | IG实现切claim adapter时清KV，只给已生成reference/claim前缀，回到reference时恢复原prompt。旧v2全源final没有这一隔离；不能把枚举ID或短quote称作复现，也不能直接搬其adapter/cache处理到当前Qwen引擎 |
| [JCA固定作者实现](https://github.com/jenhsia/joint_constraint_aggregation/tree/3cb4413e4b305893d3206a302eea29954279b66e) | 从问题抽共享要求，区分候选与属性有明确联系还是仅共现 | 原任务、训练/调用预算和输入不同；未选到支持只能代表本次未解决，不证明材料中不存在语义等价答案 |

固定来源、许可及6组CPU源码反例见[复用备忘](../../runs/source-local-support-research/README.md)。复核发现的ReClaim隔离差异很重要：若以后研究这个环节，需要另行声明各阶段可见材料、完整QA如何保留、模型/adapter身份与全部调用/KV成本；本轮没有暗中改变当前输入合同。

## 当前薄原型

[实现与说明](../../runs/source-local-support-prototype/README.md)直接复用 `bench.atomic_judge._positions`、现有实验 `Catalog/SpanRef/close_selection`。没有新matcher或在线哈希；定位器从已有可见来源建立一次ref映射。

```text
requirements: [{text, supports: [{reference_id, quote, qualifiers: [quote, ...]}]}]
```

每个quote只在指定原block.text中逐字匹配，不strip、不改Unicode/换行/标点、不向其他来源找替代。零命中、唯一命中、多命中分别保留；多命中全部返回且不默认第一处。同文异版本/来源继续独立，字符坐标相对EvidenceBlock正文，不混入全文件或UTF-8字节偏移。跨源支持必须逐源明列，不能拼成伪连续span。

qualifier是显式选择的上下文候选，不是系统推断的条件证明。已有明确block依赖只用于完整输入的可见闭包，缺项保留；不会扩大quote或自动补语义条件。完整正文/QA保持，位置选择本身不节省其prefill。

不提供自由全局`uncovered`字段作为宿主认可的事实结论。空supports记`unresolved`；没有任何选段时为`no_selection`。混合计划中，`location_binding=bound`只说明**已经选择的地址**全部唯一，内层未解决要求仍在；`coverage/semantic_support/condition_completeness`始终未认证。这不是回答准入门，普通final仍可能自行生成错误缺源声明，不能宣称删掉一个字段就解决幻觉。

## 本轮验证与边界

- 6项通用反例通过：小数点/空白、错源、同文异版本/重复位置、Unicode局部坐标、显式条件与缺依赖、空支持不推出不存在。包含“漏掉真实条件仍能位置匹配”的负例。
- 前轮111处审查位置中，只有43条是当前来源，另42答案、24准备、2历史。43条来源记录包含41个唯一坐标，覆盖12题；本轮全部定位到同源原区间。它们是选择性assistant审查见证，不是穷尽事实集或线上gold。
- 真实8B tokenizer计这43个片段为6–51 tokens、合计889。这里只测已有见证的表示尺寸，没有把全部要求、条件和JSON包装视为免费，也不据此保证384-token生成会正常结束。
- 没有模型、HTTP、heldout、参考答案或新质量评分；旧v1/v2和48次真实调用的原件不改。独立分类与反例边界见[设计审查](../../runs/source-local-support-review/design-review.md)。

当前保留为离线工具。下一机制必须具体处理需求指代、同源错误对象、条件遗漏或final改义中的某个通道，并说明相对简单直接回答为何值得；仅43处定位通过或文字更短不足以重开GPU。单次工程基座继续集成，尚未获收益支持的两阶段发布/恢复状态机不作为工程交付前置。

本轮离线实现、原始固定源码见证、引用坐标与说明另存为[26文件研究快照](../../runs/source-local-support-artifact-v1.tar.xz)，137396 bytes，SHA256 `c28ec6252f81789ba11232a2b48e0056371d7dcc115b42c7deb9100f350979eb`。复算仍使用匹配核心源码及此前已导出的8B tokenizer metadata；快照不含模型权重或环境。

## 公开归因监督的实际适配检查

[LongCite-45k固定数据](https://huggingface.co/datasets/zai-org/LongCite-45k/tree/b776b01faf6e132a3d9cc373de1b46cd6630d772)的实际布局为单个约5.73GB JSONL。本轮先固定6个等距1MiB字节窗口，每窗只取前两条完整记录，共12条/6MiB；未依内容扩样或下载全量。原窗口、HTTP Content-Range、坐标、脚本与[审查结论](../../runs/longcite-data-review/README.md)保留。这是字节分层便利样本，不能推断总体质量率。

100个statement含81个引用区间，全部能精确回定位；29个区间跨chunk、15个statement联合多个区间。确有时间、分支和量词保留的正例，因此其价值不局限于引用格式。但选择性原文审查同时发现相邻句错引、来源执行主体变成用户义务，以及把不同条件下数值组合成一般范围。合法地址没有排除本项目最关心的错误。24条观察标记为assistant_audit，不当gold，不修改原target。

12条仅有prompt/response，没有实际多轮角色、canonical URI、文档版本或answerability字段。40个空cite statement混有标题和实质断言，不能自动标成无支持。原Qwen3 tokenizer计数中6条仅prompt就超过当前32768上限，未为凑窗口裁掉正文；这是裸训练字符串计数，不是训练成本。与既有dev白名单做exact匹配未命中，也不能据此宣布全量去污染；未读取heldout内容。

**当前技术决定：复用断言与多段支持区间的表示思想，暂不直接训练原数据。** 下一项投入应解决多轮条件、无支持监督和来源隔离的实际缺口；若仍缺这些信息，格式SFT不能冒称范围保真。不会自动扩引用到邻句来“修好”target，也不以同一模型自评生成金标。用户的持续研发授权不构成阻碍；是否训练由实际适配性和预期净收益决定。
