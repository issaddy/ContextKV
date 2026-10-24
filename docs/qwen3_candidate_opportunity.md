# 既有 Qwen3 轨迹的候选结构审计

2026-10-24。复用已经完成的 Qwen3-8B 四域159轮、22会话参考，**零新增模型/HTTP调用**，逐token复现全部原REBUILD输入，再构造137个真实前序R状态上的一步APPEND候选。当前最重要的结果是：较高前缀复用量不等于较少新增输入工作，更不等于完整响应更快。按证据交集分组呈现出两类明显不同的取舍，没必要先为所有位置做GPU标定。

## 复用与因果边界

直接复用现有 `build_candidates`、原参考runner的完整材料/历史装配、真实8B tokenizer和 `ExecutionLedger` 数据结构。临时ledger只表示已有R输入与实际输出的执行视图，不创建原生session、receipt、驻留或cache quote。没有新增预测器、旧7B拟合系数或模型服务。

每个会话仅由**真实R结果**推进下一状态；本轮实际答案在候选构造后作为原R结果单列，不作为本轮特征，也不注入假A后继续滚动。159个原R的IDs、hash、messages、spans、required/visible、reference map与fingerprint全部一致，22个首轮只有R，137个后续轮各有一步A/R。全部A符合当前context policy，但这只是32768窗口/输出预算下的候选可行性，不证明并发GPU容量或速度。

原参考是Q0、non-thinking、`.7/.8/20/0`、输出上限1024；不能冒充后来的D目录提示。两候选均保留完整QA及本轮required正文。A多保留的nonrequired证据仅指当前policy不要求，不自动表示语义无关、有害或可以删除。

## 结构量与实际结果

对每个候选的输入长度 `n`，直接与上一轮**真实input_ids + output_ids**计算逐token最长共同前缀 `L`；另列 `floor(L/64)*64`。定义潜在新增输入工作减少为 `(n_R-L_R) - (n_A-L_A)`，额外上下文为 `n_A-n_R`。两者都是理想前缀可用假设下的形状量；不推断SGLang的实际match_limit、已插入页、GPU/host ready/pending或共享pin。

| 当前required与上一R可见证据的交集 | 位置数 | 潜在新增输入减少，中位数 | A额外上下文，中位数 | page64口径无正减少 |
|---|---:|---:|---:|---:|
| 空 | 81 | 15 tokens（范围8–39） | 2703 tokens | 18 |
| 非空 | 56 | 1108.5 tokens（范围384–3924） | 1673 tokens | 0 |

这个分组由已有key交集确定，没有调阈值寻找胜例。全137位置raw口径均为正，中位仅22；page64口径只有119个为正。A额外上下文中位2391，最大输入13217；原R最大输入9328。137位置来自22个会话，不视为137个独立系统重复；完整产物另保留按会话等权汇总。

没有证据交集时，A保留旧源，但本轮新源仍需进入模型，剩余差主要涉及原QA与模板的重排边界，未显示重复正文复用；R本来就保留完整QA，所以不能把保留旧源的成本误当保留历史的必要成本。有交集时，A才有较大的潜在重复正文计算节省，但上下文仍变长。

## 为什么不能只比较新token数

现有 `controller.cost_model.positive_prefill_features` 已含注意力几何基 `U*P + U²/2`，其中 `U=n-P`。本次仅复用该纯函数的这一项，对已保存形状做离线后处理，不创建CostModel或调用任何拟合系数。它描述dense causal attention的交互规模近似：严格token-pair计数还有线性项U/2，现有模型另有U基。它不是完整FLOPs、实际kernel耗时或ms预测；更长prefix仍会增加新query的注意力工作。调用纯函数时的load/concurrency占位参数不影响这一项，也不是host恢复量或实际并发观测。

采用page64口径，81个零交集位置的A注意力基全部高于R，A/R比值中位约1.87；56个有交集位置仍有33个更高，比值中位约1.19。这解释了“新tokens略少、注意力工作却更多”为什么可以同时发生。占用更长上下文也会影响后续decode，但没有A真实输出长度，不能把R输出当A输出计算端到端收益。

这些数值都以给定前缀可用为条件。实际KV传输、设备kernel、排队、输出、未来复用与质量未知，因此不宣布动作支配或最优策略，也不据此直接新增一个证据交集阈值规则。

## 一个需要先理解的模板边界

独立检查还确认，137个A的raw LCP均止于上一R的input长度减4，未延伸进其output。保存的原Qwen3模板在non-thinking生成提示末尾加入空`<think>…</think>`，而该assistant成为更早历史后不再渲染这段前导；这与逐token中断位置一致。它是原模板的行为，不是缓存错误。保持规范历史重渲染就必须接受这个边界，不能因业务答案相同就宣称原output KV可直接复用。

因此下一步先核模板/实际IDs与原生增量会话的接点，判断是否存在值得研究、身份明确的序列化适配；若改变渲染字节，必须视为新的prompt/template范围，重新考虑质量和成本，不能悄改默认或沿用旧cache fingerprint。当前没有修改模板，也没有据此启动新GPU实验。

## 已交付与下一投入

5项因果/溢出CPU合同与Ruff通过；[独立审查](../runs/owned-local-scope-probe-research/review-replay.json)核对全部159分母和结构公式，并另从四域晚轮分别独立组装A/R输入，均无问题。[CPU回放与合同](../runs/qwen3-candidate-opportunity/replay.py)、[逐位置数据](../runs/qwen3-candidate-opportunity/full/pairs.jsonl)、[汇总](../runs/qwen3-candidate-opportunity/full/summary.json)保留全部159分母。先固定四域8轮小样检查，再只完成一次全cohort结构回放；之后的分组/注意力分析只消费已存pairs，不重复tokenize或调用GPU。

下一步先核现有成本与原生观测能否解释两类形状，只对会改变策略选择的代表位置安排必要测量。零交集组不值得自动投入整表拟合；有交集组也必须同时看新增prefill、长上下文decode和完整响应，而非只看hit tokens。C的host准入仍默认关闭，本文没有提供自然压力或回访收益证据。语义条件与引用错绑仍是A的独立关键瓶颈，不能用这份结构数据授予质量资格。

本阶段[37文件复算快照](../runs/owned-and-opportunity-review-v1.tar.xz)同时保存四调用公开原件、独立审计、CPU回放/分组/几何脚本与产物，216244 bytes，SHA256 `7286359475c8f9ae6dae334f01f96f976c097848432cf4fea6eea50364d06feb`；36个payload核验通过。复算仍需既有159轮原件及8B tokenizer metadata，快照不重复保存模型权重或私有环境。
