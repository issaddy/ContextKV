# 直接成对收益校准：嵌套会话验证

> 归档记录：保留当时的设计、结果与复用依据；其中“下一步”和执行计划不代表当前任务。当前工作仅以[实现清单](../implementation_plan.md)为准。

2026-10-24。新增独立工具 `profiling/gain_fit.py`，已在冻结的 90 个硬件 pair 上运行。保留原 margin、稳定轮数和时间支持域时，candidate 长度组的平均 regret 从 **270.30 降至 236.14 ms**，但仍高于固定 REBUILD 的 **129.14 ms**。更关键的是，三个收益模型的动作与“时间支持且稳定时就 REBUILD”的简单规则完全一致；本轮尚未显示它们比这个规则更会区分具体案例。线上 planner、默认配置和远端服务均未修改。

结果为开发数据上的算法探索，答案质量没有进入本次评价。两臂产生的回答可能不同，实测时延更低不等于回答质量更好；这里的 regret 不是质量约束后的 oracle regret。

## 标签、特征与样本单位

复用 `runs/remote-h200-expanded/decision-dev-stratified/` 的冻结输入、提交前 native quotes 和实测 pair。数据为 5 个会话、15 个 case、cold/gpu-ready/host-only 三种独立 fixture、每种两次重复，共 90 对；它们不是 90 个独立会话样本。所有回放预测仍按隔离 c1、control=0 解释，不还原原始 P2/c4 的并发缓存状态或决策传播。

目标是实测 `G = server_elapsed_ms(APPEND) − server_elapsed_ms(REBUILD)`。正值表示这次 REBUILD 更快。当前输出长度只用于训练长度模型的离线标签；当前结果文本、生成 token、实测时延及 oracle 动作不作为预测特征。

收益模型用三个提交前特征拟合 `G − 原预测G` 的 ridge 残差：两臂预测 first-token/restore-prefill 成本差、预测 Decode 成本差、输入 token 数差。当前时间 artifact 的 first-token 项是服务器联合 TTFT 范围，不被当作纯 CUDA Prefill kernel 时长。标准化、截距和系数仅由训练会话计算；case ID、session ID 和 fixture 名称只用于分组，不作为数值输入。

拟合先把相同 case/state 的重复观测平均，再按会话、会话内 case、case 内受支持状态逐级等权。四种长度来源 shared/action-only/candidate/paired-delta 使用相同收益特征、alpha 网格 `[0.1,1,10,100]` 和单侧阈值网格。完整数据中 82/90 对处于相同的原时间支持集，对应 41 个 state cell；其余 8 对保留原回退动作。每个外层模型有 12 个训练 case、32 或 36 个受支持 cell。

## 嵌套验证及堆叠泄漏处理

外层整会话留一 H。收益模型训练所用的长度预测不能直接取全体会话的已有 OOF：那些训练行的长度模型可能使用过 H 的标签，造成间接泄漏。本工具按以下次序计算：

1. 对每个收益内层验证会话 J，令 S 为排除 H、J 的三个会话。
2. 在 S 内再次按会话交叉拟合长度模型，为收益训练行生成长度特征；给验证 J 的长度预测只用 S 拟合。长度模型自身的正则选择、P90 和误差也只用对应训练子集。
3. 用这些内层预测选择收益正则和单侧阈值。选择目标是保留原门槛时的等会话平均 regret，误 REBUILD 的损失作为平局判据。
4. 最终收益模型在排除 H 的四个会话上重新拟合。其训练长度特征仍在这四个会话内部交叉拟合；只有最终测试 H 使用已有 `output-length-exploration-v2/validation.json` 中 H 的严格 OOF 预测。

阈值计算使用训练内层 OOF 的正侧乐观残差 `max(0, G_hat − G)`：先平均重复，再取每 case 跨状态最大值，选择经验分位数 0/0.5/0.8/0.9，其中 0 明确代表“不加阈值”。alpha 和分位数在同一组内层标签上联合调参，**只有外层结果是报告口径**。这不是 Conformal 校准，也没有样本外覆盖率或总体尾部保证。最深层长度拟合只有少量会话，嵌套消除了这种标签泄漏，不消除小样本高方差。

四组都沿用同一个冻结的 shared prior；OOF 隔离的是新增模型拟合，不追溯声称历史 prior 的构造独立于开发数据。程序记录了每次长度拟合的训练/测试会话、外层收益模型系数、完整内层搜索和选择；没有加载 full-fit 长度模型。

## 决策方式与对照

工具复用原 planner 产生时间支持、原始 margin 和基准动作。普通受支持分支使用 `stable 且 修正G > 原margin + 单侧阈值`；强制重建、不可用报价/未校准等分支继续使用原动作。margin 仍基于原始两臂成本与误差，不因收益校准把两臂成本重新分摊而改变。

每个长度来源输出 baseline、ridge、ridge+单侧阈值三组；每组又输出保留原门槛、只去 margin、只去 stability、同时去二者四个版本，共 48 组。消融复用原门槛下选出的同一拟合结果与阈值，**不会为每个消融重新挑参数**。无 margin 指移除原 planner 的 abs/relative/error margin；单侧模型自身学得的阈值仍保留。线上逻辑没有加入这些实验分支。

## 保留原门槛的外层结果

均为等会话平均 regret，单位 ms；该 cohort 每会话恰有相同 pair 数。四组 baseline 与此前 conditional length replay 的数值逐项精确一致，支持集合也精确一致。

| 长度来源 | 原预测 | 收益 ridge | ridge + 单侧阈值 | ridge REBUILD 次数 |
|---|---:|---:|---:|---:|
| shared history | 366.55 | 236.14 | 236.14 | 26/90 |
| action-only | 342.67 | 301.93 | 301.93 | 20/90 |
| candidate | 270.30 | 236.14 | 236.14 | 26/90 |
| paired delta | 280.36 | 236.14 | 236.14 | 26/90 |
| 时间支持且稳定则 REBUILD | — | 236.14 | — | 26/90 |
| 固定 APPEND | 437.05 | — | — | 0/90 |
| 固定 REBUILD | 129.14 | — | — | 90/90 |

部分外层选择了正单侧阈值，但在这批留出样本上没有进一步改变动作。shared、candidate、paired-delta 的 ridge 动作全部等于上述简单规则。因此这些组的 regret 改善不足以证明学到了优于简单规则的细粒度收益排序。

同样，决策改善不能直接称为点预测更准确：受支持 pair 的 gain MAE，shared 为 568.86→581.84，action 为 630.30→611.88，candidate 为 590.01→588.36，paired 为 572.54→608.05 ms。许多 fold 选择很强的正则，主要表现为训练侧正残差截距修正，不能把它描述为稳定识别了每题输出差。

## 改善与误切换落在哪里

candidate 组新增 12 次 REBUILD，来自 3 个 case 的两个 warm 状态、各重复两次；其中 8 次减少 regret、4 次增加。每个下列数值是同 case/state 两次测量的均值：

| case / 会话 | 状态 | 实测 G | 原预测 G | 修正 G | 动作改变的 regret 变化 |
|---|---|---:|---:|---:|---:|
| c65e01ec / 1c041ce4 | gpu-ready | +25.95 | +105.06 | +325.22 | −25.95 |
| 同上 | host-only | +28.41 | +111.27 | +331.58 | −28.41 |
| 11012108 / c6c3b02c | gpu-ready | −254.53 | +114.17 | +357.86 | +254.53 |
| 同上 | host-only | −252.49 | +118.81 | +363.49 | +252.49 |
| fdf19ecc / d5b1e735 | gpu-ready | +992.38 | −9.64 | +171.13 | −992.38 |
| 同上 | host-only | +997.35 | −2.93 | +177.98 | −997.35 |

candidate 的误 REBUILD 损失均值从 2.99 增至 14.26 ms，错失 REBUILD 的收益损失从 267.30 降至 221.88 ms；净变化为 −34.16 ms。主要收益集中在 d5 的一个 case，不能把重复状态/重复测量当成多个独立成功会话。

| 会话 | candidate 原预测 regret | 收益 ridge regret |
|---|---:|---:|
| 1c041ce4 | 390.06 | 384.02 |
| b4614daa | 246.45 | 246.45 |
| c01c8cf1 | 270.24 | 270.24 |
| c6c3b02c | 223.65 | 279.99 |
| d5b1e735 | 221.08 | 0.00 |

修正后仍有 236.14 ms 平均 regret：按互斥原因分类，稳定轮数未达到对应 189.27 ms，时间支持回退对应 32.61 ms，错误 REBUILD 对应 14.26 ms。这是静态样本的机制分解，不是相互独立的因果贡献。分类之外 54/90 对恰好选择了当次更快动作。

## 匹配门槛消融

表内使用同一已选 ridge，未针对消融重新拟合或挑阈值。单侧版本在这批样本上的动作与 ridge 一致，完整记录仍分别保存。

| 长度来源 | 原门槛 | 仅去 margin | 仅去 stability | 同时去二者 |
|---|---:|---:|---:|---:|
| shared history | 236.14 | 236.14 | 137.54 | 154.63 |
| action-only | 301.93 | 280.36 | 225.23 | 190.35 |
| candidate | 236.14 | 236.14 | 146.65 | 154.63 |
| paired delta | 236.14 | 236.14 | 154.63 | 154.63 |

这说明在该静态数据上，保留稳定轮数确实限制了部分收益，但删除所有门槛并不总更好。即使表内最低值 137.54 ms 仍高于固定 REBUILD。没有据此修改默认稳定轮数；这批 pairs 不包含更频繁重建后的长期缓存传播与后续轮次成本。

## 对后续改进的具体含义

本轮已把“直接校准收益”做成可运行、可审计的对照，但它没有超过简单的稳定期重建规则。下一轮若扩充模型，应预先限定少量提交前 query/history 特征，检验能否分开上述 c6 的错误重建与 d5 的漏判；不能加入 case ID、当前答案长度或 oracle G。现有三特征的大部分信号仍来自既有成本预测，对生成长度差的判断不足，仅增加 Prefill 拟合复杂度没有直接依据。

另一条独立工作是用真实 epoch 生命周期评估一次切换的增量代价与后续摊销，解释固定稳定轮数的作用；不能直接用当前静态 oracle 标签证明应永久取消 stability。8 对时间支持回退仍应通过有针对性的短 suffix 标定补齐，而不是放宽支持域。两项均应沿用相同候选预算和质量核验，不把此次开发选择升级为最终实验结论。

## 复现与验证

```bash
python -m profiling.gain_fit \
  --cases runs/remote-h200-expanded/decision-dev-stratified/cases.jsonl \
  --pairs runs/remote-h200-expanded/decision-dev-stratified/pairs-lookup.jsonl \
  --validation runs/output-length-exploration-v2/validation.json \
  --config runs/remote-h200-expanded/decision-dev-stratified/config.yaml \
  --cost-artifact runs/remote-h200-expanded/cost-dev-expanded-positive-conditional.json \
  --output runs/gain-exploration-new
python runs/gain-exploration-v2/analyze.py
```

输出目录必须是新目录。本次最终结果在 `runs/gain-exploration-v2/validation.json` 和 48 份 `decision-*.jsonl`：每行含 held-out case/state/repetition、预提交特征、预测 G、门槛、实际 G、所选动作和实际 regret。`mechanisms.json` 记录基线复现、共同支持集合、简单规则一致性、改变动作的 case/state 及残余 regret 分解。早期 v1 保留，v2 增加简单规则对照；两次拟合结果一致。

专项测试覆盖四类模型的外层标签扰动不改变其拟合/预测/选择、重复测量不增加样本权重、case 级阈值、支持回退和强制重建保持；与长度、配置接线和 Prefill 测试合计 44 项通过，Ruff 通过。另有独立只读审查核对嵌套堆叠边界，未发现当前结果标签进入预测特征。这里只验证工程与上述隔离协议，不能消除 cohort 选择、历史 prior 来源和仅五个开发会话的局限。
