# 机制审查与修复：正确证据、有限阅读、有净收益的保留

## 结论与范围

本次修复以 `0431d40137f6e2afbf56060ad35c49c4d8c2bad1` 为审查基线，针对
[2026-11-12 公开评测](final_results.md)暴露的问题。固定 Qwen3-32B 和既有评分定义，
没有改写历史成绩，也没有将局部续跑混入主队列。

原结果说明长文阅读有增加可回答信息的潜力，但当前机制未把这部分信息价值转化为稳定的
完整响应优势。MultiDoc2Dial 约 2.46 分和 Loong 8.60 分的增益，不能抵消其他任务回退
及显著增加的耗时。Loong 约 30 倍 GPU GiB·s 主要反映在线时长，不能解释为峰值显存增加 30 倍。

本分支交付可复测的机制修复。CPU 回归证明来源绑定、状态迁移、费用保留、调度有界和
软配额行为，不能证明真实模型质量、P95 或目标场景最佳。本次环境没有正式 GPU 服务、
Qwen 权重和完整实验资产，最终验收仍需在实验环境完成。

## 四条机制约束

1. 数值必须与字段、实体、时期、原始单位及来源区间绑定。原文出现过某个数字，不代表它属于目标字段。
2. 完整统计仍要求必要来源覆盖和计算；一个统计 Need 不应强迫无关 Need 全部读完。重试也必须作为新动作报价和预算准入。
3. 缓存保留需要在 Native 命中、回载和重算基准上，覆盖控制、驻留及竞争影响；风险 token 数不能代替净收益。
4. 物理调用完成、证据完成、答案交付是不同状态。失败不能消失在费用、训练或会话调度中。

## 缺陷与修改

| 问题 | 原失效路径 | 修改后行为 | 主要代码 |
|---|---|---|---|
| 相邻字段错配 | 多数字摘录歧义时接受模型值 | 原文字段唯一时绑定原值，否则保留缺口 | `controller/reading.py` |
| 后一窗口吞掉前一实体缺口 | numeric support 按整个 Need 清除早期 gap | 数值完成由覆盖与计算证明；repair 精确关闭字段/实体/时期缺口 | `controller/reading.py`、`controller/read_recovery.py` |
| 截断反复恢复失败 | 已付 RESULT_READY 继续解析相同截断输出 | 失败留账；更小子 READ 重新准入；旧调用不重复生成 | `controller/execution.py`、`controller/rolling_policy.py` |
| 已读完却没有动作 | REDUCE、GAP、ANSWER 同时被禁止 | 原来源局部字段修复；耗尽后明确部分答复/无法确定 | `controller/read_recovery.py` |
| 无关阅读阻塞统计 | 前置 READ 使用全计划集合 | 依 Need/共享数值范围隔离，真实覆盖检查继续保留 | `controller/reading.py`、`controller/context_budget.py` |
| 整条剩余路线反复报价 | 每轮编译所有待执行 READ | 默认只精确报价未来 2 个 READ；完整预算不删，执行前重新报价 | `controller/rolling_policy.py` |
| 失败成本训练偏差 | 失败质量记零，成本头却只用成功分支 | 使用具有完整实测费用的全部已终止分支 | `training/plan_gain_completion.py` |
| 弱动作资产默认部署 | 资产可加载即参与动作选择 | joint observed 默认验证开发证据；不通过明确回退规则 | `controller/action_admission.py`、`controller/plan_profiles.py` |
| 无定价保护缓存 | next_use 用风险 token 当效用 | 无有效 paired-Native 价格即 Native；净值必须覆盖控制费用 | `controller/prefix_retention.py` |
| 并发保护超额 | 两会话按同一旧快照分别提交 | Native 注册 owner 前原子检查逻辑因果页并集 | `engine_patches/contextkv_runtime.py` |
| P1 一题失败阻塞后题 | 异常边界包住整段 conversation | P1 逐题隔离并确认清理；P3 保留真实已提交历史依赖 | `bench/study.py`、`bench/study_workloads.py` |
| 失败 token 丢失 | 仅正常 response 汇总费用 | 持久化 Native/step/auxiliary 账本结算；未知值及已知下界分开 | `bench/study_costs.py`、`bench/result_records.py` |

### 数值正确性

对于“甲公司：流动资产合计 100 万元；总资产 500 万元；货币资金 20 万元”，
模型给出 `r=流动资产合计, v=500` 时，解析器现在绑定为源字段的 `100 万元`。
字段不存在、引用错误来源、跨期多值或单位无来源依据时，拒绝事实并保留 `numeric_grounding_failed`。

这是保守的来源校验。多期表格没有明确列对应时仍保留缺口，字段同义词及一般问题的
语义定位仍需要任务模型/numeric selector。严格校验可能降低召回，因此“拒绝了错误事实”
不能直接推出 benchmark 正确率提高。明确的零值会保留，缺失值不转为零。

### READ 恢复与终态

`RecoverableReadError` 只用于已保存真实 receipt 的 READ 截断/结构化解析错误。
指纹不一致、receipt 不匹配、Native 状态未知及 REDUCE 截断不被伪装成安全重读。

失败父调用在物理账本中为 COMPLETE，但 claims/covered_units/support 为空，实际费用保留。
子 READ 使用新身份，先保存计划再提交。回归覆盖 RESULT_READY 和失败 COMPLETE 两个重启点，
不会重新支付父请求。子结果的原始来源区间并集才建立覆盖，父结果始终没有虚假覆盖。

分包最多两层。优先拆多个来源单元；单个长单元只在行、句、记录边界切分，不按字符中点
切断一个数值记录。短表头、实体前缀及已声明依赖作为独立原始来源区间带入。
所有子任务仍受 token、step、窗口与 Native 容量检查。

字段修复在必要阅读完成后执行，默认每 Need 最多两次，每次最多 4 个局部正文单元，正文
窗口为 1536 字符；依赖另受窗口和请求预算限制。repair_gaps 保留条件、字段、实体和时期。
只有匹配缺口且通过来源校验的事实才能关闭对应缺口。

恢复/继续阅读预算耗尽后，`ContextPlan.terminal_gaps` 允许明确部分答复/IDK。
它不使严格 evidence sufficiency、source coverage 或统计完整性检查通过，也不生成零值。
ANSWER 去除相关已派生完整结论并明确说明无法确定的点。交付证据不足说明不等于答对，
仍由原 benchmark 评分器裁定。

### 控制与保留费用

固定 `quote_lookahead=2`，50 个独立 READ 的路线，READ 报价条目计数由
`50+49+…+1=1275` 降为 `3×48+2+1=147`。这是相同候选形状下的确定性控制工作量
对照，不能换算成 Qwen 耗时；ANSWER 报价与其他控制另计。

远端步骤和预算没有删除；执行时会获得新 Native 报价。请求内复用相同消息的精确长度，
限制保存的完整 token 数组数量。KV 命中、Native 负载和报价不被缓存成过期事实。
无有效保留模型时，不再编译仅为未来保留而生成的 ANSWER 来源投影。

next_use 与 paired_value 都要求有效 pricing_asset 才能新建/续租。缺少资产、特征桶或实测
长度范围时使用 Native。保留收益计入 owner 控制、一次集合固定控制费用、竞争影响及配置
的 GPU/Host 驻留价格，不用 risk token 冒充毫秒，也不重复扣除 Native 已包含的缓存收益。

保护预算是受保护逻辑因果页的软配额，不是 GPU 物理容量上限。必须部署同步的 SGLang
overlay 才有跨进程原子保障。配额拒绝记为 UNRETAINED 并继续 Native；共享已有页、释放
以及缩小额度后的幂等操作仍可执行。

## 学习资产准入

joint + observed_actions 默认 `action_policy: validated`。显式 learned 用于研究消融，
仍严格校验 reader、输入协议和训练绑定；显式 rule 用于规则对照。最终 reason 和观察记录
报告实际使用的策略及拒绝原因，不能把规则回退记成学习收益。

| 默认工程门槛 | 值 |
|---|---:|
| 与训练隔离的开发会话 | 至少 5 |
| 有 direct 配对参考的开发状态 | 至少 30 |
| READ 收益符号样本 | 至少 20 |
| 其他已训练动作/表示分支样本 | 每类至少 10 |
| 收益符号一致率 | 每类至少 60% |
| 已选动作相对 direct 的平均净目标增益 | 严格大于 0 |

阈值可通过 validation_thresholds 声明。这是工程准入，不替代置信区间和独立测试。
现有 `−0.001167` 净目标及 `38.46%` READ 符号一致率不应通过；GAP、REDUCE 和表示
分支也需独立覆盖，不能由总体准确率掩盖。

监督输入协议同步升级，显式绑定 grounding、Need 依赖、恢复、终态和报价语义。
同 reader/profile 的不兼容旧资产在 validated 模式只读取审计元数据，实际回退规则；
不执行旧权重，不自动迁移 metadata。reader/profile 不匹配仍报错。

需重新采集当前协议的 frozen-state 分支。新成本目标为 `terminal_attempt_incurred_cost_v2`，
纳入具有完整实测费用的成功和终止失败，共用 direct 按 record ID 去重。未知、非有限及
不合法费用不补零。READ 次数和延迟 residual 校准同口径。该标签仍不含共同已付状态、
共同准备及响应结束后的清理；正式资源积分继续覆盖完整在线窗口。

`fit --dev` 将开发报告、asset ID、会话分区、数据 digest 与成本目标绑定到资产。
当前 dev 同时参与延迟 residual 校准，资产明确记录这一点，报告不能称作独立 test 成绩。

## 验证与验收

安装项目 evaluation / plan-gain-training 依赖后运行：

```bash
make test
python3 -m compileall -q controller adapters bench training engine_patches
git diff --check
```

回归包含来源错配、多实体/多期歧义、表头依赖、失败账本、两个重启切点、真实 rolling.run
拆包与终态、Need 隔离、报价条目数、Native 并发软配额、P1/P3 调度、终态费用去重和
实际 CPU 成本拟合。模型输出和传输使用受控替身，不代替 GPU 运行。

正式验收沿用[运行手册](final_runbook.md)：

1. 更新 overlay，冻结同一权重、tokenizer、采样、窗口、资源配额、cohort、到达计划和 code SHA。
2. 重放原失败/错字段用例，再重跑各 benchmark 原 50 题、并发 4 的三方法完整队列。
3. 分别报告交付、IDK、错误答案和协议失败；主质量继续失败记零，P1/P3 分开。
4. 建议至少 3 次队列重复，交错运行方法；质量同题配对并按 conversation 聚类，完整队列与已交付 P95 分开。
5. 单独比较 validated/rule/显式 learned、Native/排列/带定价保留及 fact reuse。记录实际回退策略。
6. 包含排队、检索、辅助、阅读、归约、编译、报价、保留控制及失败费用；未知 token 保留为空并报告已知下界。

验收顺序是：实现可靠 → 阅读/保留的完整链边际净收益为正 → 按原目标证明质量 +5 分
或同质量完整响应 P95 −15%，并同时说明交付和资源代价。三者不能互相替代。

## 尚未完成的研究目标

- 完整长文统计仍可能需要大量真实阅读；本次去掉错误强制依赖与重复控制，不宣称消除
  所有全范围阅读。进一步少读需要结构感知的表格/记录定位和可验证覆盖。
- VersionQA 的检索版本、版本对应与差异综合需要分别标注和监督，不能靠数字校验一次解决。
- 查询 LoRA、任务头与支持模型的覆盖不足仍在；本次没有新 GPU 训练权重，也没有据现有结果替换 teacher。
- 50 题中一题差异就是 2 个百分点，需要配对结果与重复评测判断稳定性。
- GPU 正式复测前，不能声称实现端到端加速、恢复所有历史失败题或在四个 benchmark 达到最佳。
