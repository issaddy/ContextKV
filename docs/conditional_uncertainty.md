# 可选的并发条件误差

positive prefill 原有全局 OOF P99 为 472.57 ms。它来自 44 个独立特征单元，每个单元先取其 OOF 请求绝对误差最大值，再计算经验分位数。同一 c4 单元即使包含更多请求也只有一票；但全局聚合仍会把 c4 的波动带到 c1 决策。

本次改动保留原 NNLS 系数、OOF 拆分、点预测、支持域及 `z`，只新增可选的 `uncertainty_mode: conditional`。默认仍为 global；旧 artifact 缺少该字段时行为不变。配置的 `uncertainty_mode: null` 沿用 artifact 保存的模式，显式 global 可以覆盖 conditional artifact。

## 固定分层与回退

分层键仅使用决策前可观察的并发和 native quote 预计实际恢复的 host tokens 是否大于零。依次选择：

1. 相同并发、相同 host 状态，至少 20 个独立 feature cells。
2. 相同并发的聚合层，至少 20 个 cells，且其观测包含目标 host 状态。
3. 原全局误差。

细层的 20 个单元门槛固定，不根据 90 个配对案例的收益调节；重复 OOF 行或一个单元内的请求数不会增加独立 cell 数。没有对应层的 c2/c3 使用全局误差，不套用 c1。host 成本项消融时，误差层仍由真实 quote 的恢复状态选择。当前数据情况如下：

| 层 | 独立 cells | 经验 P99 | 是否可用 |
|---|---:|---:|---|
| c1，无 host 恢复 | 13 | 14.13 ms | 不足，退到 c1 |
| c1，有 host 恢复 | 9 | 13.70 ms | 不足，退到 c1 |
| c1 聚合 | 22 | 14.92 ms | 可用 |
| c4，无 host 恢复 | 13 | 544.37 ms | 不足，退到 c4 |
| c4，有 host 恢复 | 9 | 320.03 ms | 不足，退到 c4 |
| c4 聚合 | 22 | 526.12 ms | 可用 |

c4 的条件误差可以比全局值更大；这不是统一降低安全系数。20 cells 只是启用经验分层的最低数据量，不等于证明 99% 覆盖。整组留出的原则与 [scikit-learn grouped cross-validation 文档](https://scikit-learn.org/stable/modules/cross_validation.html#cross-validation-iterators-for-grouped-data) 一致：相关重复样本不能同时进入训练和验证。分组条件覆盖与总体覆盖是不同问题，[Duchi 的 Sample-Conditional Coverage 研究](https://proceedings.neurips.cc/paper_files/paper/2025/hash/7d83f9e9462c417e208d55e83a5058a8-Abstract-Conference.html) 也专门区分了这些目标。本实现没有增加 conformal 校准或声称相应覆盖保证，只有可审计的经验误差表。

## 从旧 OOF 直接构建与复算

```bash
python -m profiling.positive_prefill \
  --input-artifact runs/remote-h200-expanded/cost-dev-expanded-positive.json \
  --output runs/remote-h200-expanded/cost-dev-expanded-positive-conditional.json
python -m profiling.decision_pairs rescore \
  --cases runs/remote-h200-expanded/decision-dev-stratified/cases.jsonl \
  --input runs/remote-h200-expanded/decision-dev-stratified/pairs-lookup.jsonl \
  --config runs/remote-h200-expanded/decision-dev-stratified/config.yaml \
  --cost-artifact runs/remote-h200-expanded/cost-dev-expanded-positive-conditional.json \
  --output runs/remote-h200-expanded/decision-dev-stratified/pairs-positive-conditional.jsonl
```

第一步只读取保存的 whole-feature-cell OOF 记录，不重新拟合系数，不读取配对生成结果；保留原 artifact。第二步复用既有报价和冻结输入，不调用引擎。

90 个 dev pair 的所有点预测、支持标记、实测输出和时延逐字段保持一致，只有误差及其产生的选择变化。覆盖仍为 82/90，校准 pair 上的预测成本差符号准确率仍为 56.1%。选择由 82 次滞回＋8 次回退，变为 76 次滞回＋6 次成本切换＋8 次回退。新增切换来自 **3 个 cold 案例各 2 次重复**，六次都是实测更快的 REBUILD；不能计作六个独立场景。等会话平均 regret 从 437.05 ms 降为 366.55 ms，减少 70.50 ms。逐 pair 记录在 `conditional-uncertainty-diagnostic.json` 及 `pairs-positive-conditional.*`。

这是已查看过的相同开发数据上的探索性改善，不是独立验证，也没有质量结论。它只解决 c1 被 c4 全局误差压制的一部分切换；点预测偏差及候选输出长度差异仍需单独处理。

## 与可选输出长度模型的组合边界

`CostModel` 可接受启动时加载的 `output_length_model`。该模型只读取候选、已提交 session 和共享历史先验：预测均值影响期望 Decode 成本，P90 影响风险成本和容量风险；`output_length_error_tokens`、模型类别和训练范围标记随每个 CostEstimate 保存。

首版不把长度误差乘 TPOT 后叠加到两臂的切换门槛，因为这与原共享先验不计长度残差的对照不公平，也会混淆两种误差定义。越界时恢复 shared prior 并记录 fallback，不改变原 GPU 时间/报价支持的 `calibrated` 语义。这个字段不代表输出长度准确或答案质量保证。若后续使用长度差的成对误差，须独立验证并同样校准 shared/action-only 对照。
