# 候选长度与条件误差的 OOF 决策验证

本次使用既有 15 个冻结 dev case、5 个会话、3 种缓存状态、每状态 2 次重复，共 90 个真实配对观测。没有新增生成或引擎请求。长度预测采用按会话留出的均值；P90 与误差也来自各外层 fold 的训练会话内部 OOF。全量拟合 artifact 没有参与这里的选择。

`profiling.length_decisions` 将每个冻结 candidate 绑定到其 OOF 预测，通过既有 `CostModel.predict_native`、`rescore_pairs` 和真正的 planner 执行选择。保留报价、回答、输出长度、真实时延和失败行。报告中的当前观测只用于计算 regret；更改当前答案、时延或 validation 的标签不改变预测/选择，已有合同测试。shared/action-only/candidate/paired-delta 均使用相同时间模型、真实报价及决策参数。

这里的压力为隔离 fixture 的 c1，`prediction_control_ms=0`，`delta_abs=0`、`delta_rel=.05`、`z=1`、`min_stable_turns=2`。它不是原始 P2/c4 到达顺序回放，也不是远端采用 433 ms 门槛的工程运行。冻结的 433/512 全局历史先验由各组共同复用；LOSO 隔离本轮残差拟合，没有重新验证这份旧全局先验的构造。

## 八组真实计划器结果

全部组均 90/90 可比较，82/90 位于时间标定支持域，另外 8 次走原固定策略回退。regret 是已选择动作的真实 server elapsed 减去该 pair 两臂较小值，仅作离线诊断；没有把真实较快动作或真实成本提供为预测特征。每会话都有 18 个观测，因此本次等会话平均与全 pair 平均一致。

| Prefill 误差 | OOF 长度 | 成本切换次数 | 滞回次数 | 成本差符号正确率（82 个校准 pair） | 等会话平均 regret |
|---|---|---:|---:|---:|---:|
| global | shared | 0 | 82 | 56.10% | 437.05 ms |
| global | action-only | 0 | 82 | 56.10% | 437.05 ms |
| global | candidate | 0 | 82 | 56.10% | 437.05 ms |
| global | paired-delta | 0 | 82 | 60.98% | 437.05 ms |
| conditional | shared | 6 | 76 | 56.10% | 366.55 ms |
| conditional | action-only | 18 | 64 | 56.10% | 342.67 ms |
| conditional | candidate | 14 | 68 | 56.10% | 270.30 ms |
| conditional | paired-delta | 22 | 60 | 60.98% | 280.36 ms |

这些改动相对本组原始 shared/global 选择有改善，但尚未胜过固定 REBUILD：它在相同配对观测上的平均 regret 为 **129.14 ms**，固定 APPEND 为 437.05 ms。两项固定策略数字也不含质量约束，不能直接当作完整业务运行结论。

候选模型的长度 arm MAE 从 shared 的 167.84 tokens 降至 131.06，但长度差 MAE 为 119.36，略差于 shared 的 115.82；paired-delta 的差值 MAE 为 113.69。更好的单臂长度或差值 MAE并不自动产生最低选择 regret，必须通过同一 planner 检验。

## 每会话结果

下表均为 conditional 误差下的平均 regret（ms）；会话 ID 使用可唯一匹配的前缀，完整 ID 在 JSON 中保存。

| 会话 | 固定 REBUILD | shared | action-only | candidate | paired-delta |
|---|---:|---:|---:|---:|---:|
| `1c041ce4` | 0.00 | 390.06 | 384.02 | 390.06 | 384.02 |
| `b4614daa` | 146.46 | 654.57 | 450.17 | 246.45 | 246.45 |
| `c01c8cf1` | 156.10 | 358.35 | 270.24 | 270.24 | 270.24 |
| `c6c3b02c` | 145.87 | 208.68 | 279.99 | 223.65 | 279.99 |
| `d5b1e735` | 197.28 | 221.08 | 328.94 | 221.08 | 221.08 |

改进并不均匀：candidate 的主要改善来自 `b4614daa`，`c6c3b02c` 则比 conditional/shared 更差。因此不把总体平均下降解释为所有会话都受益。

## 剩余 regret 的机制

对 conditional/candidate 的错误选择，按固定优先级分类：先归入未校准回退，再区分错误 REBUILD、非正预测收益、margin/稳定轮数阻挡。它是可复核的规则分类，不是对多个相互作用因素的独立因果归因。

| 类别 | pair 数 | 对全部 90 pair 平均 regret 的贡献 |
|---|---:|---:|
| 未校准回退后选错 | 4 | 32.61 ms |
| 预测收益非正而实际 REBUILD 更快 | 4 | 44.22 ms |
| 预测收益为正而错误切换 REBUILD | 2 | 2.99 ms |
| 仅 margin 阻挡 | 4 | 1.21 ms |
| 仅稳定轮数阻挡 | 6 | 54.26 ms |
| margin 与稳定轮数同时阻挡 | 20 | 135.01 ms |

50 个 pair 已选择较快动作；同一 case 的重复/缓存状态不是独立样本。三个具体误选说明了问题：

- case `fdf19ecc`、host-only：预测总收益 −2.93 ms，实测 REBUILD 快 997.52 ms。Prefill 差预测 +3.83 ms、实测 −0.31 ms；主要遗漏在 Decode 差（预测 −6.76 ms，实测 +997.83 ms），不能靠继续减小 Prefill 误差解决。
- case `11012108`、cold：Prefill 差预测 +62.20 ms、实测 +62.63 ms，已相近；Decode 差预测 +168.48 ms、实测 −200.05 ms，导致错误 REBUILD，regret 137.42 ms。
- case `de788b0d`、cold：预测收益 160.77 ms 已超过 margin 145.93 ms，但前次重建后只过 1 轮，稳定门槛阻挡；实测收益 965.72 ms。这展示稳定门槛的机会成本，但不证明全会话执行中取消防抖必然更好。

全部 82 个校准 candidate pair 的当前 margin 都由 `.05 × APPEND 总预测成本` 主导，已经不是两臂 Prefill 误差之和主导。按原 OOF 预测做诊断性消融，得到：

| 长度模型 | 原设置 | 仅去 margin | 仅去稳定轮数限制 | 两者都去 |
|---|---:|---:|---:|---:|
| shared | 366.55 | 226.08 | 376.29 | 167.52 |
| action-only | 342.67 | 301.93 | 266.06 | 211.92 |
| candidate | 270.30 | 280.36 | 293.88 | 190.87 |
| paired-delta | 280.36 | 236.14 | 166.77 | 137.54 |

表中仅去 margin 是 `delta_abs=delta_rel=z=0` 的机制诊断，**没有用于修改默认配置或采用零门槛**。candidate 单独去掉任一门槛反而更差；二者的作用相互依赖。即使最激进的 paired-delta 消融仍略差于固定 REBUILD 129.14 ms，且静态配对无法衡量长期重建频率和缓存状态传播。

8 次支持域回退有具体缺口：两个 APPEND 候选在 gpu-ready/host-only 下的 `(P,U)` 分别为 `(6336,113)`、`(5632,155)`，各重复两次；`U` 低于现有 Prefill 标定最小值 192。native quote 本身并不 uncertain。这里可定向补少量短 suffix、高前缀的 GPU/host 标定单元，而不放宽支持域判断。

## 下一步可执行改进

1. **直接校准成对收益和乐观误差。** 对外层训练会话构造 `G = T_APPEND − T_REBUILD` 标签，输入只用两候选的已有预测 Prefill/Decode 差、长度差特征、真实报价及已提交历史。用内层会话 OOF 拟合收缩的收益修正，再对 `G_hat − G` 的正侧残差做校准。当前真实成本只作为离线标签/损失，不作为运行时特征；同样的方法必须用于 shared/action-only/candidate/paired-delta，防止只给新方法额外优势。
2. **把门槛目标改为明确的误切换损失，而不是继续手动减小 z。** 可以分别报告错误 REBUILD 的额外时延与错失 REBUILD 的可得收益，在训练会话中选择折中并在外层验证。[Conformal Risk Control](https://research.google/pubs/conformal-risk-control/) 给出以单调损失而非单纯覆盖率为校准对象的思路。但总 regret 对阈值并不单调；不能直接套用其保证。可先校准有界的单侧误切换损失，同时报告错失收益，按会话分组验证，并保留稳定轮数/软切换成本的匹配消融。
3. **检验两臂共享误差是否真的抵消。** 已有同 cache fixture 的配对允许估计 `e_APPEND − e_REBUILD`，无需把单臂绝对误差直接相加。不过本次 c1 first-token 残差在 45 个 case/state 均值上的相关系数仅 0.06；先平均两次重复、再取每 case 跨状态最大误差后，成对绝对误差 P95 为 12.44 ms，两臂绝对误差之和 P95 为 13.51 ms，抵消幅度有限。不能预先假定强相关，更不能直接外推到 c4。配对控制与协方差影响比较误差的原则可参照 [Kleijnen 的共同随机数实验分析](https://pubsonline.informs.org/doi/pdf/10.1287/mnsc.34.1.65)，本项目没有复现其随机数耦合算法。
4. **优先补明确缺失单元。** 在专用 profiler 上增加接近上述两种真实形状、suffix 小于 192 的 GPU/host 单元，保持固定模型、布局、并发和 warmup/repetition 协议；先恢复被拒绝点的真实覆盖，再重新比较 planner，不用外推支持域掩盖缺测。

上述收益校准可直接复用现有 90 pair 和 session-LOSO 工具进行下一轮探索；参数选择必须留在内层，外层不再挑参数。只有 5 个会话，不能声称可靠总体尾部、独立最终测试或质量非劣效。Prefill 条件误差仍是经验分位数，不是概率保证。

## 长度模型的运行适用范围

启动校验包含模型/tokenizer/template fingerprint、生成预算和 feature schema；运行时输入长度超出训练范围会恢复 shared prior。它不保证 system prompt、布局或文档分布迁移的预测准确性，`source_configuration` 只是记录来源。Q0 上训练的 artifact 不能将 Q1/Q2 自动视为已校准；更换提示词、layout 或领域后应重新评估/训练。当前真实工程集成保持 Q0 与相同 context 配置，且标为 full-fit dev 内工程验证，不与这里的 OOF 结果混用。`CostEstimate.calibrated` 继续表示时间表/报价支持，不是长度或质量保证。

## 实际 full-fit 工程集成，与 OOF 分开

远端随后以提交 `ea08c19`、full-fit `candidate.json` 和 positive/conditional 时间 artifact 执行一次 Q0/P2/c4、1024 输出上限的 41 轮 Cloud dev 运行。它的绝对切换门槛为 433.09 ms，和上文隔离配对的 c1、control=0、delta_abs=0 条件不同。这次只验证完整接线与运行，不作为 OOF 效果或未见样本性能证据。

本地复核了两份 artifact 的原始 SHA256，均与运行 manifest 精确一致；按源码集合规则重算 `ea08c19` 的 controller hash 也匹配。cold 记录中 GPU/host unique tokens 均为零、无活动/排队请求，reset 后身份与运行 manifest 对应。没有由配置 hash 推断 GPU 型号或驱动等价。

| 集成观测 | 结果 |
|---|---|
| 完成与失败 | 41/41 success，0 failed |
| 动作 | 28 APPEND，13 REBUILD |
| 决策原因 | 5 initial/only-feasible，23 hysteresis，13 uncalibrated fallback，0 cost-saving |
| 两候选轮次 | 36 轮；其中 34 轮两臂预测输出均值不同 |
| 长度模型实际分支 | 55/72 个预测使用 candidate；17/72 个超出训练输入范围，恢复 shared prior |
| 时间支持 | 72 个预测臂中 58 个 calibrated；23/36 轮两臂均支持 |
| 规划压力档位 | c1：8 轮，c3：10 轮，c4：23 轮 |
| 输入/输出范围 | 2460–16192 input tokens；75–1024 output tokens，1 轮截断 |
| 生命周期 | 13 次退役请求均 ACK，其中 8 次替代退役、5 次会话关闭，无 retry |
| Control / TTFT P95 | 235.33 / 363.46 ms |
| Complete / runner 完整轮次 P95 | 5073.20 / 5085.93 ms |

候选长度已产生不同的两臂长度先验，但本次没有触发成本切换；不能把上文离线 c1 的切换收益移植到这次较高压力、较高绝对门槛运行。退役 ACK 仅表示协议完成，不被当作物理 token 已释放的证明。compact 省略的长度诊断另从原始预测导出到 `runs/remote-h200-retrieval/runs/candidate-length-hardware/length-diagnostics.json`：72 个预测中 55 个明确标记 `candidate/in_range=true`，17 个为 `candidate:shared_range_fallback`。后者 5 个短于训练最小值 3433，12 个长于最大值 11217；36 个两候选轮次中 16 轮至少一臂回退。每臂都保留 331.2 tokens 的经验误差诊断，未把它加入切换门槛。长度范围回退与上表 GPU 时间未校准回退是不同机制，不能混计。

证据目录为 `runs/remote-h200-length/runs/candidate-length-hardware/`，包含 compact、cold、config 和 manifest；本地标量及身份核对保存在 `runs/output-length-exploration-v2/hardware-integration-summary.json`。该工程运行的回答、历史、动作与旧 matrix 不必相同，没有进行质量等价匹配，不据此宣称相对旧 matrix 的性能提升或质量非劣效。

## 复现与证据

```bash
python -m profiling.length_decisions \
  --cases runs/remote-h200-expanded/decision-dev-stratified/cases.jsonl \
  --pairs runs/remote-h200-expanded/decision-dev-stratified/pairs-lookup.jsonl \
  --validation runs/output-length-exploration-v2/validation.json \
  --config runs/remote-h200-expanded/decision-dev-stratified/config.yaml \
  --cost-artifact runs/remote-h200-expanded/cost-dev-expanded-positive-conditional.json \
  --output runs/output-length-exploration-v2/decisions
PYTHONPATH=. python runs/output-length-exploration-v2/analyze_decision_mechanisms.py
```

输出目录必须是新目录。主要结果为 `decisions/summary.json`、八份 `decision-*.jsonl`、`decision-mechanisms.json` 与 `paired-prefill-residual-diagnostic.json`。工具在启动时核对来源、完整 case/session 覆盖及 OOF mean/P90/error schema；逐候选只查预绑定预测，不重新 hash，也不访问当前标签。全量拟合模型或缺失案例会被拒绝。
