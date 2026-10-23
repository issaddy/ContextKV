# Native budget：定向补采与冻结quote反事实验证

2026-10-24。四组目标形状已完成真实H200补采，96个目标请求均满足事前形状及执行receipt合同；新stage artifact保留旧数据与冻结先验。对两组既有41轮开发轨迹的反事实重放，native-budget等待回调支持覆盖分别由56/72、60/72补到72/72，但共同支持样本的等待MAE仅下降约1.58ms、0.64ms。**已证明的是目标形状覆盖扩展，尚未证明闭环时延改善。**

本报告承接[原生预算观测与首次闭环](native_budget_validation.md)。待执行的新比较使用同一冻结源码/补丁下的两artifact对照；不能把以前 `3b852aa` 的真实闭环与当前 `a2cfff7` 运行直接拼成同源因果对比。

## 真实补采与未删除的旧数据

原始目录：`runs/remote-h200-native-budget-targeted/runs/native-budget-targeted-profile-v1/`，含四组Measurements、telemetry、manifest、runtime、配置、source-deployment和execution。GPU采样使用冻结 `a2cfff7645b8e69c9e8931c5bc5cf7954a5eda5a`、SGLang `3f36fc08331f64cd63c4fbbdf39cc402f38bb38b`及研究补丁 `e004ca96d62a5786f303c465da140513c7213f13ff023407991ec162ef82d92c`；GPU/host容量16,384/279,040 tokens，page64，FULL Python tree、write_through、cache模式、layer_first/kernel、TMA disabled。不要将后续本地拟合修复的commit改写为采样源码。

每组覆盖c1/c2、output128/512，各配置暖1批、正式3批。目标生成使用固定输出预算与ignore_eos；它们用于阶段标定，不是自然EOS质量任务。四组各24目标请求，分别为18正式、6暖机：

| 组 | 输入n | 总prefix P | GPU部分D | host load L | 新prefill U | exact/目标数 |
|---|---:|---:|---:|---:|---:|---:|
| host-6144-131 | 6275 | 6144 | 0 | 6144 | 131 | 24/24 |
| host-5184-1129 | 6313 | 5184 | 0 | 5184 | 1129 | 24/24 |
| host-1216-4204 | 5420 | 1216 | 0 | 1216 | 4204 | 24/24 |
| mixed-2176-3017 | 5193 | 2176 | 1216 | 960 | 3017 | 24/24 |

导出审核为 `runs/native-budget-targeted-review/audit-v1/audit.json`，`issues=[]`，记录每个请求的expected/observed n/P/D/L/U、receipt rid和完成状态。96目标全部exact，其中72正式、24暖机；前缀准备的一-token请求单列，不计入96目标或正式拟合样本。mixed是事前准备GPU1216+host960后执行的实际混合状态，没有把全host测量改标签。

合并保留旧552条正式数据和原36条prepared→actual tier漂移：concurrent-small 6条、concurrent-long-device 30条。没有删除漂移、按耗时挑样或用未来实际host load替换事前L。旧prepared合同与新mixed的prepared-and-execution-exact合同保持区分；新三个host组另经原始执行核查均exact。合并得到624正式请求、276批，排除的208暖机包含旧184和新24。旧数据不因新fixture通过而被追认为execution-exact。

## 首次拟合失败与先验来源修复

第一次拟合在来源校验处停止，错误为 `development outputs must match the report already bound to the base artifact`，日志 `audit-v1/fit.log`保留。原因是用已拟合stage artifact继续细化时，顶层measurement provenance已经描述stage profiling，而原经验输出分布的报告绑定保存在 `stage_model.development_output_lengths.source`；旧代码仍从前一种metadata位置查开发报告，错误地拒绝了原先验。

`809a3c185b1d3e62947895dae5df46b9b1c802ca`修复了这个继承路径：已有冻结stage分布时，完整复制其长度数组、来源路径、SHA和解释；若另外指定development report，仍需与该原SHA一致。它没有关闭报告绑定检查或从新采样的实际输出重新估计长度。49项相关合同测试通过；成功重拟合日志另存 `fit-refit-fix.log`，没有覆盖首次失败。

新artifact为 `runs/native-budget-targeted-review/audit-v1/cost-stage-targeted.json`：

- SHA256：`5ebf9e4cba04537ab9bef8e0ede8be5a64982cfd350abb5474825513425b742e`。
- 原v2 SHA256：`3d455c64f8e73c87451ddee6491b5ce9f7580ab7f77f9ce8b0f0d03b44ebdaf3`，原文件保持不变。
- 624条正式数据，forward独立feature cells由38增至46；output128/512为同形状的预算覆盖，不把每个输出配置假称为独立forward形状。
- 独立逐字段核验settings、decode_points、positive_prefill、经验长度分布及完整原来源、output_priors_source均与v2相同。原先验仍来自已完成的41条开发输出及其原报告SHA，而非本次目标生成。
- stage的等待标定范围仍为 `homogeneous_burst_only`，`online_heterogeneous_waiting_validation=false`。新的whole-cell OOF产物保存在 `stage-validation.json`；补采覆盖不能自动变成异质在线等待精度证明。

## 反事实重放保持原运行身份

只读入口：`runs/native-budget-targeted-review/counterfactual_replay.py`；完整结果 `counterfactual-v1.json`，简报 `counterfactual-summary.md`，合同核验 `counterfactual-contract-check.json`。

```bash
.venv/bin/python runs/native-budget-targeted-review/counterfactual_replay.py --execute
```

这里的execute只读取本地已保存数据，没有HTTP、tokenizer、生成、拟合或choose_plan。先调用原 `runs/native-budget-review/replay.py`，以**原v2**通过每个run manifest的artifact SHA和开发split检查，精确重现各自记录的等待函数；随后从原events重新读取同一quote行，核candidate ID、input hash和输入长度，再用新artifact作明确标记的反事实预测。新artifact从未被冒充为原运行artifact。

两组轨迹分别为旧等待 `native-budget-observation-v1` 和native-budget `native-budget-closed-loop-v1`：各41success、36 quotes/72候选臂。每组五个会话的首轮无quote，仍保留；只有实际执行且receipt输入相同的36臂有标签，未执行臂没有伪造实测等待。两个artifact对记录引擎的fingerprint/cache_config检查均为matching scope，但该检查不额外证明GPU、驱动或时序等同。

预测只使用quote状态、原先验与采样artifact，输出预算沿原run配置。标签是server pre-forward联合wall interval，不是纯排队或CUDA时间。附加合同实验确认：

1. 故意将新artifact传入原helper作为recorded，会继续被原SHA合同拒绝。
2. 只在内存把实际pre-forward标签增加12,345ms、实际output标签增加999 tokens，144个候选臂的预测全部不变。标签影响事后误差，不成为当前预测特征；原文件未修改。

## 同支持人群的误差与新增覆盖分开报告

下表均为**native-budget等待公式**，只切换stage artifact；共同样本使用同一quote和同一真实执行标签。

| 冻结开发轨迹 | 等待支持：全部臂 | 等待支持：执行臂 | 共同支持n | MAE v2→targeted，ms | actual超过upper情景 |
|---|---:|---:|---:|---:|---:|
| 旧等待闭环 | 56/72→72/72 | 28/36→36/36 | 28 | 600.678→599.096 | 12→12 |
| native-budget闭环 | 60/72→72/72 | 30/36→36/36 | 30 | 438.787→438.144 | 18→15 |

没有失去已支持的等待回调样本；共同样本误差变化很小，不能宣称主要等待瓶颈已解决。新增支持的8个旧轨迹执行样本，targeted MAE为1,560.900ms；新增6个native-budget轨迹执行样本为516.720ms。它们作为新增覆盖人群单列，**不能与原28/30条的均值直接比较改善**。新artifact全部36条MAE分别812.831ms、451.240ms也属于更大的分母，不是直接的退化或改进对照。

逐会话共同样本有改善也有回归。按JSON会话排序，旧轨迹targeted−v2 MAE约为−0.025、−0.008、+3.549、+0.014、−9.723ms；新轨迹约为−0.033、−0.046、−4.295、+0.015、+0.421ms。upper只是较大风险情景，不是置信界，超过它的次数减少不能代替均值精度与闭环收益。

保留的empirical-service等待公式始终支持各36执行样本，MAE分别739.864→739.525ms、440.059→443.263ms。该结果用于区分“补充阶段采样”与“更换等待公式”，不将二者合称一个改进。

另有一个尚未支持的idle setup：旧轨迹 `b461…` 第7轮未执行APPEND，输入14,137 tokens，在两个artifact下均不支持；所以72/72是等待callback支持，**不是全plan已校准或必然eligible**。两个轨迹的全部36个执行臂setup均支持，此剩余项不改变上表共同执行样本分母。

## 同源闭环比较：尚无结果

下一步已安排在同一冻结remote `a2cfff7` / e004补丁下，两个artifact各41轮、两次反转执行顺序，共164轮。此时应保持数据、策略、预算、引擎配置和公共材料条件一致，并核对实际quotes、动作、等待、完成时间、回退及资源。**截至本报告，164轮结果尚未导入或分析。**

这轮才用于检查新增支持是否改变动作并带来净收益；现有离线重放没有模拟反事实闭环，也不预测时延改善。之前 `3b852aa` 的两次真实闭环继续作为冻结输入来源和历史开发证据，不能与当前 `a2cfff7` 的新运行混称“同源码两artifact比较”。即使新164轮完成，定向形状来自已观察开发quotes这一事实仍须保留；独立test与质量验收另行完成。
