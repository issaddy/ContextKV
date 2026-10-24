# Native budget：定向补采与冻结quote反事实验证

> 归档记录：保留当时的设计、结果与复用依据；其中“下一步”和执行计划不代表当前任务。当前工作仅以[实现清单](../implementation_plan.md)为准。

2026-10-24。四组目标形状已完成真实H200补采，96个目标请求均满足事前形状及执行receipt合同；新stage artifact保留旧数据与冻结先验。对两组既有41轮开发轨迹的反事实重放，native-budget等待回调支持覆盖分别由56/72、60/72补到72/72，但共同支持样本的等待MAE仅下降约1.58ms、0.64ms。**已证明的是目标形状覆盖扩展，尚未证明稳定的闭环时延改善。**

本报告承接[原生预算观测与首次闭环](native_budget_validation.md)。后续164轮比较使用同一冻结源码/补丁下的两artifact对照；不能把以前 `3b852aa` 的真实闭环与当前 `a2cfff7` 运行直接拼成同源因果对比。

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

## 同源闭环比较设计与解释边界

后续在同一冻结remote `a2cfff7` / e004补丁下完成两个artifact各41轮、两次反转执行顺序，共164轮。设计保持数据、策略、预算、引擎配置和公共材料条件一致，并核对实际quotes、动作、等待、完成时间、回退及资源。完整原始结果与独立核验如下；此前反事实结果保留原分母和原解释。

这轮用于检查新增支持是否改变动作并带来净收益；前面的离线重放没有模拟反事实闭环。之前 `3b852aa` 的两次真实闭环继续作为冻结输入来源和历史开发证据，没有进入下述配对。定向形状来自已观察开发quotes这一事实仍须保留；独立test与质量验收另行完成。

## 同源164轮：回退减少，尚无稳定加速

原始目录为 `runs/remote-h200-native-budget-targeted-closed-loop/runs/native-budget-targeted-closed-loop-v1/`；58文件archive SHA为 `aa354f1f4788120c4579e104cd61541be97d82f6ec7e9d48413912c8a880efcb`。独立只读脚本 `runs/native-budget-targeted-closed-loop-review/analyze.py`与派生 `summary.json`保留四组全量结果、164个SQLite阶段、每轮input/history/receipt核查、候选估计、会话统计和轨迹对比。脚本不拟合、不调用模型或引擎。

### 完整性与相同配置

部署清单273个tracked文件与 `a2cfff7` Git树逐项匹配，patch SHA为上述e004；四组实际controller源码hash均为 `b8882bd9ce1d9a19ca0ddd271b201cea05bb6c61672769ee58683a8037860a26`。模型/指纹、缓存配置、runtime线程设置、数据及split、P2/c4/seed42/Q0/1024、planner与ablation等配置均相同。完整配置只有 `cost_model.artifact` 和输出 `state_dir`不同；各manifest的artifact SHA与已导入原文件相符。r1顺序为v2→targeted，r2为targeted→v2。

四组各41 success，无unfinished/timeout；四次独立KV cold hook均返回成功，开始时GPU/host unique tokens都是0，实际池16,384/279,040。KV冷态不等于全部进程状态相同；这里仍只具有两次反转顺序的开发重复。

164个阶段均COMMITTED，receipt、task/attempt、动作、输入/输出数量、server TTFT与benchmark一致。全部audit输入IDs的既有token hash匹配实际receipt，输出IDs与答案匹配；每轮actual history等于该方法此前真实输出。另用固定Qwen tokenizer解码实际输入，逐项核查完整QA角色/文字、当前Question和visible正文span，164条全部相符。解码核查没有将raw生成token重新编码并冒称是同一token化过程；执行身份由原IDs与receipt绑定。

### 完整响应、会话权重与尾延迟

完整pipeline时长定义为 `max(t_response_done,t_commit)−benchmark t_arrive`，包含该轮入口至响应/提交的成本。下表的“会话均值”先在各会话内平均，再对五个会话等权；“会话P95均值”先计算各会话P95再等权平均，**不是**把41轮混合后的P95。

| 重复 / artifact | 等权会话均值 ms | 会话P95均值 ms | 混合41轮P95 ms | 混合41轮P99 ms | cohort wall s |
|---|---:|---:|---:|---:|---:|
| r1 / v2 | 5010.265 | 8138.964 | 10181.312 | 10238.536 | 71.274 |
| r1 / targeted | 5285.112 | 8727.952 | 9444.109 | 10721.999 | 76.165 |
| r2 / v2 | 5213.519 | 8716.500 | 9499.512 | 9879.794 | 75.293 |
| r2 / targeted | 4887.086 | 7543.169 | 7994.328 | 8892.777 | 68.655 |

targeted的等权均值在r1增加5.49%、r2减少6.26%，方向相反；两重复等权汇总为5111.892→5086.099ms，仅约−0.50%。r1的五个会话均值都更慢；r2有三个更快、两个更慢。不能因两轮混合P95都较低，就声称会话尾延迟一致改善：r1会话P95均值反而增加7.24%，r2减少13.46%。两重复混合82轮P99也从10227.712增至10446.893ms。少量重复及不同输出轨迹不支持稳定加速结论。

SLO显式保持TTFT≤2.0秒、TPOT≤0.1秒，依据已记录客户端时间；token observation为chunk，不能冒称逐token时间序列。有限观察窗口结果如下：

| 重复 / artifact | 满足SLO请求数/41 | SLO goodput req/s |
|---|---:|---:|
| r1 / v2 | 22/41 | 0.308668 |
| r1 / targeted | 19/41 | 0.249459 |
| r2 / v2 | 20/41 | 0.265629 |
| r2 / targeted | 22/41 | 0.320444 |

两重复总计42→41个SLO good请求，没有显示一致收益。以上是有限workload的吞吐/goodput，未外推稳态容量。

### 覆盖影响了决策，但没有消除其他不足

| 重复 / artifact | REBUILD / APPEND | uncalibrated fallback | cost_saving | 两候选均eligible计划/36 | eligible候选/72 |
|---|---:|---:|---:|---:|---:|
| r1 / v2 | 25 / 16 | 16 | 9 | 20/36 | 45/72 |
| r1 / targeted | 22 / 19 | 7 | 11 | 29/36 | 64/72 |
| r2 / v2 | 22 / 19 | 14 | 7 | 22/36 | 50/72 |
| r2 / targeted | 24 / 17 | 7 | 12 | 29/36 | 64/72 |

每组另有5个初始only-feasible计划，没有双候选quote，未把它们藏进36的分母。targeted两轮都减少回退并增加可比较双候选；等待候选中的 `unsupported_waiting_timing` 从18/12项降为0/0，但r2仍有2项 `unsupported_running_timing`。候选的其他时序支持/eligibility缺口仍在，不能将本轮叫“全候选已校准”。一次fallback可选中自身eligible的候选，因此fallback数不等于最终selected不eligible数。

两重复动作总量相近：v2为47 REBUILD/35 APPEND，targeted为46/36；逐任务动作并不相同。实际prefill总tokens由222,070增至227,217，输出总tokens由46,617增至46,841，不能把回退减少等同于工作量减少。单轮control均值为r1 139.926→136.120ms、r2 143.710→148.826ms，也没有稳定下降。host实际恢复轮数为r1 22→23、r2 23→21，原始H2D/D2H和occupancy统计均保留，未将这些过程量单独当作效益。

### 输入分叉与质量边界

两组对应任务的当前问题和P2固定top5正文均41/41相同，但r1完整输入IDs相同仅7/41、r2仅6/41；actual history相同也仅7/6轮。输出IDs/答案相同为2/41、1/41，动作相同26/41、33/41，visible evidence key集合相同26/41、31/41。甚至同artifact跨重复也只有7/6个完整输入相同，说明不能用同task ID替代同输入执行轨迹。

派生JSON仍提供same-input及same-input/output子集，但这些是受策略、生成和调度共同影响的事后子集，不是因果控制组。本轮确实完成了同源码、同公共配置的策略实验；并未完成固定输出/固定动作的机制重放。

截断全部保留：r1 v2/targeted分别1/2条，r2分别3/1条，共7/164。v2的截断包括 `b461…` t2、t6、t8和 `1c041…` t4；targeted包括 `b461…` t8与两次 `1c041…` t4。完整task ID在JSON中。四组都没有语义评分或质量非劣验收；代码/条件/引用、脚本问题是否与实际历史相符也未在此替代人工审查。传输/全文绑定通过不代表答案正确，截断数差异也不能代替质量判定。

当前结果支持保留定向标定作为“扩展可用决策覆盖”的工程成果，同时继续检查等待状态、输出轨迹和实际代价。它不足以将新artifact宣布为稳定性能赢家；下一步机制分析应保持固定先验与强对照，区分覆盖带来的动作变化、原生调度状态与输出长度影响，不能只继续降低门槛以增加eligible比例。
