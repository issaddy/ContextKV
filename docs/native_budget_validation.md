# Native budget：普通 EOS 观测与同快照重放

2026-10-24。首次完整普通 EOS 工作量已完成：旧 `native_workload` 的 Workload_gate0，五个 Cloud dev 会话、P2/c4、Q0、输出上限 1024，共 **41/41 成功并 COMMITTED**。产生 36 次双候选报价、72 臂预测；五个首次请求只有一个可行候选，没有报价。此次先验证可选 `native_budget` 的身份、预算与时间合同，再在同快照离线重放；没有用本轮实际输出训练或选择参数，也没有质量准入结论。

## 原始证据与运行身份

原始目录为 `runs/remote-h200-native-budget/runs/native-budget-observation-v1`，stage artifact 原样导入 `runs/remote-h200-native-budget/runs/stage-model-dev-v2/cost-stage.json`。归档 SHA256 为 `fdbc05c241bbfc42598629cd86d6de9164bb5802c08e76d197b2b633cc505303`。运行使用部署源码 `3b852aa`、固定 SGLang `3f36fc08331f64cd63c4fbbdf39cc402f38bb38b`、研究补丁 `e004ca96d62a5786f303c465da140513c7213f13ff023407991ec162ef82d92c`；job manifest 实际控制器源码 SHA256 为 `45ac1b23b5e2eea0cc69241ac2d3c2100ca5e648610b004c57bfdef30577ea39`。Commit 是部署记录，实际文件 hash 是另外的观测；cold hook 不自行证明部署 commit。

实际 GPU/host 池为 16384/279040 tokens，page64、FULL Python tree、write_through、cache 模式、layer_first/kernel、TMA disabled、sample_pool enabled，与冻结 stage artifact 的 fingerprint/cache_config_hash 匹配。这个匹配只证明配置范围，不证明新等待模型准确。启动后独立 cold hook 返回 0，GPU/host unique tokens 均为 0；保留完整 before/reset/after 原始状态。

可复现入口与派生物：

```bash
.venv/bin/python runs/native-budget-review/replay.py \
  --input-dir runs/remote-h200-native-budget/runs/native-budget-observation-v1 \
  --artifact runs/remote-h200-native-budget/runs/stage-model-dev-v2/cost-stage.json \
  --execute --output runs/native-budget-review/summary.json
.venv/bin/python runs/native-budget-review/inspect_contract.py
```

这些 ignored 脚本只读本地已保存文件，不调用引擎或 tokenizer。`summary.json` 保留完整 turn 状态、41 个 SQLite stage、所有 quote/候选、无 quote 的初始请求和 unsupported；`contract-check.json` 另从原始事件/receipt复算身份、公式和阶段绑定，并定位失败回调。源文件 hash 在分析入口记录一次，未增加在线 hash。

## 普通 EOS 与时间合同

36 批的实际 `new_token_ratio` 范围为 0.098–0.7。逐批按原生预算成员重算 `min(max_new_tokens−generated, clip) × ratio`（ignore_eos 时乘 1），与 `native_running_budget_tokens` 的差值全部为 0。

Running/service、running/budget 分别有 62 次成员出现，waiting 有 26 次；**这是重复观测次数，不是独立请求数**。所有成员 `ignore_eos=false`，62 个 budget 出现均实际使用 ratio 缩放。每个摘要都有原生 request_id，均通过 `(engine_restart_id, request_id)` 精确关联已保存 receipt.rid；不再按输入长度/时间猜请求身份。本轮 budget/service 的 ID 集合在每批都相同，且成员 native_finished 均为 false，所以没有真实覆盖未清理 finished、retracted 或 overlap 集合差异。实际 clip=4096，声明输出上限 1024，没有触发 clip 截断；这些边界仍只有 CPU 源码合同测试。

Waiting 的最近匹配有 19 次 known、7 次 unknown：GPU-only 2、host-only 11、mixed 5、已观测零前缀 1。Known match 的同 scheduler monotonic 年龄为 57.21–191.60 ms，全部不晚于 quote。Running 的 62 次均有记录，年龄 60.82–4683.79 ms。它们只说明最近一次原生匹配的形状与时间，**不证明当前仍驻留**，unknown 也没有改写成 cold 确证。

41 个 SQLite stage 与 benchmark 的 receipt、输入长度、实际输出数、server TTFT 全部一致。36 个被报价后实际执行的候选，其 input_hash 与终态 receipt 完全相同，因而可关联服务端标签。本文的测量量为

`actual_pre_forward_ms = server_ttft_ms − forward_to_first_token_ms`。

两项来自同一服务端 monotonic 时钟。该量是首个 forward 之前的**前置联合墙时**，包含可能的准备/恢复/排队，不能称纯排队或 CUDA 时间。另一个候选没有实测标签，不能由该对照推断动作的因果收益。

## 同快照重放：尚未改善共同支持样本的误差

使用原 quote 的输入长度/缓存形状、生成前保存的 mean/P90 prior、相同 stage artifact 与配置，旧模型 72 臂的等待 mean 和 upper 全部精确重现，差为 0。无需重建 token 内容；这两个等待函数本身只读取标量特征。实际输出数只用于事后残余诊断，没有进入任何预测回调。

旧模型等待回调支持 72/72 臂；新模式支持 56/72，剩余 16 臂是八个报价的两个候选，均为 `unsupported_waiting_timing`。56 个共同支持臂中，9 个等待均值发生变化。下表只比较存在真实标签的已执行臂；“原 eligible”是实际旧规划器的选择资格，不能与新 support 混为一谈。

| 标签人群 | 旧/新可比样本 | 旧 MAE ms | 新 MAE ms | 旧/新 actual−mean ms |
|---|---:|---:|---:|---:|
| 全部已执行且有 quote | 旧 36 / 新 28 | 739.86 | 600.68 | +631.42 / +434.14 |
| 原 eligible 已执行臂 | 旧 34 / 新 26 | 721.99 | 566.60 | +607.17 / +387.25 |
| **两模式均支持的已执行臂** | **28 / 28** | **586.40** | **600.68** | **+446.97 / +434.14** |

前两行人群不等，不能用它们宣称误差改善。共同支持的 28 臂中，新模式的 MAE 略高；实测超过较大风险情景的个数从 9 增为 12。风险情景不是概率上界，这个计数也不是置信区间覆盖率。结果说明原生预留语义和身份合同已经可用，但**并未证明等待精度改善**。

## 新 unsupported 的具体来源

16 臂全部来自均值情景中四种历史 host/mixed match 的 queued forward 定价，每种在两个报价的双臂中重复出现。它们的 c1 准备项全部 supported；失败的是 c=2 forward 在冻结采样几何中的支持范围。不是 match 年龄超限或形状不符，也不是把缺少 ID 当作未知。

| `(P,U,L,c)` | 最近采样点 | 归一距离 |
|---|---|---:|
| (6144,131,6144,2) | (8064,128,8064,1) | 0.64937 |
| (5184,1129,5184,2) | (4032,64,4032,1) | 0.60993 |
| (1216,4204,1216,2) | (0,4096,0,1) | 0.54217 |
| (2176,3017,960,2) | (2048,2048,0,1) | 0.50208 |

这些点的最近距离均超过原阈值 0.5，本轮保持 unsupported，没有为使新方法可选而放宽阈值。下一轮可以单列“native budget + cold service”与“native budget + 历史 match 服务情景”，判断预算分离和服务情景各自的作用；也可用新的独立 c2 host 采样补足真实形状，但须新 artifact 与独立验证，不能把已有点伪装成已覆盖。

同源码、同配置的新模式完整 41 轮工程闭环现已完成，见下一节。即使闭环耗时变化，P2 历史、动作、输出长度和排队状态都可能传播，仍需报告实际输入/输出匹配与失败，不能把一次有序开发运行当作质量等价或吞吐因果结论。

## 新模式完整闭环：有平均时延变化，尚无因果胜出

新运行原始目录为 `runs/remote-h200-native-budget/runs/native-budget-closed-loop-v1`，21 文件归档 SHA256 为 `53db5d4c379905b2db91682a17570db6c10414ab2ce3388e1eff2dd8bb632182`。同一部署源码与 e004 补丁、模型/指纹、cost artifact、cache_config、数据/协议/负载均相同，完整 config 差异只有 `state_dir` 和 `admission_estimator`。两次均为各自新进程后的 KV cold reset；这是旧后新的一次有序开发对照，没有随机化顺序或重复实验。

离线 `runs/native-budget-review/compare_closed_loop.py` 复用已有报告读取与分布函数，结果在 `closed-loop-comparison.json`。82 个 audit 的 input IDs hash 与 receipt 一致，输入数和输出 IDs 全部匹配；每轮实际历史恰为该臂此前完整用户问题与真实生成答案，未删 QA。两臂均 41/41 成功，0 unfinished，分别 1 个输出截断，冷置后两级占用均为 0。

Pipeline 完成时间统一为 `max(t_response_done,t_commit) − benchmark t_arrive`。

| 实测项 | 旧 native_workload | 新 native_budget |
|---|---:|---:|
| 五会话等权平均 Pipeline ms | 4940.50 | 4393.91 |
| 全 41 轮平均 / P95 Pipeline ms | 4900.19 / 7584.49 | 4353.50 / 7943.37 |
| Control 平均 / P95 ms | 140.94 / 236.39 | 140.91 / 256.44 |
| SLO 好请求数 / 41 | 20 | 24 |
| 观测窗口 SLO goodput req/s | 0.27739 | 0.38789 |
| REBUILD / APPEND | 22 / 19 | 24 / 17 |
| cost_saving / fallback / hysteresis | 10 / 7 / 19 | 10 / 12 / 14 |
| 候选 selection_eligible / 72 | 63 | 53 |
| 实际输入 / 输出 tokens | 231858 / 22719 | 220156 / 21508 |
| 实际 Prefill / host loaded tokens | 106994 / 81792 | 107580 / 62528 |
| Host 恢复轮数 | 22 | 20 |
| 完成 D2H / H2D bytes | 7369392128 / 4690280448 | 7332691968 / 3585605632 |

SLO 使用既有 TTFT≤2 s、TPOT≤0.1 s，并基于客户端 chunk 观察；观测窗口分别为 72.10/61.87 s，不能外推稳态服务容量。两臂所有 72 个候选的 `calibrated=false`，解析选择资格没有被误写成经验校准。新方法 fallback 增加，正与历史 host 服务情景支持域缺口相关；本次并没有消除该缺口。

跨臂相同 task/query 为 41/41，但实际 input IDs 与历史仅 11/41 相同，output IDs/答案仅 6/41，动作与可见证据 key 集合各 35/41。相同 input 子集平均 Pipeline 为 3143.72→2758.27 ms；相同 input+output 的六轮为 3096.10→3142.28 ms。子集是事后形成且很小，仍受并发邻居/排队不同影响，不能作为随机对照。总体均值下降伴随输入和输出数量下降、动作/历史传播及更高 P95，故不能直接归因为预算分离的净效果，更不能据此声明答案质量相同。

新运行有独立的 36 quotes/72 arms。按实际保存的 estimator=`native_budget` 重放，等待均值逐项精确重现；JSON 的 `old_saved_mean_difference_ms` 此时表示“旧反事实估计减新记录”，不应被误读为复现失败。新等待回调支持 60/72，12 个 unsupported 仍为 waiting timing，已执行臂可用 30 个。在这些**新运行的相同 30 个支持样本**上，旧等待公式 MAE 为 437.22 ms，新为 438.79 ms；actual 超出较大风险情景的个数为 11/18。这个样本与前节旧闭环的 28 臂不同，不可跨行拼接为独立泛化结果。源码、合同与完整闭环已验证，等待精度改善仍未证实。

## 定向标定入口与合同（CPU与目标GPU采样已完成）

后续四组96目标请求已全部完成并满足exact合同，合并拟合与冻结quote反事实结果见[定向补采验证](native_budget_targeted_validation.md)。以下保留入口与采样合同；覆盖扩展不代表等待误差或闭环收益已成立。

保持同 16K GPU、279040 host、page64、layout/IO/source/model，先补 c2 的缺失形状，不放宽 0.5 支持距离。`profiling.sglang_probe` 已允许任意正整数 prompt 长度；只有可复用 prefix 向下页对齐。因此 suffix 131、1129、4204 本身可以精确保留。可用页内分数 `(P+0.5)/n` 避免浮点正好落在页边界，CPU `calibration_grid` 已核得到目标 P。

| n / 目标 prefix / host load | CLI prefix fraction | 当前 CLI 能否原样实现 |
|---|---|---|
| 6275 / 6144 / 6144 | 0.97920318725099598 | host_only，能 |
| 6313 / 5184 / 5184 | 0.82124188183114211 | host_only，能 |
| 5420 / 1216 / 1216 | 0.22444649446494466 | host_only，能 |
| 5193 / 2176 / 960（GPU1216） | 总 prefix 可为 0.4191218948584633 | 新增 mixed + device-prefix-tokens=1216；须真实 quote 验收 |

前三种各单独调用现有 CLI，防止长度与分数笛卡尔积引入其它无关形状。最低沿用每 case 暖 1、正式 3、并发 2、output128：三种共 24 个目标请求（18 正式、6 暖机），另有每次 fixture 的 24 个一-token 前缀准备请求。可再独立加入 output512 作为预算稳健性检查，目标请求数加倍；不是为挑更好结果而选择预算。三个 case 的两路预热前缀及 `2×(n+128)` 均小于16K，仍以实际 prepared quote/receipt 为准。

```bash
python -m profiling.sglang_probe --config <同16K配置> --base-url <专用服务> \
  --model-path <固定模型> --runtime-manifest <同机runtime.json> \
  --lengths 6275 --prefix-fractions 0.97920318725099598 \
  --outputs 128 --concurrency 2 --cache-states host_only --warmup 1 --repetitions 3 \
  --output <新目录>/host6275-c2.jsonl
```

Mixed case 现已实现小型事前 fixture：先将完整 2176 前缀备份并以已有 host_only API 驱逐到 host，再用目标前 1216 token 加一个不同 sentinel 做一-token 生成并 settle，使目标报价严格验证 GPU1216、host-only960、recoverable2176 且无 pending transfer。最终每个 receipt 必须仍为 D1216/L960、actual_prefill3017；不符即失败，保存原始 result 与 after metrics/DMA，不回填或改标签。没有新增引擎 API 或修改在线 native admission。调用 `--cache-states mixed --device-prefix-tokens 1216`；总 prefix 仍由前述分数确定。

`PrefillCase` 只追加可选 `device_prefix_tokens`，默认记录仍保持原五字段；旧 CLI 默认三态未改变。`stage_fit` 的 mixed 特征 L 由事前 `P-D` 生成，再核对实际 receipt。**不能把全 host2176 的测量改标签成 mixed960**。旧三态继续使用 prepared 特征合同，实际 tier 漂移仍单独审计：重新加载旧 552 条正式记录全部通过，36 条漂移仍原样保留（concurrent-small 6、concurrent-long-device 30），未从未来实际恢复改写 L。每个 group 的 prepared/exact 合同、正式请求数与漂移数写入新拟合 provenance；新 mixed 始终 exact，无全局放宽或自动删旧样本。本次三个纯 host 目标的精确执行由采样后的独立审核另核，不冒称所有旧样本都具有精确实际 L。

ignored入口 `runs/run-native-budget-targeted-profile.py` 已完成真实采样：四形状、c1/c2、output128/512，各暖1/正式3，共72正式+24暖机目标请求，另有120个一-token前缀准备请求；它独占启动/清理16K服务并恢复保存的研究服务。新96目标均complete且prepared/receipt形状exact，独立audit为issues=[]。27项初始合同及后续57项组合测试验证工程路径，实际GPU结果另以原始telemetry/receipt核对；不能把CPU double当硬件证据。原始目录和逐形状结果见[定向补采验证](native_budget_targeted_validation.md)。

合并复用现有 `profiling.stage_fit`：保留原四组完整 Measurements、compact stage 与各原 manifest，再追加新组，给每组传 `--profile-manifest GROUP=PATH`；所有 run_id 必须完整对应、共同 fingerprint/cache_config_hash 一致。原始 telemetry 逐请求压缩为现有 `group/case/index/warmup/timing/receipt` compact，不按时延排序，暖机不入训练。用冻结旧 stage artifact 作为 `--base-artifact` 可保留已有 decode、长度先验及其来源，输出新的 stage artifact 与 whole-cell OOF validation，不覆盖 v2。定向形状来自开发运行，补采仍是开发定向校准；须另用新采样/闭环检验误差，不能把同一测量标签同时当特征或独立验证。

合并首次fit在冻结先验来源绑定处失败，`809a3c1`修复继承路径后保留原decode/经验长度与来源，成功生成624正式请求、46 forward cells的新artifact。两组冻结quotes的native-budget等待覆盖补齐到72/72，共同支持MAE只小幅变化，且一个未执行APPEND的idle setup仍不支持；新同源164轮闭环已完成：回退30→14，但等会话均值两次变化方向相反，平均仅约−0.50%，SLO通过42→41，未证明稳定加速。完整失败日志、artifact SHA、人群分母和解释边界见[补采与反事实验证](native_budget_targeted_validation.md)。
