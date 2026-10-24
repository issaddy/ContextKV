# 输出长度预测：开源复用、候选条件建模与验证路线

核查日期：2026-10-24。本文核查 2024–2026 年五项直接相关方法及作者实现，提供正在推进的开发路线；下表区分可复用能力与尚需实测的适配。训练和部署状态以代码及运行产物为准，不把文献性能当作本项目结果。

**立即以已有 dev 实现并训练候选条件长度预测，用会话 OOF 和离线 rescore 选择改进方向。** 当前 [90 对诊断](../archive/stratified_decision_validation.md) 中，84/90 对的 Decode 差主要来自输出 token 数差；这给出了明确的优化目标。先并行比较低成本条件特征模型与一种冻结外部编码器/小预测头，复用 SSJF 的预测边界、LTR 的相对次序思想与 TIE 的有上限评价。尚无现成结果证明本场景有效，正是需要适配、训练和验证的部分；无需等到有额外 30 个会话才启动工程。

## 1. 需要预测的对象和可用信息

对动作 `a∈{APPEND, REBUILD}`，需要的是目标 Qwen 在完整候选输入 `x_a`、固定模板与生成设置下的服务输出长度 `N_a`，尤其是同轮差值 `ΔN=N_APPEND−N_REBUILD`。模型名称、revision、tokenizer、system prompt、temperature、stop 规则及输出上限都属于标签的条件。

一般长度预测主要区分不同请求，例如“简短事实问答”和“长代码生成”。本项目的两臂 query 完全相同，必需证据与原始 QA 也相同，但冗余正文、引用方式、消息位置和上下文距离不同。因此：

- 只输入 query 或会话公共统计的预测器，必然给两臂相同结果，不能解决当前缺口。
- 输入整个候选的外部编码器原则上能区分两臂，但可能因截断只看到相同片段，或只学到“长输入对应长输出”的弱关系。需要**同 query 内的配对验证**，不能仅看跨 query 的 R²、MAE 或排序相关性。
- 历史已完成回答可以作为特征；当前答案、当前真实长度、未来轮次、gold/reference/质量标签不能进入本轮预测。采集后的两臂长度仅用于离线训练或评估。
- 对固定上限 `M=1024`，成本标签是实际发送的 `min(N_natural,M)`。触顶的 1024 对这个服务成本目标是有效观测，但对自然终止长度只是右删失下界；不能据此宣称完整答案长 1024，也不能通过缩小 `max_new_tokens` 制造收益。

## 2. 五项方法及其适配边界

“作者开销”均为原文环境中的测量，未在本项目复现；不同计时范围不可横向拼接。

| 方法 | 原理与训练数据 | 在线成本与本项目适用条件 |
|---|---|---|
| **SSJF，2024**：[论文 §3–4](https://arxiv.org/html/2404.08509v1)，[作者仓库](https://github.com/James-QiuHaoran/LLM-serving-with-proxy-models) | 外部 BERT 加预测头，比较回归与分桶，采用 L1 回归；用目标生成的 prompt/长度数据训练，支持拼接历史后保留末尾 512 tokens。 | 不修改生成 LLM；每候选需要外部前向，可将两臂成批。作者报告预测平均约 7.6 ms，但不是本机长上下文开销。**优先复用数据组织和独立 predictor 结构**；最后 512 tokens 是否保留布局差异必须验证，上游权重不能视为已校准的 Qwen 长度模型。 |
| **Learning to Rank / LTR，NeurIPS 2024**：[论文 §4.2](https://arxiv.org/html/2408.15792v1)，[作者仓库](https://github.com/hao-ai-lab/vllm-ltr) | 小型 OPT 输出排序分数，用 ListMLE 学习输出长度次序；原文用 10K 完整生成样本，prompt 截断到 2048 tokens。目标是队列内排序，不是精确 token 数。 | 可独立部署，不改 Qwen 权重；仍需两候选编码及预测开销。**复用“关注相对次序”的训练/评价思想**，但分数不是 token 数，不能直接放进 Decode 成本公式。仅预测谁更短，也不能判断节省是否足够抵消 Prefill/恢复成本。 |
| **ELIS，2025**：[论文 §4.2、§6](https://arxiv.org/html/2505.09142v1) | 外部 BGE 与回归头；初始 prompt 及生成中的部分回答用于迭代剩余长度预测。作者采集多个目标 LLM 的输出，并报告模型匹配训练优于直接使用原聊天记录。 | 不改生成模型权重，但完整系统需要迭代执行/优先级接口；论文报告约 11.04 ms 的调度与初始预测联合开销。**仅初始 prompt 预测与目标模型重标注原则可用于当前选择**；已生成回答的信号不能提前借给未执行的候选。此次未确认作者代码入口，不声称可直接调用。 |
| **TIE，2026**：[论文 v2 §3–7](https://arxiv.org/html/2604.00499v2)，[作者仓库](https://github.com/Hyzheng-code/TIE) | 外部 DeBERTa 预测 log-t 分布参数，再按有上限期望与 CVaR 排序。原文训练使用 45K prompts×20 次生成，训练温度 0.7。 | 不改 Qwen；作者实现额外保留一张 predictor GPU，异步成批，并报告其负载下平均调度延迟 4.26 ms。**可复用截断建模和不确定性评价**，不直接移植调度器或重采样规模。本项目 greedy 重复内长度稳定，主要难点是布局条件与模型误差，不能未经检验假设每个 prompt 都服从该随机采样分布。 |
| **EGTP / PLP，ICLR 2026**：[论文 §3、Appendix E](https://arxiv.org/html/2602.11812v1)，[作者仓库](https://github.com/xiehuanyi/LP_Bench) | EGTP 对主模型隐藏状态作熵加权聚合，再预测长度分布；PLP 利用生成途中状态逐步更新。ForeLen 区分静态和动态预测任务。 | 论文约 0.65 ms 是复用已有 Prefill 状态后的增量，不能当作执行前预测两个候选的总成本。当前只读 quote 不返回这些状态；为两臂先做 Prefill 会改变成本和缓存状态。**复用静态/动态分开评价的设计，暂不接入在线选择**。 |

当前最适合工程尝试的是 SSJF 式外部预测边界，LTR 适合作为同 query 差值方向的评价参考；TIE 适合约束不确定性的处理。ELIS/PLP 的动态优势不能转借为 pre-submit 信息。即使预测模块不更新 Qwen 参数，也不代表无需更改推理服务或没有额外计算。

## 3. 固定源码核查：避免“下载即可用”的误判

| 作者仓库快照 | 可复用入口及实际限制 |
|---|---|
| SSJF `4b866866a32626677a1841a3b92875b93b1f03ab`，Apache-2.0 | [`latency_prediction.py`](https://github.com/James-QiuHaoran/LLM-serving-with-proxy-models/blob/4b866866a32626677a1841a3b92875b93b1f03ab/output-token-len-prediction/latency_prediction.py) 提供 BERT-base/tiny 分支。可作为离线训练参照；不需要连同 Triton/vLLM 调度环境迁移到本工程。 |
| LTR `13bbf6ff3dab661791d41362551b089e5f77c91c`，Apache-2.0 | [`train/`](https://github.com/hao-ai-lab/vllm-ltr/tree/13bbf6ff3dab661791d41362551b089e5f77c91c/train) 复用 allRank；[`trainer.py`](https://github.com/hao-ai-lab/vllm-ltr/blob/13bbf6ff3dab661791d41362551b089e5f77c91c/train/trainer.py) 含真实输出长度标签和 tokenizer 截断。应复用损失/评价思想或小预测器，不能安装整个旧 vLLM fork 来替换当前原生路径。 |
| TIE `ce6ddc4d7abe6a4a0821a03d463ad7af86113850`，Apache-2.0 | [`model_train.py`](https://github.com/Hyzheng-code/TIE/blob/ce6ddc4d7abe6a4a0821a03d463ad7af86113850/train/model_train.py) 默认 `MAX_LENGTH=512`，[`ua_predictor.py`](https://github.com/Hyzheng-code/TIE/blob/ce6ddc4d7abe6a4a0821a03d463ad7af86113850/vllm/v1/core/sched/ua_predictor.py) 为 2048，且有本地模型路径占位。直接接入前必须统一训练/推理输入合同；本次不修上游或启动它。 |
| LP_Bench `170a47893e1351ecc062186aafc686ab3f8be3d1` | [`feature_extractor.py`](https://github.com/xiehuanyi/LP_Bench/blob/170a47893e1351ecc062186aafc686ab3f8be3d1/EGTP/egtp/feature_extractor.py) 实际调用 backbone 前向获取 hidden states/logits；[`default.yaml`](https://github.com/xiehuanyi/LP_Bench/blob/170a47893e1351ecc062186aafc686ab3f8be3d1/EGTP/configs/default.yaml) 的 `prompt_prefix_k=4`。短共同前缀可能完全没有候选差异；不能把此默认配置等同于全候选语义预测。此快照未确认仓库许可证，不复制其源码。 |

上述是源码阅读结论，不是重新训练或复现论文结果。ELIS 只有已核查论文，不以第三方同名实现补足作者代码。较新的内部状态路线也没有解决当前“决定前不能先执行两个候选”的信息边界，因此本轮不继续扩为新内核或 speculative decoding 工程。

## 4. 现有开发数据训练、会话 OOF 与冻结测试

### 4.1 现在可做的训练探索与后续数据扩展

已经检查的 15 cases 来自原 Cloud dev 的 5 个会话，可直接用于工程训练和按会话留一 OOF。每一 fold 将一个会话的全部轮次、两臂、缓存状态和重复共同留出，其他会话用于拟合；所有已看数据上的模型设计与调参都记录为开发探索。OOF 能检查跨会话迁移和明显过拟合，但不将这批已分析数据重新命名为独立最终 test。正式 test 保持冻结。其他域的可用 dev 会话也可加入训练，并分别报告域内/跨域结果。

数据扩展与实现并行：优先补足当前弱覆盖的会话、年龄和长度区间，按真实可用的会话数安排训练/校准/评估或嵌套分组交叉验证，不设固定会话数量门槛。有关联文档族和近重复问法记录分组与跨组重叠；新增独立开发会话用于检查选定改进是否仍成立。最终模型、阈值和方法组合冻结后再运行 test；不能用同一 case 的另一缓存状态或重复伪造额外独立样本。

每个 case 冻结同一 pre-turn 历史、同一 query 和 required 集合，只执行现有两种布局。先用一个固定缓存准备协议采集长度标签；在预先选定的评估子集上补两次重复和 cold/GPU/host 检查，识别数值路径带来的输出变动。所有方法使用固定 Qwen revision、Q0 模板、greedy 设置、1024 上限和原 stop 合同。以 session/case 为采样单位，不把缓存状态重复当作训练样本量膨胀。

### 4.2 可落地模型族与逐步扩展

先保留两个简单对照：当前共享历史先验；仅从训练集学习的 APPEND/REBUILD 常数偏移。它们能检验改进是否只是“总让重建预测更短”。第一个候选条件模型可用少量固定特征的正则化回归，输入为历史已完成长度统计、候选真实 token 数、消息/QA 数、正文与引用数量、问题到相关材料的距离等；不新增生成模型，不访问当前输出。

两臂作为同一个训练 case 等权，输出为上限内的两臂长度；在 OOF 的训练 fold 内选择正则，留出会话只产生该 fold 的预测。小样本下保留每 fold 结果并报告覆盖，不虚构可靠置信区间。冻结 SSJF 式编码器特征加小预测头可以与条件特征模型并行比较；按各自预测效果、延迟和资源占用选择后续投入，而不是等简单模型失败才允许复用神经方法。在线仅加载实际参与实验的模型。

外部文本预测器必须显式定义**候选视图**：保留当前问题、已提交历史摘要统计和本候选的布局差异，而不是盲目送入 11K-token 文本后依赖默认截断。这里的视图只供预测器使用，不修改实际生成提示。先统计两臂视图是否相同、哪些字段被截断，并加入仅交换动作标签但文本不变的控制，检测 action-name 捷径。公开模型在 Llama/聊天数据上的 checkpoint 只能作为初始化或未校准对照。

### 4.3 评价顺序与准入条件

1. **预测是否解决了条件差异。** 同时报告单臂 token MAE、配对 `ΔN` MAE/符号正确率、截断概率或区间覆盖、过度低估和长尾错误；按会话、年龄、输入长度和状态分组。全局 MAE 好而配对差不变，判为没有解决核心问题。分桶或 LTR 分数只用于方向指标，不能冒充长度。
2. **误差是否足以支持决策。** 首个单因素实验固定 Prefill/TPOT artifact、`z` 和滞回参数，替换长度预测后离线重评分；随后比较长度与条件误差的联合改进。报告有效预测覆盖、回退、切换次数和同条件 regret，保留 APPEND、REBUILD、8192 阈值。若输出分布可用，在有上限长度上求 Decode 成本期望；两个各自高分位数之差不是成本差置信区间。配对误差可从训练 fold 内的分组留出/OOF 残差估计，校准不足的区域保留诊断或固定回退，并针对该区域补数据。
3. **开销是否覆盖收益。** 测两候选的完整特征提取/编码/预测 P50/P95、CPU/GPU 占用及对主服务的干扰，将实测新增控制时间计入完整成本。先做可选路径和预算扫描，再决定部署预算；一个更重但净收益更好的模型不应被未经实测的固定毫秒门槛排除。
4. **较短回答是否合格。** 预先固定质量非劣界限，用本方法自身历史、完整证据、离线 judge 与人工盲审检查。所有截断保留；短但不完整的答案不能成为准入理由。只有完成质量检查后，再运行未参与选优的完整闭环评估；单步最快臂仍只是诊断下界。

首轮保持其它成本项和回答上限不变以定位长度模块的效果；后续允许与条件误差、候选组织联合优化，并用消融区分来源。预期复用外部预测器、相对排序和有上限不确定性方法，实际贡献来自将这些能力适配到同一问题的候选差异并取得质量受检验的整体收益。探索可继续推进，默认策略和最终结论按运行证据更新。

## 5. 已运行的低成本探索版（2026-10-24）

实现位于 [`controller/output_length.py`](../../controller/output_length.py) 与 [`profiling/length_fit.py`](../../profiling/length_fit.py)。这是 SSJF“执行前外部预测器”边界的适配，直接复用 NumPy 2.3.5 的离线线性代数；没有复制 SSJF/LTR 的编码器、训练器或调度器。新增 [PARS 原论文](https://arxiv.org/html/2510.03243v1) 的核查：它采用 pairwise margin ranking，关注有意义的长度差，并非 ListMLE。本实现只借鉴其相对差值目标，没有移植其网络或损失；原文温度 0.7 条件下的差值筛选阈值没有套用到本项目 greedy 标签。

单候选模型以已有共享历史先验为基底，拟合线性或 `log1p` 长度残差。16 个固定特征来自实际候选 token 数、消息数、已提交 QA 字符数与历史 output IDs 长度、可见/必需/额外证据数、正文字符数、当前问题长度和简短词特征，以及动作与问题/历史长度的交互。没有重新分词，没有读取当轮 output、reference 或 judge。原始 QA 长度只用字符统计，未将字符冒充 token。直接差值分支输入同 case 两臂特征之差，学习 `ΔN`，以共享均值加减 `ΔN/2` 还原两臂并裁到 1024；它目前是离线两臂模型，尚未接单候选在线接口。

90 个观测对按 15 个 case 聚合，每个状态内部先平均重复，再等权平均状态；每 case、两臂等权。外层留一整个会话，共 5 折；正则强度与线性/对数目标在训练会话内部再次 LOSO，选择配对差 MAE，再以单臂 MAE 破平局。每折 P90 也只用该折训练会话内部 OOF 残差计算。输入 token 数超出训练折范围回退共享先验；直接差值模型任一臂越界时两臂共同回退。最终全 dev 拟合 artifact 与 OOF 报告分开保存。

下表为**已分析 dev 上的探索性外层 OOF**，不能据此称为独立最终测试。已有共享全局先验 433/512 对全部模型固定；本次 LOSO 隔离的是新残差拟合，不能追溯为共享先验本身的独立训练验证。

| 长度模型 | 单臂均值 MAE / tokens | 配对 ΔN MAE / tokens | ΔN 绝对误差 P90 | 非近零标签方向正确 / 14 | 域外回退臂 / 30 |
|---|---:|---:|---:|---:|---:|
| 共享历史 | 167.84 | 115.82 | 183.27 | 0（始终预测同长） | 0 |
| 仅动作偏移 | 136.66 | 125.09 | 250.40 | 8 | 3 |
| 单候选特征 | 131.06 | 119.36 | 204.80 | 8 | 3 |
| 直接配对差值 | 171.91 | 113.69 | 181.80 | 7 | 6 |

候选特征降低单臂误差约 22%，但配对差仍略差于共享先验，说明绝对长度拟合改善不能替代决策相关评价。直接差值的 ΔN MAE 只改善约 1.8%，5 个会话中 3 个改善、2 个变差；其共同均值未改，单臂误差反而增加。15 cases/5 sessions 不支持稳定优势或显著性宣称，不能把 90 次状态重复当作 90 个独立样本。18 个触顶臂均保留在服务长度目标中。最终候选 artifact 的经验残差 P90 约 331 tokens，说明不确定性仍很大；`error_tokens` 仅诊断，`p90_tokens` 用于风险/容量，不单独加入每臂时间误差后叠加到切换阈值。

两候选特征与 stdlib 点积在本机 macOS arm64 / Python 3.12.13 上测得中位 0.0162 ms、P95 0.0194 ms（15 个真实 case，各 100 次、共 1500 次 pair；另 10 次预热）。这仅是局部 CPU 预测开销，排除了构造共享先验、planner、持久化、网络和引擎，不是完整 controller P95，也没有增加 GPU 调用。首版的主要不足在预测信息和样本覆盖，局部推理成本很小；后续可直接比较能保留两臂布局差异的冻结文本编码器，而无需先改动 Qwen。

可复现命令（输出使用新路径，不覆盖旧实验）：

```bash
python -m profiling.length_fit \
  --cases runs/remote-h200-expanded/decision-dev-stratified/cases.jsonl \
  --pairs runs/remote-h200-expanded/decision-dev-stratified/pairs-lookup.jsonl \
  --baseline-cost-artifact runs/remote-h200-expanded/cost-dev-expanded-lookup.json \
  --output runs/output-length-exploration-v2/candidate.json \
  --action-output runs/output-length-exploration-v2/action-only.json \
  --delta-output runs/output-length-exploration-v2/paired-delta.json \
  --report runs/output-length-exploration-v2/validation.json
```

`validation.json` 保存每 case 的 OOF `predicted_tokens/p90_tokens/error_tokens`、每会话摘要及内外训练范围，可用于与条件 Prefill 误差组合的逐 case 重评分。`candidate.json/action-only.json` 是全 dev 拟合后的可选部署 artifact，不能用它们在同 90 pairs 上的结果冒充 OOF。`paired-delta.json` 只供离线实验；`runtime_profile.py` 与 `runtime-profile.json` 保存微测量。各 artifact 绑定 model revision、tokenizer/template fingerprint、1024 预算、特征 schema、原采集配置和输入文件哈希；这些哈希只在离线导出时计算。加载时一次检查身份与预算，在线不添加重复哈希。11 项定向测试涵盖纯 stdlib 推理、预算/身份边界、重复聚合、域外回退以及修改留出标签不改变该折均值/P90/误差估计。最终 test 未触及，质量与闭环收益尚待验证。
