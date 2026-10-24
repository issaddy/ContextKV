# Qwen3 候选成本的三处定向诊断

> 归档记录：保留当时的设计、结果与复用依据；其中“下一步”和执行计划不代表当前任务。当前工作仅以[实现清单](../implementation_plan.md)为准。

2026-10-24，源码`e85663e`。基于已有159轮Qwen3轨迹的[137个一步候选](../research/qwen3_candidate_opportunity.md)，预先按结构选取三个代表位置，再执行一次受控A/R比较。实际6次正式生成、6次单token历史物化，共12次调用；全部正常完成，没有重试、追加预算或扩大矩阵。服务已停止，GPU0和端口已释放。

最直接的结论是：**在这六个实际回答中，decode占服务端完整时间的95.49%–99.16%。仅改善prefill不足以代表完整回答的收益；输出长度与回答内容必须共同分析。** 没有拟合新成本表，没有改变生产的always_rebuild默认，也未授予任何策略质量资格。

## 选择依据与实际协议

只使用预轮次的`R输入长度、A额外上下文、page64新增token节省、attention基A/R`四维。按有无required交集及attention基方向分为81/23/33个位置，分别选择IQR缩放L1距离的medoid；平局按task ID。当前输出、耗时、语义评分及历史输出长度均不参与选择。选样扰动/反序合同和独立重算通过。medoid描述本层中心形状，不是随机抽样，不能估计总体平均收益。

| 层与原task | 原位置数 | A / R输入tokens | A / R GPU命中 | A / R实际prefill |
|---|---:|---:|---:|---:|
| 无交集，`b4614…`第4轮 | 81 | 7946 / 5277 | 3968 / 1280 | 3978 / 3997 |
| 有交集、A attention基较低，`1c041…`第3轮 | 23 | 4294 / 3855 | 3520 / 704 | 774 / 3151 |
| 有交集、A attention基较高，`b2805…`第5轮 | 33 | 6076 / 3997 | 3776 / 1088 | 2300 / 2909 |

冻结直接复用已发生的R输入、历史实际输出和原材料，只在三个当前位置重建候选；所有原R字段及已保存A/R结构exact。没有伪造SQLite/native账本。第一例包含历史中实际截断的1024-token答案，完整保留该记录，不称历史答案均自然完成。

复用固定SGLang、Qwen3-8B本地scope、非思考、`.7/.8/20/0`、cap1024、正常EOS，GPU16384、实际host54272、page64、FA3/FCFS/chunk8192、write_through、host admission off。每臂先重置cache，再以相同前序`input+actual output+sentinel`物化，随后同批报价两个候选。两臂准备形状一致，实际receipt均等于各自报价，6次正式调用均为stop、host loaded均0。write_through仍可产生写回，零恢复不等于无传输或纯CUDA执行。

`seed=42`只决定臂顺序，不是每请求共同采样随机流。每处只有一次A/R，没有共同长度控制、稳态重复或独立质量测试。缓存重置不等于GPU/kernel编译状态重置，也没有自动加warmup网格。

## 原生测量与解释

| 层 | A / R首token ms | A / R输出tokens | A / R decode ms | A / R完整服务 ms |
|---|---:|---:|---:|---:|
| 无交集 | 152.98 / 101.88 | 878 / 499 | 4737.80 / 2652.47 | 4890.78 / 2754.35 |
| 有交集、A attention较低 | 26.91 / 78.55 | 604 / 618 | 3174.19 / 3241.15 | 3201.11 / 3319.70 |
| 有交集、A attention较高 | 61.82 / 70.46 | 245 / 359 | 1307.77 / 1894.42 | 1369.59 / 1964.88 |

无交集例只减少19个prefill token，却多保留2669个上下文tokens；实际首token增加51.09ms，完整时间增加2136.42ms，其中decode增加2085.33ms。不能把全部差异归因于prefill或上下文长度，因为A还多生成379tokens，回答内容也不同。

有交集且A attention较低的例子节省2377个prefill token，首token减少51.64ms；完整时间只减少118.60ms（该对观测的3.57%），A还少输出14tokens。其约5.25–5.26ms的观测平均TPOT意味着数十毫秒prefill优势很容易被十余个输出token的变化影响；这仅解释量级，不是可部署的输出长度阈值。

第三例的attention基A/R为1.53，实际A首token仍低8.64ms：说明单个attention几何项不能给总prefill成本排序，suffix数量、其他计算与运行开销也有影响。完整时间减少595.29ms，其中decode差586.65ms，同时A少输出114tokens；不把它宣传为缓存带来的30.3%加速。

`server_ttft_ms`与`forward_to_first_token_ms`都是原生wall阶段；本次两者相差约0.94–1.46ms，不将差值命名为纯排队。nonstream客户端没有首token观测。完整服务时间不含检索、Controller候选构建、prepare/retire或业务提交，因此也不是HTTP端到端延迟。

## 与实际回答一起判断

[六答案的来源审查](../../runs/qwen3-b-targeted-review/semantic.json)读取了各自完整输入与历史，保留逐字范围；属于assistant_audit，没有新judge调用或人工金标。三处均未建立A专属旧正文带来的净语义收益，但不能据此认定旧正文无用：第一处保留了直接相关的WebSocket参数说明，第三处保留了棉花替代来源。required交集为空也不等于语义无关。

- **customization ID：** 两臂都解释了标识符，却没有直接消解历史代码中的“here”，也都没有按要求引用Evidence ID。更长的A将customization ID误用于`customer_id`删除接口，并把原文的`language_customization_id`参数写成`customization_id`；这些具体错误可由已给材料核对，不能算作长回答的覆盖收益。
- **如何改进query：** 最初用户问题涉及SELECT，后续模型历史已扩展到多个产品。两臂偏向Discovery且存在产品范围混用；R额外保留条件式SQL分支，A对反馈训练作了更强的无据解释。用户目标尚有歧义，不能预设更短或更专注的一臂质量相同。
- **英国与南方棉花：** 两臂的主要肯定回答有据。R保留了1860年、英国市场最高80%的直接支持，但增加了无据动机推测；A更短，却将特定时间的部分替代扩大为“不完全可替代”。同一关键正文两臂都可见，差别不只是是否取回来源。

这些是已观察错误和覆盖差别，不证明APPEND保留旧材料导致错误，也不支持通过压短答案取得质量相等的加速。下一机制必须同时改善历史指代、产品/参数归属与有据回答覆盖；仅调整生成长度先验或引用格式无法解决这些问题。

## 本轮纠偏与下一投入

采集前发现并修复了实际fixture验收缺口：原collector允许“GPU-ready报价、执行却转host”的合成反例通过。现在cold/GPU-ready逐臂比较实际device/host/prefill，保存原结果后将漂移列失败；host-only仍允许原生准入选择重算。固定上游源码核对了首chunk计数及`n-1/page`边界，72项针对性测试通过。[原生审查](../../runs/qwen3-b-native-audit/README.md)区分CPU反例与这次实际硬件结果。

这次没有成本artifact，六个prediction均未校准，执行选择标记为always_rebuild。保存的regret只描述固定选择在本次两份输出下的耗时，不能称作B模型验证。数据已经足以否定“按新增token或attention单项就选动作”和“TTFT改善即整体改善”的简单假设；不足以决定总体策略或拟合成本系数。下一投入应优先解决来源适用性与回答内容差异，并在确有前缀复用价值、输出成本可解释时继续必要的B标定。没有理由自动扩零交集组、host/并发容量或全形状网格。

原件为`runs/remote-h200-qwen3-b-targeted/runs/qwen3-b-targeted-v1/`，19文件公共归档109264 bytes，SHA256 `d34e9a589e8256240d070b665a547f094b207b0d205d32e3c41603fcf6380782`，18个payload逐项核验通过。逐次calls保留全部12个提交和返回；native cold reset会更换restart并清空事件环，最终metrics只覆盖最后一臂的物化和正式调用，不能宣称保存了12个scheduler terminal事件。所有输入、报价、receipt和实际输出另有逐臂原件。

[独立原生审计](../../runs/qwen3-b-targeted-native-review/README.md)有285项核对、0问题，两个输入/receipt篡改反例均检出；代码不复用collector来验证自身。末次retire后GPU/host仍各6592tokens，随后owned进程被终止、GPU为空；returncode为−9，不称为优雅退出或retire物理清零。wheel/sdist与`e85663e`逐字一致；已移除两份完成打包后的生成源码目录，保留Git、发行包、构建日志及研究原件。

[选样、冻结、机制与答案审查快照](../../runs/qwen3-b-targeted-review-v1.tar.xz)共27文件、122856 bytes，SHA256 `167f16af34a535dbced74c820356530e25db4e497568cc88807a8e174dabddb9`；26个payload核验通过。它与上述真机原件分开归档，不包含私有日志、环境变量或模型权重。
