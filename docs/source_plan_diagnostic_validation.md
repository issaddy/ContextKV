# 共同提示下的 D/N/P：12 题真实机制诊断

2026-10-24。**停止将当前 v2 扩到159题；不按原样进入两阶段生产集成。** 直接回答全部完成，普通notes在固定预算内全部未完成；支持计划有部分完成，但仍出现来源适用性、信息遗漏和引用错误。已有结果足以否决这个具体实现的扩大投入，不否定AFTG/ReClaim等完整方法，也不构成直接回答的普遍质量优胜或P对N的语义比较。

## 问题与公平对照

本轮只问：相同完整材料/目录/最终通用说明下，额外准备是否有可用增量；把问题要求与支持单元联系起来能否改善条件/出处保持。复用[来源方法设计](source_bound_generation_design.md)与[全局策略](research_strategy.md)，从原795次全量方案改为一次决策性小诊断。

| 臂 | 调用与预算 | 有意变化 |
|---|---|---|
| D | 单次final，cap1024 | 共同目录和说明下直接回答 |
| N | 普通notes cap384，正常结束后final cap1024 | 多一次自由文本准备 |
| P | 支持/适用性JSON计划cap384，正常结束后final cap1024 | 计划表示、grammar与必要实现指向的组合；不分别归因 |

三臂每次调用均保留同一原8B实际完整QA和当前五源。最终文本/output IDs保持原生，宿主不加引用、不用抽取结果替换；私有准备不反馈后续题。最终通用说明逐字相同，明确“如果提供了私有准备”，没有向D暗示不存在的笔记。N/P计划失败就阻止该final，保留位置，不重试、不提高cap、不换题；final length原定保留，但本批实际final均正常stop。

固定seed42仅依既有域/会话/轮数元数据抽取四域各3个不同会话，并分配早/中/晚轮，共12会话12题。未用答案、错误标签、reference/qrels选题，没有备选或扩样。12题来自已见dev，不是159轮等概率样本、闭环P2/P3或独立test。原Q0/32B只是既有能力参考，没有重跑第四臂或借旧时延计算收益。

## 实际运行与来源

- 本地/部署源码 `04b92f8c3074ef5439dda08f3e0b87b3a7b0794b`，远端343个Git文件核验；SGLang固定stock `3f36fc08331f64cd63c4fbbdf39cc402f38bb38b`。研究补丁保持不变，这次生成不经过研究Controller。
- 用户本地Qwen3-8B，实际scope身份 `local-snapshot:9e3a121da0ce4bba14a09f453123d3ca33caf62c3097ea54ee44ab29c8dbffd2`；BF16、non-thinking、temperature.7/top_p.8/top_k20/min_p0，正常EOS；单GPU/TP1，窗口与实际GPU池均32768，HiCache/SessionRadix关闭。
- 一次stock进程，c1，server seed42。按固定选择顺序，seed42初始臂排列后循环，每臂处于每个位置4次；不宣称逐请求确定性，不按臂清cache。
- 先做3次合成接口检查：正式B0 `--run-config`一轮SSE cap64；P plan→final两次nonstream cap384/1024。实际模型/采样/输入输出绑定及每调用grammar解除均通过。probe单列，不进入12题分母；其cache/grammar预热会保留。
- 29项定向CPU合同与36个真实tokenizer首输入检查通过后才实跑。没有再跑研究采样四调用、空目录、表格或第二模型探针。

原始导入：`runs/remote-h200-source-plan-diagnostic/runs/source-plan-diagnostic-v1/`。公开归档45文件（44 payload），289232 bytes，SHA256 `14d6129a1c89b39cb7e69d382debac9b015414deb2f227cceca6abb9ad9d1f04`；私有日志/环境/权重排除。选择SHA256为 `c93ddd19a9d1dfbf5ae8159f825113e7e28cb9a73865b9b3695feacf5c0ca21f`，协议SHA256为 `ed5a172ca9e858bc33a5c5e551c1d3e5f8eba4aae25a780a0476626997b74320`。导出同时保留实现helpers与源绑定，可按保存入口复算，不改旧v1。

## 覆盖、失败与成本

| 实际观测 | D | N | P |
|---|---:|---:|---:|
| 预声明回答位置 | 12 | 12 | 12 |
| 实际HTTP生成请求 | 12 | 12 | 21 |
| 正常完成的最终回答 | 12 | 0 | 9 |
| 准备length / 被阻止的final | — | 12 / 12 | 3 / 3 |
| 所有已完成stage输出tokens | 3280 | 4608 | 5332 |
| 所有尝试的输入tokens | 55321 | 54613 | 96703 |
| 已尝试arm完整墙时合计 | 18.36s | 25.43s | 29.63s |

诊断45次+接口3次=**48次实际模型调用**；60个诊断stage位置中15个final因准备失败未调用，没有删除。所有45次HTTP都有原生终态，无未知用量；请求成功不等于准备成功。driver记录`completed_with_failures`并退出1，supervisor相应标failed并完成清理/恢复，不隐藏失败。研究服务恢复PID74135，独立health200、原7B身份和空GPU/host池已确认；其健康旁证另存于公开归档之外的`restored-health.json`。

N全部恰为384个实际output IDs，finish为length，无ignore_eos/额外stop。实际输入以Qwen3非思考后缀结束；N输出未出现该模型已用的EOS，而同服务D/P正常结束实见原生stop token。文本是在持续写多级标题、问题复述及逐源概述，尾部常未完句。**这是当前notes协议在384预算内没有完成；未发现全局EOS故障，也不能称无限循环。** P的三次截断是FiQA德国/欧元区持续纾困的原因、第8轮Container Registry授权、第4轮CocoaPods。

仅对P完成的同9题，D/P完整arm均值为1516.0/2587.1ms，P final本身更短（2016 vs 2456 tokens），但含准备共4180 tokens。这个约70.7%的观测开销差只作此共享cache、单次运行下的成本描述，不作因果性能结论；不能删除P的三项失败后宣布总体效率。输入token总量也不是实际CUDA prefill工作量。nonstream没有client TTFT；extractive诊断14.33ms另列，未冒充独立模型臂。

9份有效P计划都产生完整抽取诊断：7份在1024tokens内，2份为1327/2334而超预算，均未截断。复制原文不保证回答覆盖或适用性；它只定位“原文已有，但计划/转述没有使用”的阶段。

## 从来源到计划再到回答的错误

下列为assistant逐题原文审查，包含正例与退化，不是人类盲审、五维评分或全声明错误率。全部12题记录见[Cloud](../runs/source-plan-diagnostic-review/semantic-cloud.json)、[ClapNQ](../runs/source-plan-diagnostic-review/semantic-clapnq.md)、[FiQA/Govt](../runs/source-plan-diagnostic-review/semantic-fiqa-govt.md)，共保留111处来源/答案/计划的逐字字符位置。准备截断与未生成final不补分；[独立机械审计](../runs/source-plan-diagnostic-review/h200-v1-final/README.md)逐stage实际重渲染、分词、解码及原始绑定检查通过。

| 观测 | 说明的机制问题 |
|---|---|
| Container Registry授权：D给出当前原文支持的IAM/namespace/service-ID概要；N/P草稿均截断 | 简单问题上遍历解释全部来源先消耗预算，增加调用没有形成可用回答 |
| CocoaPods：D核心“可用Carthage/source”有据，却将BMSCore材料作为Functions安装引用；P草稿虽明确Functions，仍把BMSCore备选当直接支持 | 产品/对象适用性在计划阶段就可能错，合法unit ID或最终附ID不能纠正它 |
| SELECT排障：P正确识别缺具体环境，并对watsonx.data保留if，最后请求错误/平台细节；D/P均无实际Evidence ID，Data Engine特定语法仍以一般排障建议呈现 | 局部澄清改善与引用/范围缺口同时存在；正确目录和计划映射不保证final归因 |
| HIV累计死亡：P省去D的额外无据/时效不明关系，却又将选中段内已有的2007年死亡数据误报为缺失并传入final | 局部删错与新增错误并存；整段选中不是内容覆盖，“来源没有”的自由判断可反过来抹去已有信息 |
| Cotton gin：P去掉D的一个无据设计继承关系，但仍扩大现在时地域，且把发明/使用年代错绑到另一段 | 需区分各段的不同关系与来源冲突，不能靠将所有段落放入一组解决 |
| 债券time：所选原文直接解释为报价时间，P却称未解释并在final扩写其它含义 | 信息缺口判断本身会出错，不应作为已认证缺源标签或拒答条件 |
| 海平面：计划把另一段的“近二十年翻倍”归给只给2.5→3.4的来源，final沿用 | 错引可以起源于计划；真实引用身份不保证该段支持该关系 |
| 收养与Covid：直接回答已有主要有据结论，P未显示额外调用必需；儿童活动P减少无源指南但漏用现有局部安全建议 | 需要同时看覆盖、合理不确定与新增错误，不能以更少回答或更短文本代替质量增量 |

## 研发决定

1. **当前v2到此停止扩大。** 不执行795次全量，不放宽正常结束规则补final，不增cap反复试；N无final导致P−N的语义对照无法成立，P−D也未显示足够净价值。保留具体反例与全部原件。
2. **保留成熟单次路径作为系统集成基座。** 本批D是一个可用诊断对照，也有缺引/错绑，未获正式质量资格；不据12题把其提示自动推广为全域质量赢家。继续完成同模型、共享材料的工程集成，不为保留两阶段结构延迟交付。
3. **先改支持表示和信息使用，再决定新实验。** 当前whole-passage加自由applicability说明容易变成冗长概述，并遗漏段内事实。已有AFTG等细粒度原文定位方法更贴近这些反例；先用保存的source→plan→final记录核查精确片段与邻接条件如何表示、怎样避免跨对象拼接/无依据的缺源判断。复用既有span/身份和完整上下文，不增加新的judge服务或通用图框架。
4. 新机制只有明确改变上述错误通道、相对简单直接回答有必要时才安排下一项小验证。它必须重新声明协议，不能将这批失败重标为新方法成功；独立冻结test仍未打开。
