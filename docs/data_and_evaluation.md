# 统一 benchmark、指标与已有结果

主线使用公开任务、原始资料与共同评测口径。自构问题只用于复现缺陷，不承担整体效果结论；开发诊断、公开任务对照和独立冻结测试分别标明。

## 公开 benchmark 的选择

| Benchmark | 与课题的关系 | 当前用法 |
|---|---|---|
| [MT-RAG](https://github.com/IBM/mt-rag-benchmark) | 110个人工多轮会话、842个任务，覆盖Wikipedia、金融、政府与技术资料 | 统一主benchmark；固定原问题、来源、参考答案和会话顺序，运行P1/P2/P3 |
| [MTRAG-UN](https://github.com/IBM/mt-rag-benchmark/tree/main/mtragun-human) | 非独立问句、歧义、不可答/部分可答和回答澄清 | 官方挑战补充，使用选定轮次的P1历史，不伪造前序任务或改成新的完整会话 |
| [MultiDoc2Dial](https://github.com/IBM/multidoc2dial) | 跨文档、跨主题的目标导向对话，含文档结构 | 后续扩展候选；现有`adapters.doc2dial`处理的是Doc2Dial，不能称已经复现MultiDoc2Dial |
| [LongMemEval](https://github.com/xiaowu0162/LongMemEval) | 长期会话记忆、时间推理与知识更新 | 供扩展记忆任务参考；不替代当前外部证据多轮QA主评测 |

固定上游版本为`53fef6f5917c7e10c436f697e69a955dbc795d81`。MT-RAG的`RAG.jsonl`是110会话/842任务、每题五段的特定数据设置，不是系统的通用证据数上限。现有会话划分为22 dev/88 test，见`configs/mtrag_split_manifest.json`；已有使用记录保留，不把反复接触过的任务重新称为未见测试。

MTRAG-UN论文描述666个任务/六域，**当前公开reference文件实际为507个任务/四域**：ClapNQ142、Cloud131、FiQA77、Govt157。每个会话只发布一个选中任务，历史最多到第11轮；证据0–38段，其中130个任务没有给定证据。Banking/Telco尚未在该固定公开文件中发布。不裁成五段、不填充空材料，也不按任务答案选子集。该文件已完成507条读取、调度、完整导出和离线标签关联检查，尚未在当前版本生成507条答案。

## 协议与比较对象

| 协议 | 模型看到的历史 | 当前材料 | 回答的问题 |
|---|---|---|---|
| P1 | 数据集给定历史 | 官方给定来源，数量随任务 | 标准给定历史/材料下的质量；支持MTRAG-UN选中轮次 |
| P2 | 本方法此前实际生成并提交的QA | MT-RAG原始预检索来源 | 隔离上下文执行和缓存协同；不是完整在线检索 |
| P3 | 本方法此前实际生成并提交的QA | 本方法真实检索 | 完整A/B/C效果，计入解析、检索、选择、生成和提交 |

P2/P3沿用公开用户问题，后续问题并非交互用户模拟器。标准P1结果与采用自生成历史的P2/P3结果分别汇总。任务标签、参考答案和qrels仅离线评分使用。

矩阵的`comparison_scope: system`允许方法在`method.retriever_config`声明不同检索/意图机制，评价A/B/C整体。默认`execution`使用同一检索配方隔离B/C；若检索依赖各自实际历史，仍须核对保存的所选证据，不能仅凭配置相同就称材料相同。两类比较共用基座模型、资料范围、问题、采样、输出要求和资源口径。

预先固定、没有候选选择的配置可直接运行声明的test分区，不要求先消耗GPU做形式上的dev调参。真正调参或从开发结果选配置时，继续绑定相应冻结选择。开发流程不以所有评分工具、所有消融或所有指标完备为前置。

## 可复现入口

```bash
python scripts/fetch_mtrag.py
python scripts/fetch_mtrag.py --dataset mtrag-un
```

下载器校验一次固定文件摘要，再生成统计；取消对同一文件额外硬编码任务数/重叠数作为准入门槛。已有原始文件可加`--offline`。官方标注和所有原文保持不变。

`configs/benchmark_mtrag.yaml`是当前可执行的完整系统对照：Native B1使用已有last3混合检索，ContextKV使用conversational机制，共享同一Cloud语料和生成模型，固定两种方法，无参数搜索。当前已安装P3索引只覆盖Cloud：对应205个官方任务，既有test分区实际164轮/方法。其他领域需准备对应索引后才能计入P3全域结果；P1/P2已经能够读取全部四域。

按[部署说明](deployment.md)准备固定引擎、索引、模型与32B解释服务后，先查看计划：

```bash
python -m bench.b6_joint_sweep --matrix configs/benchmark_mtrag.yaml \
  --stage test --output runs/mtrag-system-eval
```

确认本次研究问题需要生成对照后，在同一命令加`--execute`。不默认乘上并发、种子、重复或策略大矩阵；系统承载研究另按服务目标声明少量有意义负载点。

MTRAG-UN可通过已有服务读取官方选定轮次：

```bash
python -m controller.cli --config configs/qwen3_8b_cost.yaml \
  --state-dir runs/mtrag-un-eval benchmark --input data/mtrag-un/reference.jsonl \
  --domain all --protocol P1 --split-manifest data/mtrag-un/split_manifest.json --split test
```

所有给定历史和原文计入实际token预算，溢出显式记录，不改题或裁证据。官方竞赛format checker的十段上限与公开reference文件中部分任务不一致；提交校验和无损结果导出分开，不能为通过checker损坏测试数据。

## 质量、性能与资源指标

- **质量主指标**：复用官方参考答案、来源支持和可答性/歧义处理评价，按官方任务类别及领域报告。官方脚本提供`RL_F`、参考答案相关评分及IDK/underspecified调整项；名称和算法随固定脚本记录。必要时对关键任务作来源复核，不以引用格式、改写句式或模型输出长度替代回答有效性。
- **系统主指标**：完整响应均值/P95、会话累计等待；在预先声明服务期限后报告非截断达标吞吐。当前HTTP完整返回，内部引擎TTFT仅作诊断。
- **资源主指标**：相同资源下的服务能力，以及GPU/Host唯一KV槽位积分/完成轮次。解析/检索设备、模型驻留和实际调用成本计入系统；输入tokens、命中率和追加比例用于解释结果。
- **可靠性**：失败、截断、未完成轮次、提交/恢复一致性单列。CAS、版本身份与原文边界支撑正常运行，不将无关格式或缺少某个可选指标变成研发阻断条件。

官方离线关联入口会保留当前方法的真实输入和答案，再关联`targets`、`answerability`、问题类型与多轮标签。作者时间戳不进入模型输入，也不被误判为历史发生变化：

```bash
python -m bench.official prepare --input SAVED-PREDICTIONS.jsonl \
  --references data/mtrag-un/reference.jsonl --output runs/mtrag-un-labelled.jsonl
python -m bench.report SAVED-RUN.json --output runs/summary.json
```

后续官方评分复用`bench.official`和固定上游脚本。评分模型/版本与实际历史协议一起记录；同族模型评分可用于诊断，关键非劣/优胜结论再做相应复核与会话级统计。缺少评分保留为未评估，不触发无目的的补实验。

## 已有对照结果

以下是保存的版本结果，不代表最新全部机制已经重新测量。

| 负载、版本与协议 | Native B1 | Simple | ContextKV | 可支持的结论 |
|---|---:|---:|---:|---|
| 六会话40轮/方法，`8aeeb5b`，P2，完整响应会话等权均值 | 1.187s | 1.265s | 1.258s | 三者40/40完成、无截断；该开发点ContextKV没有快于Native |
| 两会话15轮/方法，`1733f12`，P3，同检索结果，完整响应均值 | 2.671s | 2.565s | 2.160s | 此负载观察到19.1%的Native响应均值下降，实际历史/输出长度差异也参与总效果 |
| 同一P3负载，完整响应P95 | 5.226s | 4.437s | 3.928s | 改善该负载的尾延迟，不推断最大容量 |
| 同一P3负载，预定5s内完成吞吐 | 0.503 | 0.541 | 0.579 req/s | 每方法14/15轮达标；这是开发诊断期限 |

P3中GPU唯一KV积分/完成轮为Simple21712.97、ContextKV21635.63槽·秒；Host分别48178.87、49202.63。Native没有同口径物理积分。资源并非各项同时下降，不能用逻辑token补造缺失值。两控制器outbox清空、会话关闭。

`84031e5`随后在同一15轮任务接入条件选材：保留Cloudant无需预定义schema的正确解释，排除另一产品建表材料；完整响应均值2.860s、P95 4.834s。该质量增益有额外成本，API细节仍有错误。此前120条答案的同族32B评分暴露过完整性与忠实性混淆，原分数保留为诊断，不据小数差宣布质量排名。

## 当前机制效果与重复问题

`c0e28f3`至`146d1e0`完成首轮子需求状态、联合检索、相邻原文竞争及简洁上下文。四轮CLI版本/平台/来源切换是**自构开发诊断**，不是新的benchmark：状态属性均保留，完成2次追加、2次重建，但出现猜测包名和私有/公共流程混用。进一步来源选择补入Linux包及旧版本模板；不可靠的行号/用途标签已删除。

两条固定历史检查中，Linux包地址及解压后运行安装脚本恢复正确，仍混入.NET建议；公共来源切换仍误用私有地址。同历史、同材料单独测量元数据精简，输入1771→1541、4702→4121 tokens，分别减少13.0%和12.4%。这证明输入精简，不自动等于物理资源或整体速度提升。

重复错误的分析分为：任务解释是否正确 → 当前证据是否足够且适用 → 模型是否正确使用证据。相邻片段可能缺少所属章节条件，排名覆盖只衡量相关性；完整旧答案也可能强化先前错误。不得继续将它们统一归因于某个阈值，或只凭可引用ID判定事实支持。下一步机制优先恢复原文范围与当前条件的联系；在公开任务上检查是否改善同类问题。

## 原始记录与当前审查

- 已有P2对照、模型/采样/引擎记录：`runs/mainline-delivery/delivery-evidence/delivery-run/`。
- 已有P3对照及逐任务原文：`runs/mainline-delivery/closure/`；条件选材：`runs/mainline-delivery/condition-selection/`。
- 正文共享与多轮状态：`runs/quality-first/exact-body/`、`task-operations/`；最新机制与诊断：`runs/quality-first/requirements/`。
- 本轮公开数据、依赖审查、矩阵计划和CPU接入检查：`runs/benchmark-review/`。本轮未启动GPU生成或评分。
- 逐轮开发过程从本页合并移出，原始数据不删除，详细历史可从Git版本`905652d`追溯。

当前交付包含可运行工程、已测得的条件性服务收益和明确的语义瓶颈。统一公开benchmark用于判断下一阶段整体质量与成本，不为预设优势筛题，也不因少数开发案例抹去已有收益。
