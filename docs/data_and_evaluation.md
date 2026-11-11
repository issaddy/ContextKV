# 数据与评价

固定 Qwen3-32B 权重、解码设置和资源配额。公开题目保留作者的输入、参考答案和任务身份；检索、分段、上下文压缩与缓存组织属于被比较系统的实现。各数据集单独报告原指标、完整响应 P95、全部在线成本和 GPU/Host 占用。

## 当前配置与用途

| 配置 | 数据与历史 | 用途 |
|---|---|---|
| `configs/study_matrix.yaml` | MultiDoc2Dial、MT-RAG Cloud、VersionQA、Loong；作者历史 P1 | 公开任务质量与端到端效率 |
| `configs/study_system_matrix.yaml` | 原 MultiDoc2Dial、MT-RAG 问题；各方法实际回答历史 P3 | 多轮状态、事实复用和缓存协同的系统重放 |
| `configs/study_mechanisms.yaml` | `constructed_version`、`constructed_long`，独立构造 | 版本切换、共享提取与确定性统计的机制诊断 |
| `configs/study_mtrag_un.yaml` | MTRAG-UN 选中任务、完整作者历史和给定材料 | 按需检查信息不足、部分可答与非独立问题 |

四类配置使用同一执行器、账本和报告器。默认执行公开测试分区；`--split dev` 只选择已有开发输入，`--datasets` 选择本次需要的数据集。原数据没有开发分区时，不用构造题替代其开发分区。任务数与会话数从选中数据实际计算。

系统重放保持原问题及参考答案，只替换历史中的回答来源。它是公开数据上的项目实验，结果单列；问题不会随系统回答重新生成。构造任务继续使用已有独立资产，不混入公开分数。

## 公开任务与评分

| 数据集 | 当前接入范围 | 评分 |
|---|---|---|
| [MultiDoc2Dial](https://github.com/IBM/multidoc2dial) | 原 validation/test；开发选择固定会话子集，测试使用原任务 | SQuAD 式 token F1 与 EM，存储范围 0–1 |
| [MT-RAG](https://github.com/IBM/mt-rag-benchmark) | Cloud 原问题，按会话划分项目 dev/test | 作者 MTRAGEval，保留 RL_F、RB_llm、RB_alg 及官方聚合 |
| [VersionQA](https://github.com/danielhuwiler/versionrag) | 原 100 个版本问题与公开资料，单轮 | 作者二值正确性 rubric，`versionqa_accuracy` |
| [Loong](https://github.com/MozerWang/Loong) | Set1 完整资料系统适配；保留原题、原答案，修正 Link 字段 | 作者 0–100 分 rubric；另列 Perfect Rate、领域和难度 |
| [MTRAG-UN](https://github.com/IBM/mt-rag-benchmark/tree/main/mtragun-human) | 当前发布版选中问题及完整作者历史 | 原 MTRAGEval；独立参考历史诊断 |

MultiDoc2Dial 任务选择遵循[作者加载器](https://huggingface.co/datasets/IBM/multidoc2dial/blob/main/multidoc2dial.py)：当前 user 的下一条消息须为 agent，且该回复不是 `respond_no_solution`。输入保留此前全部原消息，目标仅为紧接的原 agent 回复；不合并连续同角色文本。现有导出任务需要从原始对话重新生成，原语料与检索索引可复用。步骤见 [MultiDoc2Dial 接入](guides/doc2dial.md)。

MT-RAG 与 MTRAG-UN 固定上游 `53fef6f5917c7e10c436f697e69a955dbc795d81`。Cloud 是已接入的领域子集，从发布任务中按本次清单选题，保留会话顺序和领域标识。报告其原始 MTRAGEval 指标。

VersionQA 固定上游 `2a2cbe8f285557f99dc7a79d6df7134e1e3eccff`。使用[作者判分规则](https://github.com/danielhuwiler/versionrag/blob/2a2cbe8f285557f99dc7a79d6df7134e1e3eccff/src/evaluation/evaluation_llm.py)，记录实际 judge 模型；更换 judge 后属于同 rubric 的本地评分。资料分段是检索实现，原问题、参考答案与类别保持不变：

```bash
python3 -m bench.versionqa --source /path/to/versionrag/data/raw   --questions /path/to/versionrag/data/test/evaluation_set.csv   --tokenizer /path/to/qwen3-32b --output data/evaluation/versionqa
```

`bench.study prepare` 只准备其公开资料与索引，不联动辅助训练或构造任务生成。构造资料由已有 `training.reading_data` 单独准备。构造版本任务沿用项目条件/版本评分；构造长文任务使用 `constructed_numeric_exact_v1`，最终数量和总额同时匹配才得分。它们不使用 VersionQA 或 Loong 名称。

Loong 的资料准备见 [Loong 接入](guides/loong.md)。当前系统完整保留资料、固定发布顺序，并按论文定义修正 Link the Links 的字段判断；这些输入适配沿用已有 manifest 明示，不作为作者运行脚本的原样复现。评分按已保存 task ID 续跑，复用已付 judge 结果；记录实际 judge，与原作者模型不同的运行按本地评分解释。2WikiMultiHopQA 保留独立准备和评价入口，当前任务不要求追加该实验。

## 执行与报告

主比较为 `RAG_native`、`context_cache_demand`、`LlamaIndex_native`，默认并发 1/4。缓存编译、需求保留、动作收益和事实复用各保留一个必要对照，按当前机制问题选择；不展开数据集 × 历史协议 × 旧四格 × TTL 的笛卡尔积。

每个 job 对应一个数据集、一个方法和一个并发。默认沿会话前后依赖推进，思考间隔为零；已有真实固定到达轨迹时用 `--arrival-plan` 复用。调度方式及端到端计时口径保存在运行清单，不为凑出缓存重访添加延时。

`bench.study_scoring` 按数据集调用对应评分。原指标与完整题目分母、完成率一起报告；评分缺失不冒充零分，确认没有交付的任务在完整队列指标中计零。不同数据集不合成加权 Q，构造任务也不参与公开成绩。

最终指标由 `bench.capacity_workflow` 汇总，`profiling.final_study_index` 生成对比表。GPU/Host 采样只在内存中积分，长期保存积分结果。评分结束后清理答案、原文副本、judge 明细和状态库。训练采集只保存形成训练输入与标签所需的数据。执行命令见[运行手册](study/final_runbook.md)。
