# 数据与评价

使用 Qwen3-32B 非思考模式，固定组间生成模型、采样、资料范围与资源配置。生成、任务解释和离线评分分别记录调用与成本。完整运行资产保存在 `runs/` 和 `data/`，轻量结果和成本摘要可随代码交付。

## 公开任务

| 数据集 | 用途与已接入范围 |
|---|---|
| [MT-RAG](https://github.com/IBM/mt-rag-benchmark) | 多资料多轮问答；固定110会话842任务，P3已准备Cloud的205任务，dev/test为41/164轮 |
| [MultiDoc2Dial](https://github.com/IBM/multidoc2dial) | 公共服务条件、跨文档切换；488篇原文、4110段，validation/test各661会话，合计8762轮 |
| [MTRAG-UN](https://github.com/IBM/mt-rag-benchmark/tree/main/mtragun-human) | 歧义、不可答与部分可答；507个选中任务，采用完整作者历史的P1 |
| [Loong](https://github.com/MozerWang/Loong) | 长资料中的定位、比较与推理；第一长度组323题，按32B实际分词统计预算覆盖 |
| [VersionQA](https://github.com/danielhuwiler/versionrag) | 100个单轮版本问题和34份公开文档；评估版本内容、版本目录和变更，与多轮版本演化负载互补 |

MT-RAG与MTRAG-UN使用上游`53fef6f5917c7e10c436f697e69a955dbc795d81`。接入步骤见[MultiDoc2Dial](guides/doc2dial.md)与[Loong](guides/loong.md)。并发、停顿回访和来源版本更新通过明确的负载条件补充，分别统计。

VersionQA使用上游`2a2cbe8f285557f99dc7a79d6df7134e1e3eccff`，用本地32B分词器按512 token、50 token重叠切分原文；PDF的跨度指提取后的文本。每次准备生成语料与任务文件的哈希，由本次运行绑定。以下命令需要评价环境安装`pypdf`，也可用`--pdf-text`指定已记录源文件SHA-256及提取器版本的文本；生成任务和参考答案分文件保存：

```bash
python -m bench.versionqa --source /path/to/versionrag/data/raw \
  --questions /path/to/versionrag/data/test/evaluation_set.csv \
  --tokenizer /path/to/qwen3-32b --output /path/to/new/versionqa
```

## 比较协议

| 协议 | 历史 | 当前材料 |
|---|---|---|
| P1 | 完整作者历史 | 官方给定来源 |
| P2 | 本方法实际生成并提交的问答 | 官方预检索来源 |
| P3 | 本方法实际生成并提交的问答 | 本方法实际检索 |

P2/P3从首轮连续运行。参考答案、qrels和支持标注用于离线评价。系统比较允许各方法组织不同证据；机制消融固定其余算法配置并保留各自实际历史，固定输入/执行轨迹用于局部成本归因。开发集用于选择，测试集用于独立验证。

评价同时查看任务与对象、条件和来源适用范围、必要回答点及原文支持；自动评分配合关键样本来源复核。性能报告完整响应均值/P95、会话累计等待及非截断达标吞吐。资源报告GPU/Host唯一KV积分，并计入解释、检索等辅助计算。失败、截断和未完成轮次随结果保留。

## 当前评价入口

默认主比较为并发 1／4 下的 `RAG_native`、`context_cache_demand`、`LlamaIndex_native`，采用完整 350 题、77 会话和各自真实 P3 历史。普通 RAG 和 LlamaIndex 共用发布版检索栈，所有改写、阅读和 refine 调用进入实际成本。

按需消融分别解释缓存编译、需求保留、动作收益决策和跨轮事实复用，方法对见[任务书](study/task_spec.md#六实验与验收)及 `configs/study_matrix.yaml` 的 `optional_ablations`。历史固定/收益 B × 编译四格保留其原有结果与归因范围。当前默认批次不要求重跑历史四格或遍历 TTL、辅助模型及表示组合。

`bench.study_scoring` 调用官方评分，`bench.reassess_saved` 对已存答案离线重评；
`bench.study_metrics.declared_queue_summary` 使用全部声明题目作为分母，失败、未启动、
截断及未评分条目保留，所有已付计算成本计入账本。

`profiling.study_cost_ledger` 汇总训练、在线执行、辅助模型及离线评分的实际调用和物理积分。
GPU/Host 物理驻留与唯一 KV 页分别报告，共享页只计一次，空闲驻留仍计入总成本。
实际结果由每次运行的 manifest、原始回答、评分与采样追溯；执行命令见[运行手册](study/final_runbook.md)。
