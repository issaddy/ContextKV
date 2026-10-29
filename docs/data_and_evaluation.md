# 数据、评价与当前结果

默认使用Qwen3-32B非思考模式，生成、任务解释和离线评分分别记录调用与成本。机制比较固定生成模型、采样、资料范围与资源配置。运行记录保存在`runs/`，历史实验通过Git及压缩原始记录追溯。

## 公开任务

| 数据集 | 用途与已接入范围 |
|---|---|
| [MT-RAG](https://github.com/IBM/mt-rag-benchmark) | 多资料多轮问答；固定110会话842任务，P3已准备Cloud的205任务，dev/test为41/164轮 |
| [MultiDoc2Dial](https://github.com/IBM/multidoc2dial) | 公共服务条件、跨文档切换；488篇原文、4110段，validation/test各661会话，合计8762轮 |
| [MTRAG-UN](https://github.com/IBM/mt-rag-benchmark/tree/main/mtragun-human) | 歧义、不可答与部分可答；507个选中任务，采用完整作者历史的P1 |
| [Loong](https://github.com/MozerWang/Loong) | 长资料中的定位、比较与推理；第一长度组323题，按32B实际分词统计预算覆盖 |

MT-RAG与MTRAG-UN使用上游`53fef6f5917c7e10c436f697e69a955dbc795d81`。接入步骤见[MultiDoc2Dial](guides/doc2dial.md)与[Loong](guides/loong.md)。并发、停顿回访和来源版本更新通过明确的负载条件补充，分别统计。

## 比较协议

| 协议 | 历史 | 当前材料 |
|---|---|---|
| P1 | 完整作者历史 | 官方给定来源 |
| P2 | 本方法实际生成并提交的问答 | 官方预检索来源 |
| P3 | 本方法实际生成并提交的问答 | 本方法实际检索 |

P2/P3从首轮连续运行。参考答案、qrels和支持标注用于离线评价。系统比较允许各方法组织不同证据；隔离B/C时共享材料与输入。开发集用于选择，测试集用于独立验证。

评价同时查看任务与对象、条件和来源适用范围、必要回答点及原文支持；自动评分配合关键样本来源复核。性能报告完整响应均值/P95、会话累计等待及非截断达标吞吐。资源报告GPU/Host唯一KV积分，并计入解释、检索等辅助计算。失败、截断和未完成轮次随结果保留。

## 当前Qwen3-32B验证

原始记录位于GPU容器`/home/relay/songjiayun/ContextKV/runs/qwen32-default/`，本地同名目录保存配置、摘要和清理记录。

| 项目 | 结果 |
|---|---|
| 原生生成与生命周期协议 | 32B冒烟通过；GPU池32768槽位，16GB Host实际61056槽位，page size 64 |
| 成本采集与拟合 | 146组GPU/Host/mixed测点，覆盖512–30720输入token与1/2/4/8/12并发；1314个正式请求、4252条阶段测量 |
| 阶段数据 | 140个forward单元、26个decode区间；运行文件保留45个单请求准备单元。整单元留出forward MAE为98.5ms；异质队列等待精度仍需负载验证 |
| 自然输出先验 | Cloud 41轮＋MultiDoc 55轮，96/96成功、无截断；均值259、P90 543 token，来自32B实际回答 |
| 清理后默认链路 | 一个完整Cloud开发会话8/8成功、无截断；7轮使用成本选择，0轮回退。平均完整响应10.79秒；这是运行验证，尚无同条件质量/性能对照 |
| 正式效果与场景覆盖 | 32B质量比较、长上下文、来源版本演化、并发和停顿回访尚待完成 |

自然输出采集与独立GPU标定曾同时运行，96条回答用于长度先验。后续8轮验证使用清理后的代码、32B生成与32B任务解释，模型、缓存配置和采样匹配；成本数据精简前后预测输入完全一致。

## 运行入口

先按[部署](deployment.md)准备生成和解释服务，再运行矩阵：

```bash
python scripts/fetch_mtrag.py
python -m bench.b6_joint_sweep --matrix configs/benchmark_mtrag.yaml   --stage test --execute --job-timeout 3600 --output runs/mtrag-system-eval
```

其他任务使用`configs/benchmark_multidoc2dial.yaml`、`benchmark_mtrag_un.yaml`及`benchmark_loong.yaml`。省略`--execute`可查看计划。矩阵按配置启动、记录并关闭生成服务；解释服务由外部启动。P1按完整资料预算执行，超预算题目单列。

回答固定后使用`bench.official`关联标签并调用固定上游评分脚本；`bench.quality_acceptance`比较已有结果，`bench.report`汇总完整响应和资源。工具参数见各模块`--help`，成本采集见[标定指南](guides/stage_cost_model.md)。

后续工作集中在当前需求准确进入检索、条件与来源适用范围、证据充分性及回答落实，并通过上述公开任务与真实负载评价辅助调用、完整响应和资源收益。
