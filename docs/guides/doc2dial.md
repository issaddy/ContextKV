# Doc2Dial / MultiDoc2Dial 数据与系统评测

同一个`adapters.doc2dial`入口读取Doc2Dial v1/v0.9和MultiDoc2Dial作者格式，导出完整原文、原始对话/grounding；MultiDoc2Dial另生成可直接送入现有P3运行器的任务。正常运行不需要训练。

## MultiDoc2Dial：跨文档多轮公开补充

[作者论文](https://aclanthology.org/2021.emnlp-main.498/)及[官方代码](https://github.com/IBM/multidoc2dial)提供DMV、SSA、VA、助学服务四域、488份文档。当前复用已有官方归档，SHA256为`f0c034c249663d7b3cb08b19cf2cc2c3d101372485be982621d4711931a1ce00`。原代码与数据各按发布源许可保留归属，第三方原始数据不打入Python发行包。

| 原分区 | 会话 | 原消息 | 系统问答轮 | 至少一次支持文档集合变化的会话 |
|---|---:|---:|---:|---:|
| validation → dev | 661 | 9195 | 4427 | 550 |
| test | 661 | 8938 | 4335 | 523 |

这里的“系统问答轮”将连续user消息合为一次请求、连续agent消息合为一次参考回答，中间用两个换行连接；原文字不改写。原消息ID、角色、顺序、DA、references和逐span原文仍在`supervision.jsonl`，任务中的`source_turn_ids`可追溯分组。两个分区分别有22/12个无作者回答的末尾user组，保留在原对话中，不虚构参考答案。这个明确的会话分组协议适配现有服务轮次，结果不直接与作者按单个agent发言计算的榜单混用。

准备官方文件后运行（validation分区将test替换为validation，输出改为dev）：

```bash
python -m adapters.doc2dial \
  --documents data/multidoc2dial/source/multidoc2dial_doc.json \
  --dialogues data/multidoc2dial/source/multidoc2dial_dial_test.json \
  --dataset-version archive-f0c034c24966 --source-format multidoc2dial \
  --source-uri https://doc2dial.github.io/multidoc2dial/file/multidoc2dial.zip \
  --split test --output data/multidoc2dial/test
python -m adapters.retrieval build \
  --corpus data/multidoc2dial/test/corpus.jsonl --index data/multidoc2dial/bm25 \
  --snapshot multidoc2dial-archive-f0c034c24966
```

全部488份文档参与检索，不按当前答案或grounding挑选语料。P3任务的`contexts`初始为空，生成时使用本方法实际历史和真实检索结果；targets、grounding与qrels只用于离线评价。文档ID采用可逆的`[domain,doc_id]`，发布版本标识语料快照，不冒充政策生效时间。

输出包括`corpus.jsonl`、`supervision.jsonl`、`tasks.jsonl`、`qrels.tsv`、`split_manifest.json`和`manifest.json`。重复运行会刷新这些派生产物；已保存运行仍以原输入摘要和源版本追溯。参考span按原字符坐标读取，错误坐标仍明确报错，避免引用错段。

统一矩阵入口：

```bash
python -m bench.b6_joint_sweep --matrix configs/benchmark_multidoc2dial.yaml \
  --stage test --job-timeout 3600 --output runs/multidoc2dial-system
```

配置使用共同BM25和相同Qwen3/资源预算比较Native与ContextKV，复用现有生成、提交、退役、成本及资源采样链。它是固定检索配方的系统执行比较；研究完整任务感知检索时，沿用现有`comparison_scope: system`和`retriever_config`接口，使用同一新语料构建BM25/BGE索引即可，不另造运行器。计划确认有必要执行时加`--execute`；准备与CPU接线检查不算回答质量实测。

保存答案的基础词项评分：

```bash
python -m bench.quality --references data/multidoc2dial/test/tasks.jsonl \
  --predictions SAVED-OFFICIAL.jsonl --output runs/multidoc2dial-overlap.json
```

复用作者采用的SQuAD式F1/EM定义，按task ID关联，报告[0,1]分数、领域/会话均值和缺失任务；缺失回答计零。本文分组后的参考回答是完整交流轮，不能冒充原始agent-turn分数。F1只补充语义正确性、条件适用性和来源支持评价，不作为唯一质量门槛。检索使用文档级qrels，系统性能/资源沿用统一指标。

## 原Doc2Dial能力

已有v1.0.1和v0.9 wOOD读取、完整语料导出及`adapters.dialogue_examples`严格前缀投影继续保留，可用于文档支撑对话研究。v1使用`references`列表，v0.9使用`reference`字典；每个原角色/原文和多标签保持原样。两版本有重叠，不混作独立样本。`respond_no_solution`或空reference只按原标注保留，不推断全语料不可答。

示例入口仍为`python -m adapters.doc2dial --source-format v1 ...`和`python -m adapters.dialogue_examples --supervision ... --output ...`。这些离线原消息投影与上述系统问答轮是不同产物。历史验证记录在`runs/doc2dial-integration/`，本次公共分区与接线检查在`runs/benchmark-suite-review/`。
