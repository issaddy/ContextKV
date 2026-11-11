# Doc2Dial / MultiDoc2Dial 数据与系统评测

`adapters.doc2dial`读取Doc2Dial v1/v0.9和MultiDoc2Dial作者格式，导出完整原文、原始对话及grounding；MultiDoc2Dial 另生成保持作者任务边界的公开评价输入。

## MultiDoc2Dial：跨文档多轮公开补充

[作者论文](https://aclanthology.org/2021.emnlp-main.498/)及[官方代码](https://github.com/IBM/multidoc2dial)提供DMV、SSA、VA、助学服务四域、488份文档。当前复用已有官方归档，SHA256为`f0c034c249663d7b3cb08b19cf2cc2c3d101372485be982621d4711931a1ce00`。原代码与数据各按发布源许可保留归属，第三方原始数据不打入Python发行包。

任务选择遵循[作者加载器](https://huggingface.co/datasets/IBM/multidoc2dial/blob/main/multidoc2dial.py)：选择后面紧接有效 agent 回复的 user 消息，排除作者定义的 `respond_no_solution` 回复。`task_id` 为作者 `dial_id_user_turn_id`，`turn` 保留该原 user 消息编号，输入包含此前完整原角色/原文，目标仅为紧接的单条 agent 回复。连续同角色消息不合并。任务数由导出 manifest 计算。

公开配置使用作者历史 P1；`study_system_matrix.yaml` 使用相同原题和目标，按各方法自身回答 P3 连续重放。任务边界已更新，须从原始对话重新导出 tasks/qrels/manifest；原语料和检索索引可复用。

准备官方文件后运行（validation分区将test替换为validation，输出改为dev）：

```bash
python3 -m adapters.doc2dial \
  --documents data/multidoc2dial/source/multidoc2dial_doc.json \
  --dialogues data/multidoc2dial/source/multidoc2dial_dial_test.json \
  --dataset-version archive-f0c034c24966 --source-format multidoc2dial \
  --source-uri https://doc2dial.github.io/multidoc2dial/file/multidoc2dial.zip \
  --split test --output data/multidoc2dial/test --sections
python3 -m adapters.retrieval build \
  --corpus data/multidoc2dial/test/passages.jsonl --index data/multidoc2dial/bm25-sections \
  --snapshot multidoc2dial-archive-f0c034c24966-structure-v1
python3 -m bench.retrieval_eval build \
  --corpus data/multidoc2dial/test/passages.jsonl --output data/multidoc2dial/bge-m3-sections \
  --snapshot multidoc2dial-archive-f0c034c24966-structure-v1 \
  --encoder-path models/bge-m3 --device cuda:1 --dtype bfloat16 --encoder-max-length 8192
```

全部488份文档参与检索。任务的 `contexts` 初始为空，生成时使用所选协议的历史和真实检索结果；targets、grounding 与 qrels 用于离线评价。文档ID采用可逆的`[domain,doc_id]`，发布版本标识语料快照。

输出包括`corpus.jsonl`、`supervision.jsonl`、`tasks.jsonl`、`qrels.tsv`、`split_manifest.json`和`manifest.json`。重复运行会刷新这些派生产物；已保存运行仍以原输入摘要和源版本追溯。参考span按原字符坐标读取，错误坐标会报错。

`--sections`复用[作者结构化分段原则](https://github.com/IBM/multidoc2dial/blob/2456a5609b447e1a12b716f4d658728f346f3041/scripts/data_preprocessor.py)，按连续相同section ID或标题组织段落。输出`passages.jsonl`的4110段是原文连续切片；拼接后仍完整还原488篇文档。标题路径只辅助检索，不改写正文，也不读取grounding挑选段落。`passage_id`定位段落，`document_id`保留父文档；文档级qrels评价先按父ID折叠排名。

使用`bench.retrieval_eval evaluate`评价这些章节排名时，传入`--qrels-parent-corpus data/multidoc2dial/test/passages.jsonl`。工具在离线读取标注后按原顺序折叠父文档，再计算文档级Recall/nDCG；未知片段ID会报错。原始章节排名保留在运行记录中，文档命中不代表已找回该文档内的支持跨度。

作者提供`doc_html_ts`时，同次导入将列表嵌套和`text_id`对应的原文坐标写入`metadata.source_lists`。JSONL的`text`仍是原始切片，已有BM25/BGE索引可直接复用。默认直接使用原文；显式设置`metadata.source_list_reading: true`时，`CorpusIndex`加载阶段插入作者的`ul/ol/li`结构，联合选材与生成使用同一阅读正文。列表中的“或”、前提和例外原话均保留，不自动推断逻辑标签。跨章节列表只保留本段完整的嵌套列表，不合并整篇文档或丢弃原话。

这一处理采用[Docling结构化序列化](https://docling-project.github.io/docling/concepts/chunking/)的原则，直接利用现成HTML及坐标，不引入OCR、额外模型或转换依赖。格式化正文参与既有证据身份和实际token计量。质量导出以当前可引用的实际阅读材料为准，另存原始检索结果。公开输入配对诊断未显示HTML标记的净收益，因此结构数据的保留与是否注入模型分开处理，结果见[统一评价](../data_and_evaluation.md)。

为这一语料生成推荐的实际历史检索配置，以及既有比较配置：

```python
from pathlib import Path
import yaml
for name in ("dialogue", "hybrid", "conversational"):
    cfg = yaml.safe_load(Path(f"configs/retriever_{name}.example.yaml").read_text())
    cfg.update(sparse_index="data/multidoc2dial/bm25-sections",
               dense_index="data/multidoc2dial/bge-m3-sections")
    if name != "dialogue":
        cfg["reranker_max_length"] = 8192
    Path(f"data/multidoc2dial/retriever_{name}.yaml").write_text(yaml.safe_dump(cfg))
```

两个固定BGE检查点均支持8192；评分预算设为8192可覆盖更长章节。编码器建索引只做一次，完整正文继续进入生成。全文BM25索引仍可作为简单对照使用。

系统评测统一使用 `bench.capacity_workflow`，按数据集选择检索配置和方法，见[运行手册](../study/final_runbook.md)。

保存答案的基础词项评分：

```bash
python3 -m bench.official overlap --references data/multidoc2dial/test/tasks.jsonl \
  --input SAVED-OFFICIAL.jsonl --output runs/multidoc2dial-overlap.json
```

复用作者采用的SQuAD式F1/EM定义，按task ID关联，报告[0,1]分数、领域/会话均值和缺失任务；缺失回答计零。评分单元对应原任务，公开主指标为 F1，EM 同时保留。检索使用文档级qrels，系统性能/资源沿用统一指标。需要参考答案与来源支持评分时，复用`bench.official generation-http`并加`--no-answerability-adjustment`，运行原始RB_llm/RL_F；该数据缺少MT-RAG式可答性标签，因此关闭IDK调整。针对开发决策可用`--metrics RB_llm`或`--metrics RL_F`计算所需项；补充另一项时复用已完成阶段和回答。

## 原Doc2Dial能力

已有v1.0.1和v0.9 wOOD读取、完整语料导出及`adapters.dialogue_examples`严格前缀投影继续保留，可用于文档支撑对话研究。v1使用`references`列表，v0.9使用`reference`字典；每个原角色/原文和多标签保持原样。两版本有重叠，不混作独立样本。`respond_no_solution`或空reference只按原标注保留，不推断全语料不可答。

示例入口仍为`python3 -m adapters.doc2dial --source-format v1 ...`和`python3 -m adapters.dialogue_examples --supervision ... --output ...`。这些离线原消息投影与 MultiDoc2Dial 公开任务分别使用对应作者格式。
