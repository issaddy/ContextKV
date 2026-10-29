# 检索与证据配置

CLI、HTTP和原生对照共用`adapters/retriever_factory.py`，默认配置为`configs/retriever_conversational.example.yaml`。每轮完成任务解释、检索、重排和选材，再交给共同的APPEND/REBUILD装配。

## 默认流程

1. Qwen3-32B读取当前问题、实际问答和已提交任务状态，更新任务、对象、条件及子需求。
2. BM25S与BGE-M3各取50段，以RRF融合；BGE-reranker-v2-m3对候选评分。
3. 按不同正文及适用范围选择至多五组材料。复合任务保留主任务首组，再增加子需求覆盖；同组来源保留各自引用身份。
4. 当前解释、原始问题、实际历史和原文一起进入生成。解释随成功回答提交。

BGE-M3使用CLS向量与L2归一化，文档索引为float32内积检索；重排使用原始logits。默认encoder上限8192，reranker上限1024。模型与索引在启动时加载，逐轮复用。解释HTTP调用可并发，进程内encoder/reranker通过一把锁使用GPU；等待与检索均计入完整响应。

## 索引与运行

```bash
python -m bench.retrieval_eval build   --corpus data/mtrag-official/cloud.jsonl --output data/cloud-bge-m3   --encoder-path models/bge-m3 --device cuda:0 --dtype bfloat16 --batch-size 16
python -m bench.run_sessions --config configs/qwen3_32b_cost.yaml   --state-dir runs/cloud-dev run data/mtrag/RAG.jsonl --protocol P3 --domain cloud   --split-manifest configs/mtrag_split_manifest.json --split dev   --retriever-config configs/retriever_conversational.example.yaml   --evidence-corpus data/mtrag-official/cloud.jsonl
```

先按[部署](../deployment.md)启动生成和解释服务。Cloud使用72,442条原始passages；MultiDoc2Dial使用其完整4110段语料和对应索引。稀疏/稠密索引共享资料快照。配置中的路径相对工作目录，运行manifest记录实际模型、配置和索引身份。

独立检索比较使用`bench.retrieval_eval evaluate`，保存排名、耗时和Recall/nDCG。系统比较使用实际生成历史，见[数据与评价](../data_and_evaluation.md)。

## 可选配置

| 配置 | 用途 |
|---|---|
| `stage: dense / hybrid / hybrid_rerank` | 分离稠密召回、融合与重排 |
| `query_mode: current / last3 / all_user` | 当前问题或用户历史的简单检索基线 |
| `query_mode: conversational` | 当前默认任务状态解释 |
| `query_fusion: dialogue_rrf` | 原问题与派生任务共池召回、评分和融合 |
| `evidence_selection: semantic` | 一次模型调用同时解释需求和选择原文；默认ranked |
| `source_window: true` | 选材后补入同源相邻原文 |
| `query_mode: dialogue` | 当前问题加实际QA的BGE查询视图，配合dense与可选查询适配器 |

默认关闭原文窗口和语义选材；选择是否启用依据目标任务的回答收益和完整成本。简单BM25基线也可通过`--retriever-index`加载。

## 原文与来源范围

`--evidence-corpus`提供标题、来源、版本、章节和依赖。窗口依据同源、同版本和已声明章节补入相邻原文；材料更换或版本更新继续保留独立证据身份。

配置`source_documents`后，可从同一快照的原文恢复完整自然段及相邻段落。记录包含`document_id`、`text`、`corpus_id`、`snapshot_id`、业务`version`及来源地址；支持JSONL或单JSONL的ZIP。原文与片段通过来源身份和字面位置对齐；无法对应时使用原检索片段。

实际阅读窗口进入共同候选，原排名另存为`retrieval_contexts`。共享正文减少物理重复，各来源的适用条件与引用身份仍单独保存。表格和作者列表的结构适配见[表格关系](../research/table_relations.md)。

## 可选查询适配器

`configs/retriever_dialogue.example.yaml`采用BGE查询侧LoRA，复用原文向量。`query_adapter_path`绑定输入配方、BGE基座与权重；依赖通过`uv sync --extra tokenizer --extra retrieval --extra query-adapter --locked`安装。

对话视图将当前原问题置前，按由近到远加入已提交QA，在8192-token预算内保留完整轮次；生成仍接收实际历史与原文。该模式不调用任务解释器。适配器不是默认配置，仍需在32B环境验证条件承接和完整响应。
