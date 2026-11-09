# 检索与证据配置

CLI、HTTP和原生对照共用`adapters/retriever_factory.py`。`configs/retriever_dialogue.example.yaml`展示实际历史查询和稠密检索；课题主线与配对对照见[运行手册](../study/final_runbook.md)。

## 默认流程

1. BGE查询适配器读取当前原问题和已提交的实际QA，复用冻结的文档向量。
2. 按归一化向量内积召回50段，按不同正文及适用范围选至多五组。各来源保留独立引用身份，相同原文按现有机制共享。
3. 生成使用当前原文和完整实际QA。重建后的共同前缀仍由原生Radix/HiCache复用，回答、历史与引用退役一同走持久化控制器。

示例的查询侧采用CLS与FP32 L2归一化，复用原始BGE-M3文档向量；查询上限8192。`stage: hybrid`提供BM25S/RRF融合。模型与索引在启动时加载，逐轮复用；本地GPU检索等待也计入完整响应。

## 索引与运行

`data/`、`models/`和索引不随Git保存。先准备[固定提交的Cloud passage语料](https://raw.githubusercontent.com/IBM/mt-rag-benchmark/53fef6f5917c7e10c436f697e69a955dbc795d81/corpora/passage_level/cloud.jsonl.zip)，将其中的`cloud.jsonl`放到`data/mtrag-official/cloud.jsonl`；本次语料压缩包SHA-256为`a0c5427013c65556aeddfdc246dd6068f3a7fad08ce6ae7adddaa9e87aae246d`。同时准备任务文件（`python scripts/fetch_mtrag.py --dataset mtrag`），以及`models/bge-m3`中的[BGE-M3](https://huggingface.co/BAAI/bge-m3)和已评测查询适配器。启用重排时另准备`models/bge-reranker-v2-m3`中的[BGE-reranker-v2-m3](https://huggingface.co/BAAI/bge-reranker-v2-m3)。效果对照固定实际模型版本和索引。

```bash
python -m adapters.retrieval build --corpus data/mtrag-official/cloud.jsonl \
  --index data/cloud-bm25s
python -m bench.retrieval_eval build --corpus data/mtrag-official/cloud.jsonl \
  --output data/cloud-bge-m3 --encoder-path models/bge-m3 \
  --device cuda:0 --dtype bfloat16 --batch-size 16
```

索引准备完成后，按[部署](../deployment.md)启动生成服务，再运行：

```bash
python -m bench.run_sessions --config configs/qwen3_32b.yaml \
  --state-dir runs/cloud-dev run data/mtrag/RAG.jsonl --protocol P3 --domain cloud \
  --split-manifest configs/mtrag_split_manifest.json --split dev \
  --retriever-config configs/retriever_dialogue.example.yaml \
  --evidence-corpus data/mtrag-official/cloud.jsonl
```

Cloud使用72,442条原始passages；MultiDoc2Dial使用其完整4110段语料和对应索引。稀疏/稠密索引共享资料快照。配置中的路径相对工作目录，运行manifest记录实际模型、配置和索引身份。

独立检索比较使用`bench.retrieval_eval evaluate`，保存排名、耗时和Recall/nDCG。系统比较使用实际生成历史，见[数据与评价](../data_and_evaluation.md)。

## 可选配置

| 配置 | 用途 |
|---|---|
| `stage: dense / hybrid / hybrid_rerank` | 分离稠密召回、融合与重排 |
| `query_mode: current / last3 / all_user` | 当前问题或用户历史的简单检索基线 |
| `query_mode: conversational` | 可选结构化任务解释驱动检索 |
| `query_fusion: dialogue_rrf` | 原问题与派生任务共池召回、评分和融合 |
| `evidence_selection: semantic` | 一次模型调用同时解释需求和选择原文；默认ranked |
| `source_window: true` | 选材后补入同源相邻原文 |
| `query_mode: dialogue` | 当前问题加实际QA的BGE查询视图，支持dense、hybrid、hybrid_rerank及可选查询适配器 |

默认关闭原文窗口和语义选材；选择是否启用依据目标任务的回答收益和完整成本。简单BM25基线也可通过`--retriever-index`加载。

## 原文与来源范围

`--evidence-corpus`提供标题、来源、版本、章节和依赖。窗口依据同源、同版本和已声明章节补入相邻原文；材料更换或版本更新继续保留独立证据身份。

配置`source_documents`后，可从同一快照的原文恢复完整自然段及相邻段落。记录包含`document_id`、`text`、`corpus_id`、`snapshot_id`、业务`version`及来源地址；支持JSONL或单JSONL的ZIP。原文与片段通过来源身份和字面位置对齐；无法对应时使用原检索片段。

实际阅读窗口进入共同候选，原排名另存为`retrieval_contexts`。共享正文减少物理重复，各来源的适用条件与引用身份仍单独保存。表格和作者列表的结构适配见[表格关系](../research/table_relations.md)。

## 查询适配器与可选解释

`configs/retriever_dialogue.example.yaml`展示BGE查询侧LoRA，复用冻结的原文向量。`query_adapter_path`绑定输入配方、BGE基座与权重；实际实验使用配置指定的资产。

对话视图将当前原问题置前，按由近到远加入已提交QA，在8192-token预算内保留完整轮次；生成仍接收实际历史与原文。可配置`selector`维护任务状态。选择`stage: hybrid`时，稠密检索使用上述QA视图，BM25S使用当前及前两条用户问题，两路按RRF融合。

`stage: hybrid_rerank`在同一召回池上增加BGE重排，`reranker_path`可指向原版或已合并的训练检查点。重排查询优先保留当前问题，再加入完整的近期QA，以`reranker_max_length`的一半为查询预算目标；较长的当前问题单独形成查询视图，完整问题/原文对由模型的长度配置处理。默认512-token查询目标、1024-token问题/原文对，与答案支持性重排训练一致。该路径复用现有模型加载和原文选材，逐轮新增一次本地重排，不增加32B辅助调用。

启用`selector`时，任务解释随回答提交，并用于条件承接、生成和历史来源恢复；BM25/BGE查询保持实际历史视图，按正文组排序选材。解释出的子需求保留在任务状态中，该配置不把派生需求重新送入对话查询编码器。每轮一次32B解释调用，trace分别记录`resolved_query`、实际检索文本和解释状态。省略`selector`时不产生结构化回访状态；动态版本与自然回访能力应在真实会话中分别评价。
