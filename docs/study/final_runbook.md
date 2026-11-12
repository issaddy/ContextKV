# ContextKV 运行手册

当前版本使用共享查询/证据编码、结构化文档计算、多步价值停止和 SGLang InfoFlow 执行。共享索引已建立，SGLang 精确生成、独立块构建、GPU 热复用及 Host 备份已实际跑通；正在完成辅助训练，逐 benchmark 端到端结果尚未生成。

## 路径与环境

- 项目：`/home/relay/songjiayun/ContextKV`
- 固定生成模型：`/data/models/qwen3-32b`
- 查询骨干：`models/qwen3-embedding-0.6b`
- 证据骨干：`models/jina-reranker-v3.5`
- Python：容器默认 `python3`；GPU 使用 4–7。
- 配置：`configs/study.yaml`；引擎启动见[部署](../deployment.md)。

代码和必要轻量结果可同步；模型文件、向量数组和大日志留在实验容器。

## 数据与索引

各 benchmark 分开准备和选题，使用原始问题、参考答案及对应公开评分：

| Benchmark | 任务与资料 | 评分 |
|---|---|---|
| MultiDoc2Dial | `data/multidoc2dial/{dev,test}/tasks.jsonl`、`passages.jsonl` | F1、EM |
| MT-RAG Cloud | `data/mtrag/RAG.jsonl`、`data/mtrag-official/cloud.jsonl` | 作者 MTRAGEval |
| VersionQA | `data/evaluation/versionqa/references.jsonl`、`corpus.jsonl` | 作者二值 rubric |
| Loong Set1 | `data/loong/set1-v2/tasks.jsonl`，每题附带文档 | 作者百分制 rubric、Perfect Rate |

导入入口为 `adapters.doc2dial`、`adapters.prepare_data`、`bench.versionqa`、`adapters.loong`。已有原始资料与标准题目直接复用。稠密索引使用冻结 Qwen Embedding，需要与本版骨干重新建立；Loong 的随题资料在注册后建立来源索引。

## 辅助训练

固定 Qwen3-32B 作为标签和实际答案生成器。查询训练使用会话对齐与任务关系；证据训练先完成字段/来源头，再加入实际答案集合效用；最后采集多步停止轨迹。

```bash
export CUDA_VISIBLE_DEVICES=4,5,6,7

python3 -m training.prepare_shared_labels \
  --source data/multidoc2dial/source \
  --query-model models/qwen3-embedding-0.6b \
  --endpoint http://127.0.0.1:30000 \
  --output data/training/shared/labels.jsonl \
  --train-states 24 --dev-states 8 --candidates 6 --concurrency 4

python3 -m training.shared_training query \
  --model models/qwen3-embedding-0.6b \
  --data data/training/shared/labels.jsonl \
  --output models/shared-query --device cuda:2

python3 -m training.shared_index \
  --corpus data/multidoc2dial/test/passages.jsonl \
  --snapshot multidoc2dial-test \
  --model models/qwen3-embedding-0.6b \
  --output data/indexes/multidoc2dial --device cuda:1 --dtype bfloat16

python3 -m training.shared_training evidence \
  --model models/jina-reranker-v3.5 \
  --query-model models/qwen3-embedding-0.6b --query-asset models/shared-query \
  --data data/training/shared/labels.jsonl \
  --output models/shared-evidence --device cuda:3 --field-only

python3 -m training.collect_shared_values sets \
  --config configs/study.yaml --data data/training/shared/labels.jsonl \
  --output data/training/shared/sets.jsonl --concurrency 4

python3 -m training.shared_training evidence \
  --model models/jina-reranker-v3.5 \
  --query-model models/qwen3-embedding-0.6b --query-asset models/shared-query \
  --init-asset models/shared-evidence --data data/training/shared/sets.jsonl \
  --output models/shared-evidence --device cuda:3

python3 -m training.collect_shared_values stop \
  --config configs/study.yaml --data data/training/shared/labels.jsonl \
  --output data/training/shared/stop.jsonl --concurrency 4

python3 -m training.stop_value data/training/shared/stop.jsonl \
  --output models/shared-stop.pt --device cuda:2 \
  --seconds-price 0.001 --tail-price 0.001
```

MT-RAG、VersionQA 各用自己的 corpus 路径运行 `shared_index`，输出与 `study_matrix.yaml` 中的索引目录对应。独立 corpus 的索引构建可以并行。

标签集按会话分离 train/dev，包含公开训练来源和新构造的关系/版本训练资料。集合样本保存候选集合、实际质量和完整成本；停止样本保存紧凑表示及多步回报需要的费用。必须完成的数值总体由覆盖规则执行，不学习提前停止。

## 少量端到端评测

默认每个 benchmark、每种方法 10 题；可用 `--limit-turns 50` 改为各自 50 题。三个方法使用相同题目与生成条件，每个 benchmark 单独评分和报告。

```bash
python3 -m bench.study evaluate \
  --config configs/study.yaml --matrix configs/study_matrix.yaml \
  --split test --limit-turns 10 --concurrency 4 \
  --output runs/current/public

python3 -m bench.study evaluate \
  --config configs/study.yaml --matrix configs/study_system_matrix.yaml \
  --split test --limit-turns 10 --concurrency 4 \
  --output runs/current/actual-history
```

`--methods ordinary_rag contextkv llamaindex` 和 `--datasets multidoc2dial mtrag versionqa loong` 可指定本次范围。公开评测使用参考历史；实际历史评测在会话内顺序执行，失败轮次结束后，后续轮次使用此前已提交历史。

## 保留结果

- `results.jsonl`：逐题答案、来源地址、质量、完整响应、Qwen 输入/输出 token 与缓存计数。
- `batch_cost.json`：整批 GPU/Host 占用积分和峰值。
- `scoring/manifest.json`：实际评分协议和评分成本。
- `summary.json`、`report.md`：各 benchmark 的方法对比与差值。

Qwen token 仅统计固定生成模型的调用；辅助编码费用计入完整响应、辅助耗时与物理占用。评分结束后清理临时原文评分输入和会话数据库，不保存候选计划、逐步恢复日志或重复归档。

[已有实测结果](final_results.md)及 `results/` 两份 JSON 保留重构前实验结论。本轮效果以以上新资产运行后产生的结果为准。
