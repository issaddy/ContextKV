# ContextKV 运行手册

当前版本采用问题级总体覆盖、语义集合效用与原生缓存获取成本的联合选择。固定 Qwen3-32B，查询 LoRA 和 Jina 骨干继续复用；任务、字段、集合和停止头随当前协议更新。快速对照使用每个 benchmark、每种方法相同的 20 题。当前结果见[本轮报告](shared_smoke.md)。

当前 12 组对照已完成运行和评分：普通问答、LlamaIndex 各交付 80/80，ContextKV 交付 79/80。MultiDoc2Dial、MT-RAG 同时改善质量和 P95；VersionQA、Loong 仍低于基线。结果目录为 `runs/current/core-20`，轻量指标已汇总至 `results/core-mechanism-20-20261113.json`。

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

导入入口为 `adapters.doc2dial`、`adapters.prepare_data`、`bench.versionqa`、`adapters.loong`。已有原始资料与标准题目直接复用。三项公开 corpus 的冻结 Qwen Embedding 索引已建立，查询 LoRA 更新不需要重建文档索引；Loong 的随题资料在注册后建立来源索引。

## 辅助训练

固定 Qwen3-32B 作为标签和实际答案生成器。查询训练使用会话对齐与任务关系；证据训练先完成字段/来源头，再加入实际答案集合效用；最后采集多步停止轨迹。

现有可用资产：`models/shared-query`（任务头 222/47 状态，总体覆盖头 286/63 状态）、`models/shared-evidence`（字段 54/23、集合 24/6 状态）、`models/shared-stop.pt`（12/4 状态）。斜线两侧为训练/开发数；覆盖监督与任务监督有重叠。本轮集合和停止分别采集 180、32 次真实答案。查询 LoRA、Jina 骨干与 Qwen3-32B 权重保持固定。以下命令是训练入口，已有资产可直接用于评测。

```bash
export CUDA_VISIBLE_DEVICES=4,5,6,7

python3 -m training.prepare_shared_labels \
  --task-only --source data/multidoc2dial/source \
  --query-model models/qwen3-embedding-0.6b \
  --endpoint http://127.0.0.1:30000 \
  --output data/training/shared/query_labels.jsonl \
  --train-states 160 --dev-states 24 --concurrency 4

python3 -m training.prepare_shared_labels \
  --source data/multidoc2dial/source \
  --query-model models/qwen3-embedding-0.6b \
  --endpoint http://127.0.0.1:30000 \
  --output data/training/shared/labels.jsonl \
  --train-states 24 --dev-states 8 --candidates 6 --concurrency 4

python3 -m training.shared_training query \
  --model models/qwen3-embedding-0.6b \
  --data data/training/shared/query_labels.jsonl \
  --output models/shared-query --device cuda:2 --epochs 5

python3 -m training.prepare_shared_labels \
  --query-model models/qwen3-embedding-0.6b \
  --refresh-coverage data/training/shared/query_labels.jsonl data/training/shared/labels.jsonl \
  --output data/training/shared/query_labels.jsonl --concurrency 4

python3 -m training.shared_training coverage \
  --model models/qwen3-embedding-0.6b --init-asset models/shared-query \
  --data data/training/shared/query_labels.jsonl \
  --output models/shared-query --device cuda:2 --epochs 200 --learning-rate 0.001

python3 -m training.shared_index \
  --corpus data/multidoc2dial/test/passages.jsonl \
  --snapshot multidoc2dial-test \
  --model models/qwen3-embedding-0.6b \
  --output data/indexes/multidoc2dial --device cuda:1 --dtype bfloat16

python3 -m training.shared_training evidence \
  --model models/jina-reranker-v3.5 \
  --query-model models/qwen3-embedding-0.6b --query-asset models/shared-query \
  --data data/training/shared/labels.jsonl \
  --output models/shared-evidence --device cuda:3 --field-only --epochs 3

python3 -m training.collect_shared_values sets \
  --config configs/study.yaml --data data/training/shared/labels.jsonl \
  --output data/training/shared/sets.jsonl --train-states 24 --dev-states 6 --concurrency 4

python3 -m training.shared_training evidence \
  --model models/jina-reranker-v3.5 \
  --query-model models/qwen3-embedding-0.6b --query-asset models/shared-query \
  --init-asset models/shared-evidence --data data/training/shared/sets.jsonl \
  --output models/shared-evidence --device cuda:3 --epochs 40 --learning-rate 0.001

python3 -m training.collect_shared_values stop \
  --config configs/study.yaml --data data/training/shared/labels.jsonl \
  --output data/training/shared/stop.jsonl --train-states 12 --dev-states 4 --concurrency 4

python3 -m training.stop_value data/training/shared/stop.jsonl \
  --output models/shared-stop.pt --device cuda:2 \
  --seconds-price 0.001 --tail-price 0.001
```

停止轨迹采集前，用 `python3 -m training.calibrate_kv --config configs/study.yaml --data data/training/shared/labels.jsonl --output models/infoflow-quality.json` 完成默认 4 状态的质量配对。重算比例、质量差和实测成本系数一起由控制器读取，训练采集使用相同政策。服务重启后沿用这些系数，后续请求的实测费用继续更新它们。

MT-RAG、VersionQA 各用自己的 corpus 路径运行 `shared_index`，输出与 `study_matrix.yaml` 中的索引目录对应。独立 corpus 的索引构建可以并行。

标签集按会话分离 train/dev，包含公开训练来源和新构造的关系/版本训练资料。查询监督按每个会话两轮轮转采样，覆盖更多会话；字段监督复用独立的来源标注。集合头单独拟合，使用与采集、线上相同的推断任务表示；冻结骨干只编码一次，后续轮次复用表示。集合样本保存候选集合和相对无证据回答的实际质量增益；停止样本保存紧凑表示及多步回报需要的费用，使用线上相同的 auto 缓存选择。必须完成的数值总体由覆盖规则执行，不学习提前停止。

## 少量端到端评测

默认每个 benchmark、每种方法 20 题。`--limit-turns` 可指定规模；固定种子分散选择会话并保留相邻轮次，各方法使用相同题目与生成条件，每个 benchmark 单独评分和报告。

```bash
python3 -m bench.study evaluate \
  --config configs/study.yaml --matrix configs/study_matrix.yaml \
  --split test --limit-turns 20 --concurrency 4 \
  --output runs/current/public

python3 -m bench.study evaluate \
  --config configs/study.yaml --matrix configs/study_system_matrix.yaml \
  --split test --limit-turns 20 --concurrency 4 \
  --output runs/current/actual-history
```

`--methods ordinary_rag contextkv llamaindex` 和 `--datasets multidoc2dial mtrag versionqa loong` 可指定本次范围。公开评测使用参考历史；实际历史评测在会话内顺序执行，失败轮次结束后，后续轮次使用此前已提交历史。

## 保留结果

- `results.jsonl`：逐题答案、来源地址、质量、完整响应、Qwen 输入/输出 token 与缓存计数。
- `batch_cost.json`：整批 GPU/Host 占用积分和峰值。
- `scoring/manifest.json`：实际评分协议和评分成本。
- `summary.json`、`report.md`：各 benchmark 的方法对比与差值。

Qwen token 仅统计固定生成模型的调用；辅助编码费用计入完整响应、辅助耗时与物理占用。评分结束后清理临时原文评分输入和会话数据库，不保存候选计划、逐步恢复日志或重复归档。

评分阶段缓存只保留题目 ID 和分数，原文放在临时工作目录；裁判费用只累计汇总，不保存提示、回复正文或逐调用日志。资源采样只保留时间及 GPU/Host 占用总量，用于最终积分和停止训练的费用标签。训练所需的来源标注、候选集合质量与停止轨迹保留为训练数据。

[本轮小跑结果](shared_smoke.md)包含同题质量、完整响应及批次资源。[已有实测结果](final_results.md)及对应 JSON 保留重构前实验结论。
