# ContextKV 运行手册

默认固定 Qwen3-32B，沿用现有查询、字段和集合资产，使用 Native 精确执行与明确消费者需求。固定总体先匹配适用结果、判断完整原文能否直接回答，再进行必要绑定或读取。

**训练保持暂停。启动与评测不需要 STOP 权重或 InfoFlow 校准资产。** 本版清理尚未验证端到端质量与 P95 收益；历史数据见[实测报告](results.md)。

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

## 辅助资产更新（当前暂停）

现有资产可直接用于默认主链。以下命令仅供以后更新查询、字段和集合资产时使用，固定 Qwen3-32B 生成标签与真实答案。每次新采集写入独立的 `TRAIN_RUN`，同次中断才续写，避免沿用已经改变题意的旧标签。

```bash
export CUDA_VISIBLE_DEVICES=4,5,6,7
mkdir -p runs
TRAIN_RUN=$(mktemp -d runs/train-XXXXXX)
export TRAIN_RUN
python3 - <<'PYCFG'
from pathlib import Path
import os
import yaml

run = Path(os.environ["TRAIN_RUN"])
config = yaml.safe_load(Path("configs/study.yaml").read_text())
config["shared"]["query"]["asset_path"] = str(run / "query")
config["shared"]["evidence"]["asset_path"] = str(run / "evidence")
(run / "study.yaml").write_text(yaml.safe_dump(config))
PYCFG

python3 -m training.prepare_shared_labels \
  --task-only --source data/multidoc2dial/source \
  --query-model models/qwen3-embedding-0.6b \
  --endpoint http://127.0.0.1:30000 \
  --output "$TRAIN_RUN/query_labels.jsonl" \
  --train-states 160 --dev-states 24 --concurrency 4

python3 -m training.prepare_shared_labels \
  --source data/multidoc2dial/source \
  --query-model models/qwen3-embedding-0.6b \
  --endpoint http://127.0.0.1:30000 \
  --output "$TRAIN_RUN/labels.jsonl" \
  --train-states 24 --dev-states 8 --candidates 6 --concurrency 4

python3 -m training.shared_training query \
  --model models/qwen3-embedding-0.6b --init-asset models/shared-query \
  --data "$TRAIN_RUN/query_labels.jsonl" \
  --output "$TRAIN_RUN/query" --device cuda:2 --heads-only --epochs 20 --learning-rate 0.0002

python3 -m training.prepare_shared_labels \
  --query-model models/qwen3-embedding-0.6b \
  --refresh-coverage "$TRAIN_RUN/query_labels.jsonl" "$TRAIN_RUN/labels.jsonl" \
  --output "$TRAIN_RUN/query_labels.jsonl" --concurrency 4

python3 -m training.shared_training coverage \
  --model models/qwen3-embedding-0.6b --init-asset "$TRAIN_RUN/query" \
  --data "$TRAIN_RUN/query_labels.jsonl" \
  --output "$TRAIN_RUN/query" --device cuda:2 --epochs 200 --learning-rate 0.001

python3 -m training.shared_index \
  --corpus data/multidoc2dial/test/passages.jsonl \
  --snapshot multidoc2dial-test \
  --model models/qwen3-embedding-0.6b \
  --output data/indexes/multidoc2dial --device cuda:1 --dtype bfloat16

python3 -m training.shared_training evidence \
  --model models/jina-reranker-v3.5 \
  --query-model models/qwen3-embedding-0.6b --query-asset "$TRAIN_RUN/query" \
  --init-asset models/shared-evidence --data "$TRAIN_RUN/labels.jsonl" \
  --output "$TRAIN_RUN/evidence" --device cuda:3 --field-only --epochs 1

python3 -m training.prepare_shared_labels \
  --values-only --source data/multidoc2dial/source \
  --query-model models/qwen3-embedding-0.6b \
  --endpoint http://127.0.0.1:30000 \
  --output "$TRAIN_RUN/value_states.jsonl" \
  --train-states 24 --dev-states 8 --concurrency 4

python3 -m training.collect_shared_values \
  --config "$TRAIN_RUN/study.yaml" --data "$TRAIN_RUN/value_states.jsonl" \
  --output "$TRAIN_RUN/sets.jsonl" --train-states 24 --dev-states 6 --concurrency 4

python3 -m training.shared_training evidence \
  --model models/jina-reranker-v3.5 \
  --query-model models/qwen3-embedding-0.6b --query-asset "$TRAIN_RUN/query" \
  --init-asset "$TRAIN_RUN/evidence" --data "$TRAIN_RUN/sets.jsonl" \
  --output "$TRAIN_RUN/evidence" --device cuda:3 --epochs 80 --learning-rate 0.001

```

集合采集使用线上首批可见证据，比较当前选择、相邻集合与空集合。保存实际答案质量、参考事实覆盖和集合特征；来源支持仅用于软排序，不作为完成判据。CLI 没有 `sets/stop` 位置参数，不再采集停止轨迹。

训练数据按会话划分 train/dev，并轮转选择不同深度的轮次。已有文档索引保持冻结编码空间；仅更新查询头时无需重建。MT-RAG、VersionQA 的独立索引路径与 `study_matrix.yaml` 对应。

InfoFlow 仅供显式实验：需配置 `kv_mode: auto` 或 `infoflow` 并提供 `kv_quality_asset`。现有小样本全局质量均值不构成长证据质量保证；默认评测无需运行该校准，也不将其作为启动依赖。

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

`--methods ordinary_rag contextkv llamaindex` 和 `--datasets multidoc2dial mtrag versionqa loong` 可指定本次范围。公开评测使用参考历史；实际历史评测在会话内顺序提交导出的可评分轮次，失败轮次结束后，后续轮次使用此前已提交历史。

连续会话可在 matrix 的 `cohorts.test` 中指定真实相邻轮次。`--cache-pair` 与 `bench.cache_reuse` 属于显式缓存实验，应使用与所测路线匹配的配置和校准资产；其执行阶段 P95 与包含检索、辅助编码和排队的完整 P95 分别报告。

KV 峰值直接读取页池，同时报告总驻留和不可淘汰的活动页峰值。`new_compute_kv_pages` 累计实际计算跨过的页边界，不包含 Host 搬运页；共享驻留容量按物理页计算一次。归还页会增加预分配池的可用容量，不会自动缩小该池。物理 GPU 占用按在线设备采样，评分设备不计入在线阶段。缓存配对同时给出参考 token F1；MT-RAG 的该指标与主队列官方综合分分别报告。

## 保留结果

- `results.jsonl`：逐题答案、来源地址、质量、完整响应、Qwen 输入/输出 token 与缓存计数。
- `batch_cost.json`：整批 GPU/Host 占用积分和峰值。
- `scoring/manifest.json`：实际评分协议和评分成本。
- `summary.json`、`report.md`：各 benchmark 的方法对比与差值。

Qwen token 仅统计固定生成模型的调用；辅助编码费用计入完整响应、辅助耗时与物理占用。评分结束后清理临时原文评分输入和会话数据库，不保存候选计划、逐步恢复日志或重复归档。

评分阶段缓存只保留题目 ID 和分数，原文放在临时工作目录；裁判费用只累计汇总，不保存提示、回复正文或逐调用日志。资源采样只保留时间及 GPU/Host 占用总量，用于最终积分。来源标注与候选集合质量作为训练数据保留。

[当前结果](results.md)包含各 benchmark 的同题质量、完整响应及批次资源；[连续会话指标](../../results/cooperation_optimization.json)与[公开参考历史指标](../../results/current.json)分别对应容器中的逐题记录。
