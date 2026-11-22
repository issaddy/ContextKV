# ContextKV 运行手册

当前版本采用问题级总体覆盖、语义集合效用与原生缓存获取成本的联合选择。固定 Qwen3-32B，复用查询 LoRA 和 Jina 骨干；任务、字段、集合和停止头按当前协议训练。

当前总体任务使用完整来源编码和按文档覆盖的证据投影，普通任务使用学习到的集合效用和停止决策。固定生成模型只接收预算内原文与计算结果。总体上下文使用新增 KV 页预算；跨轮来源需求和稳定引用头连接原生缓存保留。当前效果见[实测报告](results.md)。

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

当前默认资产：`models/shared-query`（任务头 286/63、总体覆盖头 382/95 个训练/开发状态）、`models/shared-evidence-sections`（沿用字段头，章节集合 24/6 状态）、`models/shared-stop-sections.pt`（24/6 会话、90 个终端答案）。集合头使用 180 个实际答案标签，来源为 MultiDoc2Dial 官方训练会话；评测会话独立。集合头为 7 参数信息增益校准，停止头使用 16 维共享摘要、12 维状态与 32 维隐藏层。候选采用作者章节，扩展批大小直接读取 `retrieval.evidence_batch`，当前为 12。停止特征包含下一批未编码来源的 token 量及当前证据的实际缓存报价。固定生成权重，查询和字段权重本轮未更新。

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
config["shared"]["runtime"]["kv_quality_asset"] = str(run / "kv.json")
config["shared"]["runtime"]["stop_asset"] = str(run / "stop.pt")
(run / "study.yaml").write_text(yaml.safe_dump(config))
PYCFG

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
  --model models/qwen3-embedding-0.6b --init-asset models/shared-query \
  --data data/training/shared/query_labels.jsonl \
  --output "$TRAIN_RUN/query" --device cuda:2 --heads-only --epochs 20 --learning-rate 0.0002

python3 -m training.prepare_shared_labels \
  --query-model models/qwen3-embedding-0.6b \
  --refresh-coverage data/training/shared/query_labels.jsonl data/training/shared/labels.jsonl \
  --output data/training/shared/query_labels.jsonl --concurrency 4

python3 -m training.shared_training coverage \
  --model models/qwen3-embedding-0.6b --init-asset "$TRAIN_RUN/query" \
  --data data/training/shared/query_labels.jsonl \
  --output "$TRAIN_RUN/query" --device cuda:2 --epochs 200 --learning-rate 0.001

python3 -m training.shared_index \
  --corpus data/multidoc2dial/test/passages.jsonl \
  --snapshot multidoc2dial-test \
  --model models/qwen3-embedding-0.6b \
  --output data/indexes/multidoc2dial --device cuda:1 --dtype bfloat16

python3 -m training.shared_training evidence \
  --model models/jina-reranker-v3.5 \
  --query-model models/qwen3-embedding-0.6b --query-asset "$TRAIN_RUN/query" \
  --init-asset models/shared-evidence --data data/training/shared/labels.jsonl \
  --output "$TRAIN_RUN/evidence" --device cuda:3 --field-only --epochs 1

python3 -m training.prepare_shared_labels \
  --values-only --source data/multidoc2dial/source \
  --query-model models/qwen3-embedding-0.6b \
  --endpoint http://127.0.0.1:30000 \
  --output data/training/shared/value_states.jsonl \
  --train-states 24 --dev-states 8 --concurrency 4

python3 -m training.calibrate_kv --config "$TRAIN_RUN/study.yaml" \
  --data data/training/shared/labels.jsonl --output "$TRAIN_RUN/kv.json" --concurrency 4

python3 -m training.collect_shared_values sets \
  --config "$TRAIN_RUN/study.yaml" --data data/training/shared/value_states.jsonl \
  --output data/training/shared/sets.jsonl --train-states 24 --dev-states 6 --concurrency 4

python3 -m training.shared_training evidence \
  --model models/jina-reranker-v3.5 \
  --query-model models/qwen3-embedding-0.6b --query-asset "$TRAIN_RUN/query" \
  --init-asset "$TRAIN_RUN/evidence" --data data/training/shared/sets.jsonl \
  --output "$TRAIN_RUN/evidence" --device cuda:3 --epochs 80 --learning-rate 0.001

python3 -m training.collect_shared_values stop \
  --config "$TRAIN_RUN/study.yaml" --data data/training/shared/value_states.jsonl \
  --output data/training/shared/stop.jsonl --train-states 12 --dev-states 4 --concurrency 4

python3 -m training.stop_value data/training/shared/stop.jsonl \
  --output "$TRAIN_RUN/stop.pt" --device cuda:2 \
  --seconds-price 0.001 --tail-price 0.001
```

训练产生的查询头、集合头、停止头和成本系数写入本次实验目录。Native 和 InfoFlow 分批执行；随后每状态一次单 token 重访测量可复用阶段费用，首次编译与冷建的已付时间保留。重算比例、质量差和实测成本系数一起由控制器读取，训练采集使用相同政策。服务重启后沿用这些系数，后续请求的实测费用继续更新它们。

范围标签按整题与各子需求分别生成。补采自然样本时，用 `--exclude-labelled-sessions` 指定已有标注文件，保留官方训练对话的原划分，并轮转采样不同深度的轮次。WikiSQL 范围样本保留原问题、表 ID、SQL 和匹配行数；同一百科页面只进入一个划分。容器中范围与集合监督位于 `data/training/shared/aux_supervision/{scope,sets}.jsonl`。单独重训现有范围头时，使用 `training.shared_training coverage --init-asset models/shared-query --data data/training/shared/aux_supervision/scope.jsonl`，其余参数沿用上面的命令，输出到新的训练目录。

跨文档配对样本位于容器 `data/training/shared/aux_supervision/manifest_dialogues.jsonl`。44 个构造会话按软件仓库划分为 24 个训练会话和 20 个开发会话，共 132 轮；第二轮保存 306 次证据集合干预的回答标签。

实际多轮停止轨迹来自 MultiDoc2Dial 官方训练会话。按声明题单运行正常 P3 问答，用上一轮提交给引擎的提示预热，分别保留 GPU 需求或释放到 Host。每次扩展 12 段，记录当前实际回答的 F1、引擎回执及继续一步的费用。同一条轨迹中，输入 token 和执行路径均相同的状态共用平均质量标签，各自保留实测费用。

```bash
asset=training/assets/multidoc2dial-stop-dialogues.jsonl.gz
python3 -m training.prepare_stop_histories --config "$TRAIN_RUN/study.yaml" \
  --targets "$asset" --tasks data/multidoc2dial/train/tasks.jsonl \
  --output "$TRAIN_RUN/stop-inputs.jsonl" --gpu-devices 4 5 --concurrency 4 --seed 20261122
for tier in gpu host; do
  python3 -m training.collect_shared_values stop --config "$TRAIN_RUN/study.yaml" \
    --data "$TRAIN_RUN/stop-inputs.jsonl" --seed-tier "$tier" --batch-size 12 \
    --gpu-devices 4 5 --train-states 48 --dev-states 24 --concurrency 4 --output "$TRAIN_RUN/stop-$tier.jsonl"
done
python3 -m training.stop_value "$asset" --output "$TRAIN_RUN/stop.pt" \
  --seconds-price 0.001 --tail-price 0.001 --seed 42
```

InSCIt 自然对话使用作者训练分区。每个会话取一条引用多篇文章的回答，以及最近的一条引用单篇文章的回答，两题都保留实际对话历史。干预集合包含作者引用的段落、逐篇移除后的段落，以及在被移除段落的原位置等量填入历史段落的组合。文章数量描述该回答的来源，逐事实覆盖由固定生成模型的实际回答评定。[固定版本的原始数据](https://github.com/ellenmellon/INSCIT/blob/2319fb85932b9528a13417223ffc6fc629ae8087/data/train.json)

```bash
python3 -m training.prepare_inscit data/training/inscit/train.json "$TRAIN_RUN/inscit-pairs.jsonl" \
  --tokenizer /data/models/qwen3-32b
python3 -m training.prepare_shared_labels --states "$TRAIN_RUN/inscit-pairs.jsonl" \
  --train-states 0 --dev-states 0 --controlled-train-families 0 --controlled-dev-families 0 \
  --query-model models/qwen3-embedding-0.6b --label-context-tokens 8192 \
  --output "$TRAIN_RUN/inscit-labels.jsonl"
python3 -m training.collect_shared_values sets --config configs/study.yaml \
  --data "$TRAIN_RUN/inscit-labels.jsonl" --output "$TRAIN_RUN/inscit-sets.jsonl"
```

容器 `data/training/shared/aux_supervision/inscit-wide-sets.jsonl` 保存 24 段候选池的段落 ID、集合特征和回答标签，可直接作为集合头的 `--data`。同目录的 `inscit-orders.jsonl` 按题记录两种文章顺序的答案得分、事实覆盖率、输出长度和引擎计数。每种顺序先用上一轮作者引用的证据做一次单 token 预填充，再用实际历史回答当前问题。文件中的 `ms` 是本次生成调用耗时。

顺序实验的逐事实文件每题有四个 `trials`。`evidence_order` 和 `reference_order` 分别索引同一行的 `orders`，前者决定段落在提示中的位置，后者决定从 S1 开始的来源编号。`covered_facts` 与 `reference_facts` 逐项对应，交换答案展示顺序后两次判定有分歧的记为 `null`。每次测量先清空引擎缓存，再预填充上一轮原文。并发为 1；`first_token_ms` 从调度器接收到预填充完成，`decode_ms` 为随后至生成结束的时长。开发题的 `matched_output` 保存等长输出时的引擎耗时；`primed_articles` 中的两组各预热一篇文章，组内两侧仍使用相同的问题、历史和证据。每组的两个测量值按 `orders` 排列。[顺序实验数据](../../training/assets/inscit-order-facts.jsonl)

逐事实配对文件可直接传入集合头训练的 `--data`。每行包含只给历史的对照和两个等长证据集。`sets[].quality` 是该条参考事实被回答且有可见材料支持的二值标签，`answer_quality` 保留整题得分，`covered_facts` 记录回答是否命中该事实。`fact_witness` 的字符区间为左闭右开，可按题号和来源回到 `inscit-sets.jsonl` 的原段落或历史发言核查。[逐事实配对文件](../../training/assets/inscit-supported-swaps.jsonl)

集合采集使用线上首批可见证据，比较当前选择、相邻集合与空集合。固定教师先根据参考答案拆分必要事实，再匿名评判各答案。每个集合保存答案质量、逐事实覆盖标记、输出长度和选择特征。MultiDoc2Dial 用参考答案 F1，其余训练来源用固定教师对照参考答案评分。开发报告中的 `dev_chosen_quality` 和 `dev_chosen_fact_recall` 取自有标签的干预集合中效用最高的一项。线上选择还要考虑完整候选集合、KV 报价和容量。字段损失只使用与标注对齐的子需求。范围头的开发指标使用与服务一致的整题和子需求联合路由。

MT-RAG、VersionQA 各用自己的 corpus 路径运行 `shared_index`，输出与 `study_matrix.yaml` 中的索引目录对应。独立 corpus 的索引构建可以并行。

标签集按会话分离 train/dev，包含公开训练来源和新构造的关系/版本训练资料。查询监督轮转覆盖早期和后续轮次。仅更新任务头时仍训练 Need 与冻结检索空间的对齐。集合拟合直接读取采集时保存的特征。集合/停止采集共用线上读取、状态准备、回答编译和预算执行器；最终质量来自实际结论与原文组成的回答，扩展费用包含新增编码、读取和计算。集合拟合以同题空集合的答案质量为基准；停止样本保存紧凑表示与多步回报费用，使用线上相同的 auto 缓存选择。必须完成的数值总体由覆盖规则执行，不学习提前停止。

## 少量端到端评测

默认每个 benchmark、每种方法 20 题。会话内按轮号的数值排序，每次从各会话依次取相邻两轮，最后用 `--limit-turns` 截断整套队列。使用预先选定的题单时，该值应至少等于题数最多的 benchmark，否则只会评测部分题目。各方法使用同一题单和生成条件，按 benchmark 分别报告质量和完整 P95。

```bash
python3 -m bench.study evaluate \
  --config "$TRAIN_RUN/study.yaml" --matrix configs/study_matrix.yaml \
  --split test --limit-turns 20 --concurrency 4 \
  --output runs/current/public

python3 -m bench.study evaluate \
  --config "$TRAIN_RUN/study.yaml" --matrix configs/study_system_matrix.yaml \
  --split test --limit-turns 20 --concurrency 4 \
  --output runs/current/actual-history
```

`--methods ordinary_rag contextkv llamaindex` 和 `--datasets multidoc2dial mtrag versionqa loong` 可指定本次范围。公开评测使用参考历史；实际历史评测在会话内顺序提交导出的可评分轮次，失败轮次结束后，后续轮次使用此前已提交历史。

连续会话可在 matrix 的 `cohorts.test` 中指定真实相邻轮次；当前清单保存在 `results/cooperation_optimization.json` 的 `cohorts`。添加 `--cache-pair` 会在 ContextKV 生成后，保持实际证据成员、问题、已生成历史与到达偏移一致，先执行语义排序 Native 和缓存成本排列的 auto，再以 Native 重放 auto 使用的相同来源顺序，并在相同顺序下以 auto 省略未来来源需求。各路分别从冷 KV 开始，使用贪心生成并保留答案一致数。第三路直接读取已选顺序，省去在线报价时间。请求只在内存中暂存，输出 `cache_pair.json`；该配对的 P95 从输入就绪计时，完整队列的 P95 包含检索与辅助编码，质量按各 benchmark 原协议评分。

KV 峰值直接读取页池，同时报告总驻留和不可淘汰的活动页峰值。`new_compute_kv_pages` 累计实际计算跨过的页边界，不包含 Host 搬运页；共享驻留容量按物理页计算一次。归还页会增加预分配池的可用容量，不会自动缩小该池。物理 GPU 占用按在线设备采样，评分设备不计入在线阶段。缓存配对同时给出参考 token F1；MT-RAG 的该指标与主队列官方综合分分别报告。

## 单请求引擎诊断

此入口用于诊断预填充和解码开销。每条分支使用相同证据与逆序预热，分别执行原顺序 Native、按实际缓存成本排列的自动路径、强制 InfoFlow。一次输出完整回答，一次仅生成一个 token 分离预填充开销；预热费用随结果保留。连续会话归因使用上面的 `--cache-pair`。

```bash
python3 -m bench.cache_reuse \
  --config configs/study.yaml --data data/training/shared/labels.jsonl \
  --task-id 611468ee1567561df83307d715e1abf5_1 \
  --output runs/current/cache-pair.json

python3 -m bench.cache_reuse \
  --config configs/study.yaml --data data/training/shared/labels.jsonl \
  --task-id 611468ee1567561df83307d715e1abf5_1 \
  --prefill-only --output runs/current/cache-prefill.json
```

## 保留结果

- `results.jsonl` 保存逐题答案、来源地址、质量、完整响应、Qwen 输入与输出 token 及缓存计数。已完成请求的 `slot_wait_ms` 记录等待并发名额的时间，已计入 `complete_ms`。
- `batch_cost.json`：整批 GPU/Host 占用积分和峰值。
- `scoring/manifest.json`：实际评分协议和评分成本。
- `summary.json`、`report.md`：各 benchmark 的方法对比与差值。

Qwen token 仅统计固定生成模型的调用；辅助编码费用计入完整响应、辅助耗时与物理占用。评分结束后清理临时原文评分输入和会话数据库，不保存候选计划、逐步恢复日志或重复归档。

评分阶段缓存只保留题目 ID 和分数，原文放在临时工作目录；裁判费用只累计汇总，不保存提示、回复正文或逐调用日志。资源采样只保留时间及 GPU/Host 占用总量，用于最终积分和停止训练的费用标签。训练所需的来源标注、候选集合质量与停止轨迹保留为训练数据。

[当前结果](results.md)包含各 benchmark 的同题质量、完整响应及批次资源；[连续会话指标](../../results/cooperation_optimization.json)与[公开参考历史指标](../../results/current.json)分别对应容器中的逐题记录。
