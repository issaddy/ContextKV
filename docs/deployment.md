# Qwen3-32B部署

推荐方案使用一个Qwen3-32B非思考、原生NGRAM服务。查询适配后的BGE负责实际历史稠密检索，每轮把当前原文与实际QA重建后生成；原生Radix/HiCache继续匹配共同前缀，控制器保留事务提交、epoch和引用退役。容器项目为`/home/relay/songjiayun/ContextKV`，模型为`/media/ssd1/qwen3-32b`。

## 准备与启动

首次配置时，在项目根目录准备固定引擎、查询适配器依赖与模型身份；已准备好的GPU工作区可省略bootstrap命令：

```bash
export CONTEXTKV_MODEL_SCOPE_PATH="$PWD/configs/model_scopes/qwen3-32b.local.json"
bash scripts/bootstrap_sglang.sh
source .venv-sglang/bin/activate
export CUDA_VISIBLE_DEVICES=0
export RAYON_NUM_THREADS=4 TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
export NO_PROXY=127.0.0.1,localhost
export no_proxy="$NO_PROXY"
export CONTEXTKV_FINGERPRINT_JSON="$(cat runs/fingerprint.json)"
export CONTEXTKV_SAMPLE_POOL=1
CONTEXTKV_HICACHE_GB=16 CONTEXTKV_HICACHE_LAYOUT=layer_first \
SGLANG_HICACHE_TMA_TRANSFER=0 bash scripts/launch_sglang.sh \
  --host 127.0.0.1 --port 30000 --max-total-tokens 32768 \
  --mem-fraction-static 0.65 --max-running-requests 8 \
  --attention-backend fa3 --chunked-prefill-size 8192 --schedule-policy fcfs \
  --speculative-algorithm NGRAM --speculative-ngram-max-bfs-breadth 1 \
  --speculative-num-draft-tokens 12
```

SGLang固定为`3f36fc08331f64cd63c4fbbdf39cc402f38bb38b`。补丁支持FULL模型的普通解码和原生NGRAM；实际解码参数进入缓存配置与成本适用范围。保持原生startup warmup，`/health`就绪后再开始测量。性能对照只在其自有空闲服务上清空缓存；运行中的业务缓存按原生生命周期管理。

## 多轮检索服务

同一环境的另一终端运行：

```bash
CUDA_VISIBLE_DEVICES=0 python -m controller.cli serve \
  --retriever-config configs/retriever_dialogue.example.yaml \
  --evidence-corpus data/mtrag-official/cloud.jsonl --corpus-id mtrag
```

CLI默认加载`configs/qwen3_32b.yaml`：固定32B、32K窗口、1024输出预算、每轮REBUILD。它使用已提交实际历史、当前原始证据和原生缓存，不需要成本标定或额外解释调用。BGE权重、Cloud索引及`models/bge-m3-dialogue-query-v1`按[检索指南](guides/hybrid_retrieval.md)准备。已评测查询适配器在GPU工作区可用，本地备份为`runs/recipe-validation-20261101/query-adapter-used.tar.gz`；恢复归档可得到同名`models/`目录。它是训练资产，不能以未训练BGE视作等价替代。

当前GPU工作区已将PEFT 0.21.2与Accelerate 1.15.0装入`.venv-sglang`，标准`runs/fingerprint.json`已生成；Dense5的CLI/HTTP运行不需要实验用`PYTHONPATH`。正式环境验证与依赖记录在`runs/closeout-20261101/clean-runtime-results.tar.gz`。

模型目录、tokenizer和模板与scope保持一致。当前16 GiB Host在本环境对应61056个原生槽位，实际容量以引擎报告为准。正常服务关闭`CONTEXTKV_ENABLE_PROFILING`。`qwen3_32b_scope.example.yaml`继续用于独立启动探针；`qwen3_32b_cost.yaml`保留成本策略对照，使用前按[成本指南](guides/stage_cost_model.md)绑定匹配当前解码与资源配置的实测数据。

## 可选任务解释

需要结构化任务状态或自然回访历史来源时，可配置32B解释器。它可共享上述NGRAM服务：在对应检索配置中设置`selector.base_url: http://127.0.0.1:30000`和`device: cuda:0`。`query_mode: dialogue`保留实际QA检索视图；`query_mode: conversational`让解释结果驱动检索。解释调用和选材的完整开销随该配置计入评价。

现有`retriever_conversational.example.yaml`保留独立解释实例的30001端口配置。使用该配置时，在GPU1上以同一模型、原生NGRAM、16K槽位和8 GiB Host启动30001服务；共享部署则修改上述两个字段。推荐默认方案省略`selector`；显式来源引用、资料快照迁移及引用退役继续由共同证据和控制器处理，结构化回访能力由解释配置提供。

HTTP请求与恢复见[接口指南](guides/http_retrieval.md)。原生stock与patched对照的模型、采样和缓存预算分别记录；实验只管理自己启动的进程。当前推荐方案的质量、完整响应与GPU/Host结果见[数据与评价](data_and_evaluation.md)。
