# Qwen3-32B部署

默认生成、任务解释和真实模型测试使用本地Qwen3-32B非思考模式。容器项目为`/home/relay/songjiayun/ContextKV`，模型为`/media/ssd1/qwen3-32b`。BGE/BM25保持检索职责。

## 准备与启动生成服务

在项目根目录准备固定引擎和模型身份：

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
  --mem-fraction-static 0.8 --max-running-requests 8 \
  --attention-backend fa3 --chunked-prefill-size 8192 --schedule-policy fcfs
```

固定SGLang提交为`3f36fc08331f64cd63c4fbbdf39cc402f38bb38b`。模型目录、tokenizer和模板必须与scope一致；实际HF loading revision为null，scope保留本地快照身份。模型或资源改变后重新核对[身份与采样](deployment.md)和[成本适用范围](guides/stage_cost_model.md)。业务生成保持非推测解码，辅助服务的NGRAM设置不复制到C生命周期路径。

在另一终端使用同一环境：

```bash
python -m profiling.smoke --config configs/qwen3_32b_scope.example.yaml \
  --report runs/qwen3-32b-smoke.json
python -m controller.cli --config configs/qwen3_32b_cost.yaml serve
```

`qwen3_32b_cost.yaml`加载当前32B实测成本；`qwen3_32b_scope.example.yaml`保留固定REBUILD，供新环境启动和补采。模型scope绑定本地配置、分词器和模板；更换模型目录时先用`python -m engine_patches.contextkv_model_scope --help`查看检查入口。真实资源以引擎报告为准，当前16GB Host对应61056个KV槽位。正常服务关闭`CONTEXTKV_ENABLE_PROFILING`。

## 多轮检索服务

任务解释使用另一个32B实例，生成GPU0、解释与BGE检索共享GPU1。两实例使用独立缓存池，完整响应计入全部在线调用。在另一终端启动解释服务：

```bash
(
  source .venv-sglang/bin/activate
  export CUDA_VISIBLE_DEVICES=1
  export CONTEXTKV_MODEL_SCOPE_PATH="$PWD/configs/model_scopes/qwen3-32b.local.json"
  export CONTEXTKV_MODEL_PATH=/media/ssd1/qwen3-32b
  export CONTEXTKV_FINGERPRINT_JSON="$(python -c 'import json; from dataclasses import asdict; from controller.config import load_config, create_model_template; print(json.dumps(asdict(create_model_template(load_config("configs/qwen3_32b_scope.example.yaml").model).fingerprint)))')"
  CONTEXTKV_SAMPLE_POOL=1 CONTEXTKV_HICACHE_GB=8 \
  bash scripts/launch_sglang.sh --host 127.0.0.1 --port 30001 \
    --max-total-tokens 16384 --mem-fraction-static 0.65 --max-running-requests 8 \
    --attention-backend fa3 --chunked-prefill-size 8192 --schedule-policy fcfs \
    --speculative-algorithm NGRAM --speculative-ngram-max-bfs-breadth 1 \
    --speculative-num-draft-tokens 12
)
CUDA_VISIBLE_DEVICES=0,1 python -m controller.cli --config configs/qwen3_32b_cost.yaml serve \
  --retriever-config configs/retriever_conversational.example.yaml \
  --evidence-corpus data/mtrag-official/cloud.jsonl --corpus-id mtrag
```

NGRAM仅用于任务解释实例，由原生目标模型和语法验证。保持启动warmup；需要清空辅助缓存时，先确认服务空闲，再使用原生`/flush_cache`。生成端使用研究接口维护业务epoch和退役状态。

检索默认`conversational`、`single`和`ranked`，配置见[检索指南](guides/hybrid_retrieval.md)。HTTP请求、幂等和恢复见[接口指南](guides/http_retrieval.md)。原生B0/B1对照使用clean stock checkout，研究控制器使用patched checkout；[owned矩阵](data_and_evaluation.md)只管理其声明的进程，解释服务需单独记录和关闭。
