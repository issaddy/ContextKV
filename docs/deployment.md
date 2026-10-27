# Qwen3 Linux/NVIDIA 部署

当前仅使用已有 Qwen3-8B/32B 本地快照。Qwen2.5-7B 已停用，旧下载、隐式启动和恢复入口已移除。用户容器项目目录为 `/home/relay/songjiayun/ContextKV`；模型位于 `/media/ssd1/qwen3-8b`、`/media/ssd1/qwen3-32b`。代理从私有运行环境继承，凭据不进入源码、示例或公开产物。

## 准备已有模型与固定引擎

在项目根目录执行：

```bash
export CONTEXTKV_MODEL_SCOPE_PATH="$PWD/configs/model_scopes/qwen3-8b.local.json"
bash scripts/bootstrap_sglang.sh
source .venv-sglang/bin/activate
export CUDA_VISIBLE_DEVICES=0
export RAYON_NUM_THREADS=4 TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
export NO_PROXY=127.0.0.1,localhost
export no_proxy="$NO_PROXY"
export CONTEXTKV_FINGERPRINT_JSON="$(cat runs/fingerprint.json)"
```

bootstrap只安装固定 SGLang commit `3f36fc08331f64cd63c4fbbdf39cc402f38bb38b` 及其依赖，并通过现有scope核对用户已有文件、tokenizer和模板，不下载生成模型。更换模型目录时更新scope。实际HF loading revision为null，scope记录元数据身份。见[模型scope](guides/model_scope.md)。

控制器的 `uv.lock` 与独立 `.venv-sglang` 分工不变；引擎依赖由固定上游pyproject定义。记录实际安装环境，不用CPU锁替代引擎依赖。

## 启动与使用

下面采用已有H200功能检查的资源参数，运行时选择空闲GPU与端口。

```bash
CONTEXTKV_HICACHE_GB=8 CONTEXTKV_HICACHE_LAYOUT=layer_first \
SGLANG_HICACHE_TMA_TRANSFER=0 bash scripts/launch_sglang.sh \
  --host 127.0.0.1 --port 30000 --max-total-tokens 16384 \
  --mem-fraction-static 0.8 --max-running-requests 8 \
  --attention-backend fa3 --chunked-prefill-size 8192 --schedule-policy fcfs
```

真实运行配置为 `configs/qwen3_8b_scope.example.yaml`。在另一终端使用同一引擎环境启动 `controller.cli ... serve`，或按[项目入口](../README.md)运行P2/P3。32B须同时更换scope和共享配置，不共享8B的成本artifact。容量以实际服务池/分配记录为准，不以请求的比例或GB代替。

仓库同时提供可直接装配的 `configs/qwen3_8b_cost.yaml`：使用 Qwen3-8B 实测 `configs/calibration/qwen3-8b-h200.json`、`native_budget` 等待估计和无固定轮数冷却的成本选择。对应效果见[评价](data_and_evaluation.md)。加载该配置时，在上述引擎启动命令前设置 `export CONTEXTKV_SAMPLE_POOL=1`，与 artifact 的 cache configuration 保持一致；实际 GPU/Host 槽数应为 16384/54272。对照运行的完整 argv、线程设置、模型和环境记录保存在 `runs/mainline-delivery/delivery-run/servers/`。正常服务无需设置 `CONTEXTKV_ENABLE_PROFILING=1`。

引擎就绪后，在项目根目录运行：

```bash
python -m controller.cli --config configs/qwen3_8b_cost.yaml serve
```

该配置的状态目录是 `runs/contextkv-service`。成本文件随源码与 sdist 提供，不需要重新采集。对应采集来源、支持范围及本地公开副本见[阶段成本说明](guides/stage_cost_model.md)。

服务变更后的必要协议检查可使用 `python -m engine_patches.smoke --config configs/qwen3_8b_scope.example.yaml --report runs/qwen3-smoke.json`。该命令实际生成回答，用于引擎或模型变更后的协议检查。CPU检查与真实运行边界见[引擎说明](../engine_patches/README.md)。

## 清理版本的配置接口

当前配置移除了`planner.min_stable_turns`与`ablation.hysteresis`；使用旧自定义YAML时删除这两项。`delta_abs/delta_rel/z`保留为成本差诊断参数，动作按预计完整成本选择。`fallback_strategy`仍用于成本模型未覆盖范围及显式shadow执行；缺报价直接报告接口错误。历史报告中的`hysteresis`仅在离线统计中读取。

自定义模板实现`render_many`，自定义引擎实现`aclose`。ByteTemplate、HuggingFaceTemplate、MockEngine和SGLangEngineClient已统一实现这些接口。显式启动原生资源监控需要安装`evaluation`依赖组；缺少依赖时在运行入口报错并关闭客户端。

## 多轮检索服务

`retriever_conversational.example.yaml`使用独立32B解释服务保持多轮任务与变化条件，8B继续生成业务回答。两个服务复用同一固定引擎，各自使用自己的scope、tokenizer和缓存池。解释服务采用原生NGRAM输出片段草稿和目标模型/语法验证，直接沿用现有temperature=0的解析协议。无需重新bootstrap或下载模型。在另一终端从项目根目录启动解释器；子shell中的设置不改变8B终端：

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
```

这三项参数仅用于32B解释实例。宽度1适配固定引擎的FA3/page64路径；沿用原生启动warmup，不把`--skip-server-warmup`带入常规服务。有限同输入诊断已显示解析耗时改善，首次调用成本、两处非逐字一致输出及完整运行范围见[评价](data_and_evaluation.md#解释服务的原生解码优化)。公开主比较使用原来的非推测解释服务，其固定运行记录保持原样；8B生成与成本配置继续使用原执行方式。

随后在控制器终端运行（线程设置沿用上文）：

```bash
CUDA_VISIBLE_DEVICES=0,1 python -m controller.cli --config configs/qwen3_8b_cost.yaml serve \
  --retriever-config configs/retriever_conversational.example.yaml \
  --evidence-corpus data/mtrag-official/cloud.jsonl --corpus-id mtrag
```

实际拓扑为8B生成GPU0、32B解释与BGE检索共享GPU1。32B常驻进程在本次16K池配置下约70,008 MiB，解释等待计入完整响应；两个模型的KV池不能当成一个池汇总。多轮示例同时连接30000生成端口和30001解释端口，完整延迟包含两者。现有owned矩阵管理其声明的生成进程；单独使用该矩阵时，解释服务由操作者按上面命令启动并记录，两者分别关闭。

NGRAM辅助服务使用原生`/server_info`、`/metrics`观测；受控对照需要清空缓存时，在服务空闲后调用原生`/flush_cache`并确认成功。8B非推测生成的控制器继续使用既有研究接口。辅助观测缺失单独记录，不能覆盖已持久化完成的业务结果。

检索示例默认`conversational`、`single`和`ranked`；可选联合选材、来源窗口和原始文档阅读按[混合检索指南](guides/hybrid_retrieval.md)配置。`evidence_selection: semantic`可与`source_window`组合，使用同一selector预算和证据元数据。任务状态、条件继承及来源选择原理见[多轮机制](research/source_grounded_query_resolution.md)。

## 分析和实验的边界

正常服务不启用冷缓存标定。只有需要解决具体成本不确定性时，在独立服务显式设置 `CONTEXTKV_ENABLE_PROFILING=1` 并选定必要测点；该接口会改变缓存状态。旧7B或其他recipe的测量仍可只读分析，不能绑定到Qwen3执行。

原生B0/B1使用clean stock checkout；研究控制器使用patched checkout。优先让[owned显式矩阵](guides/b6_owned_matrix.md)管理各自进程、生成真实manifest并保留完整失败记录。该入口不恢复外部服务；后续supervisor也不得恢复已退役7B。默认plan不启动模型，`--execute`才执行所声明的有限任务。

共享配置、原始输入/输出、源码、实际资源、失败及后续恢复信息一起保留。当前 Qwen3 成本、六会话 A/B/C 对照和适用范围见[数据与评价](data_and_evaluation.md)。每批保留实际源码身份，不将历史批次改写为最新版本证据。
