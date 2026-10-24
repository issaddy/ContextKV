# Qwen3 Linux/NVIDIA 部署

当前仅使用已有 Qwen3-8B/32B 本地快照。Qwen2.5-7B 已停用，旧下载、隐式启动和恢复入口已移除。用户容器项目目录为 `/home/relay/songjiayun/ContextKV`；模型位于 `/media/ssd1/qwen3-8b`、`/media/ssd1/qwen3-32b`。代理从私有运行环境继承，凭据不进入源码、示例或公开产物。

## 准备已有模型与固定引擎

在项目根目录执行：

```bash
export CONTEXTKV_MODEL_SCOPE_PATH="$PWD/configs/model_scopes/qwen3-8b.local.json"
bash scripts/bootstrap_sglang.sh
source .venv-sglang/bin/activate
export CUDA_VISIBLE_DEVICES=0
export NO_PROXY=127.0.0.1,localhost
export no_proxy="$NO_PROXY"
export CONTEXTKV_FINGERPRINT_JSON="$(cat runs/fingerprint.json)"
```

bootstrap只安装固定 SGLang commit `3f36fc08331f64cd63c4fbbdf39cc402f38bb38b` 及其依赖，并通过现有scope核对用户已有文件、tokenizer和模板，不下载生成模型。改变本地模型目录须重新生成/核对scope，不能修改路径后沿用不匹配身份。实际HF loading revision为null，metadata identity不冒充权重内容验证。见[模型scope](model_scope.md)。

控制器的 `uv.lock` 与独立 `.venv-sglang` 分工不变；引擎依赖由固定上游pyproject定义。记录实际安装环境，不用CPU锁替代引擎依赖。

## 启动与使用

下面是已有H200功能检查所用的一组明确资源参数，不是性能最优配置。启动前由操作者确认所用GPU与端口空闲；脚本不停止其他服务。

```bash
CONTEXTKV_HICACHE_GB=8 CONTEXTKV_HICACHE_LAYOUT=layer_first \
SGLANG_HICACHE_TMA_TRANSFER=0 bash scripts/launch_sglang.sh \
  --host 127.0.0.1 --port 30000 --max-total-tokens 16384 \
  --mem-fraction-static 0.8 --max-running-requests 8 \
  --attention-backend fa3 --chunked-prefill-size 8192 --schedule-policy fcfs
```

真实运行配置为 `configs/qwen3_8b_scope.example.yaml`。在另一终端使用同一引擎环境启动 `controller.cli ... serve`，或按[项目入口](../README.md)运行P2/P3。32B须同时更换scope和共享配置，不共享8B的成本artifact。容量以实际服务池/分配记录为准，不以请求的比例或GB代替。

服务变更后的必要协议检查可使用 `python -m engine_patches.smoke --config configs/qwen3_8b_scope.example.yaml --report runs/qwen3-smoke.json`。它会实际生成，只在引擎/模型变更需要确认时运行，不是每次文档或分析变更后的例行实验。CPU检查与真实运行边界见[引擎说明](../engine_patches/README.md)。

## 分析和实验的边界

正常服务不启用冷缓存标定。只有需要解决具体成本不确定性时，在独立服务显式设置 `CONTEXTKV_ENABLE_PROFILING=1` 并选定必要测点；该接口会改变缓存状态。旧7B或其他recipe的测量仍可只读分析，不能绑定到Qwen3执行。

原生B0/B1使用clean stock checkout；研究控制器使用patched checkout。优先让[owned显式矩阵](b6_owned_matrix.md)管理各自进程、生成真实manifest并保留完整失败记录。该入口不恢复外部服务；后续supervisor也不得恢复已退役7B。默认plan不启动模型，`--execute`才执行所声明的有限任务。

共享配置、原始输入/输出、源码、实际资源、失败及后续恢复信息一起保留。已完成的[四调用验收](owned_local_scope_validation.md)属于当时源码；本轮scope-only清理先以源码/CPU合同验证，不将旧批次改写为新版本硬件证据。
