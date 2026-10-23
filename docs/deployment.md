# Linux/NVIDIA 部署与实测

本项目在独立虚拟环境安装固定 SGLang 源码，避免更改容器原有的 torch、transformers 或服务。用户提供容器的项目目录是 `/home/relay/songjiayun/ContextKV`。先用单 GPU 验证，并为本项目选择空闲端口。代理从运行环境继承；凭据只存放于权限为 `600` 的 `runs/network.env` 等忽略文件中，不进入命令示例、日志和版本库。

## 安装固定版本

在项目根目录执行：

```bash
# 需要代理时，先加载自己的运行配置。
source runs/network.env
bash scripts/bootstrap_sglang.sh
source .venv-sglang/bin/activate
export CUDA_VISIBLE_DEVICES=0
export NO_PROXY=127.0.0.1,localhost
export no_proxy="$NO_PROXY"
export CONTEXTKV_MODEL_PATH="$PWD/models/Qwen2.5-7B-Instruct"
export CONTEXTKV_FINGERPRINT_JSON="$(cat runs/fingerprint.json)"
```

脚本固定 SGLang `3f36fc08331f64cd63c4fbbdf39cc402f38bb38b` 和 Qwen2.5-7B-Instruct `a09a35458c702b33eeacc393d103063234e8bc28`。引擎自身 pyproject 规定 torch/CUDA/attention 依赖，安装后的完整列表记录于 `runs/engine-environment.txt`。主项目的 `uv.lock` 是 CPU 控制器环境锁文件，不能用它替代引擎依赖。默认跳过原生 Rust 扩展，使用固定的 Python TreeCore。

## 分阶段验证

先关闭 host tier，确认模型、CUDA kernel、IPC 与业务协议：

```bash
CONTEXTKV_ENABLE_HICACHE=0 bash scripts/launch_sglang.sh \
  --host 127.0.0.1 --port 30000 --mem-fraction-static 0.15 --max-running-requests 8
```

这里的显存比例是 H200 集成检查示例。显存较小的卡应按模型权重与 KV 池容量调整；正式实验记录实际池容量，不将比例相同当作容量相同。在另一终端激活相同环境并设置相同模型路径后执行：

```bash
python -m engine_patches.smoke --config configs/sglang.yaml --report runs/gpu-smoke.json
```

通过后停止本项目进程，再启用 host tier 和隔离标定接口：

```bash
CONTEXTKV_ENABLE_PROFILING=1 CONTEXTKV_HICACHE_GB=16 \
  CONTEXTKV_HICACHE_LAYOUT=layer_first SGLANG_HICACHE_TMA_TRANSFER=0 \
  bash scripts/launch_sglang.sh --host 127.0.0.1 --port 30000 \
  --mem-fraction-static 0.15 --max-running-requests 8
python -m engine_patches.smoke --config configs/sglang.yaml --report runs/host-smoke.json
python -m profiling.sglang_probe --config configs/sglang.yaml \
  --lengths 128 --prefix-fractions 0,.5 --outputs 8 --concurrency 1 \
  --warmup 1 --repetitions 3 --output runs/calibration-smoke.jsonl
```

`CONTEXTKV_ENABLE_PROFILING=1` 的冷缓存/host-only 准备会改变该服务器缓存状态，只能在独立标定服务使用。冒烟网格只验证测量管线，不能覆盖真实多轮请求范围。正式标定扫描开发集长度、prefix/suffix、host-load、输出长度和并发区域，随后运行 `python -m profiling.fit MEASUREMENTS --output ARTIFACT`，检查留出误差和覆盖率，再设置 `cost_model.artifact` 与实际控制开销门槛。

控制器直接使用本地模型时，可将运行配置的 `model.name` 指向模型目录，设置 `model.local_files_only: true`。模型 revision、tokenizer 和 template 指纹仍须与引擎一致。SSE 默认开启；`engine.stream: false` 只用于不需要客户端 TTFT 的诊断。

## 验收证据

依次保留源码核验、安装环境、启动日志、smoke JSON、标定 manifest/原始观测、开发集选择及冻结文件、测试逐轮结果、质量评分和最终汇总。GPU 冒烟通过仅证明这个模型和配置的协议链路；完整质量、预测泛化、容量收益和并发性能须分别给出实验。记录不通过的配置及失败原因，不以模拟或 fake HTTP 测试替代真实运行。
