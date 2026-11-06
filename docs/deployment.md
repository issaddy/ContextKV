# 当前 GPU 部署

本轮源码先提交远端，容器任务与 SSH 已关闭，等待用户清理和更新容器。最近使用入口为 `ssh relay`、`ssh songjiayun@infra-wlf3-ge103-1.idchb2az3.hb2.kwaidc.com`、`sudo nerdctl exec -it test-0518 bash`；项目目录 `/home/relay/songjiayun/ContextKV`，模型 `/data/models/qwen3-32b`。

复用容器当前 Python、PyTorch 和 CUDA 内核。`bootstrap_sglang.sh` 仅准备官方 SGLang 0.5.18 源码与当前项目补丁，不创建虚拟环境、不重装依赖、不要求升级驱动。源码提交和实际包版本进入 runtime manifest；课题固定 Qwen 权重、tokenizer 和组间推理条件。

```bash
export CONTEXTKV_MODEL_SCOPE_PATH=/absolute/path/qwen3-scope.json
bash scripts/bootstrap_sglang.sh
source runs/engine-runtime.env
```

默认启用 Unified Radix 的 Python TreeCore。主比较的 BF16 权重/KV、窗口、GPU/Host 配额、page size、采样和推理参数由运行配置共同冻结。新源码补丁尚待实际 GPU 端到端验证，不能套用旧机器的性能记录。

`CONTEXTKV_MODEL_SCOPE_PATH` 与 `CONTEXTKV_FINGERPRINT_JSON` 取当前本地模型记录和实际 tokenizer/template。启动入口为 `scripts/launch_sglang.sh`，原生参数：

```bash
--host 127.0.0.1 --port 30000 --max-total-tokens 32768 \
--mem-fraction-static 0.65 --max-running-requests 8 \
--attention-backend fa3 --chunked-prefill-size 8192 --schedule-policy fcfs \
--speculative-algorithm NGRAM --speculative-ngram-max-bfs-breadth 1 \
--speculative-num-draft-tokens 12 --random-seed 42
```

本机 HTTP 调用同时设置 `NO_PROXY` 与 `no_proxy` 为 `127.0.0.1,localhost,::1`。任务只使用本次明确选择的设备，重新采集当前硬件上的原生成本和配对结果；旧答案与成本记录已丢弃。下载与 Git 操作临时使用用户提供的代理，不写入配置或仓库。容器更新前不再启动训练、推理与 GPU 测量；后续任务完成时关闭本任务启动的服务、observer 和 SSH 会话。
