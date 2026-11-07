# 当前 GPU 部署

进入实验容器：

```bash
ssh relay
ssh songjiayun@infra-wlf3-ge103-1.idchb2az3.hb2.kwaidc.com
sudo nerdctl exec -it test-0523 bash
cd /home/relay/songjiayun/ContextKV
source runs/study-20261106-resume/default-python/runtime.env
```

使用默认 `python3`，模型目录为 `/data/models/qwen3-32b`。

`bootstrap_sglang.sh` 准备官方 SGLang 0.5.18 源码与项目补丁，使用容器现有 Python、PyTorch 和 CUDA 内核。源码提交和实际包版本进入 runtime manifest；课题固定 Qwen 权重、tokenizer 和组间推理条件。

```bash
export CONTEXTKV_MODEL_SCOPE_PATH=/absolute/path/qwen3-scope.json
bash scripts/bootstrap_sglang.sh
source runs/engine-runtime.env
```

启用 Unified Radix 的 Python TreeCore。主比较的 BF16 权重/KV、窗口、GPU/Host 配额、page size、采样和推理参数由运行配置共同冻结。

`CONTEXTKV_MODEL_SCOPE_PATH` 与 `CONTEXTKV_FINGERPRINT_JSON` 取当前本地模型记录和实际 tokenizer/template。启动入口为 `scripts/launch_sglang.sh`，原生参数：

```bash
--host 127.0.0.1 --port 30000 --max-total-tokens 32768 \
--mem-fraction-static 0.65 --max-running-requests 8 \
--attention-backend fa3 --chunked-prefill-size 8192 --schedule-policy fcfs \
--speculative-algorithm NGRAM --speculative-ngram-max-bfs-breadth 1 \
--speculative-num-draft-tokens 12 --random-seed 42
```

本机 HTTP 调用设置 `NO_PROXY` 与 `no_proxy` 为 `127.0.0.1,localhost,::1`。GPU 设备和服务端口由运行配置指定。下载与 Git 使用运行环境中的代理。任务结束后关闭对应服务、observer 和 SSH 会话。
