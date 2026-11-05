# 当前 GPU 部署

当前入口依次为 `ssh relay`、`ssh songjiayun@public-ge108-wlf3-hangu-7.idchb2az3.hb2.kwaidc.com`、`sudo nerdctl exec -it test-0518 bash`。项目目录为 `/home/relay/songjiayun/ContextKV`，Qwen3-32B 位于 `/media/ssd1/data/models/qwen3-32b`。保留容器自带环境，固定引擎所需版本装入 `.venv-sglang`；GPU 设备、端口及截止时间由本次 launch manifest 声明。

固定 SGLang 提交为 `3f36fc08331f64cd63c4fbbdf39cc402f38bb38b`。复用已安装的匹配运行依赖和固定 vendor；模型权重与 tokenizer 不变。主比较沿用 BF16 权重/KV、32768 KV token、16 GiB Host、page64、NGRAM breadth1/draft12、fa3、JSON 空白上限16。实际环境、启动参数与原生缓存配置保存为本次 runtime manifest。

`CONTEXTKV_MODEL_SCOPE_PATH` 与 `CONTEXTKV_FINGERPRINT_JSON` 取当前本地模型记录和实际 tokenizer/template。启动入口为 `scripts/launch_sglang.sh`，原生参数：

```bash
--host 127.0.0.1 --port 30000 --max-total-tokens 32768 \
--mem-fraction-static 0.65 --max-running-requests 8 \
--attention-backend fa3 --chunked-prefill-size 8192 --schedule-policy fcfs \
--speculative-algorithm NGRAM --speculative-ngram-max-bfs-breadth 1 \
--speculative-num-draft-tokens 12 --random-seed 42
```

本机 HTTP 调用同时设置 `NO_PROXY` 与 `no_proxy` 为 `127.0.0.1,localhost,::1`。任务只使用本次明确选择的设备，重新采集当前硬件上的原生成本和配对结果；旧答案与成本记录已丢弃。下载与 Git 操作临时使用用户提供的代理，不写入配置或仓库。按用户本次指定的停止时间结束训练、推理与 GPU 测量，关闭本任务启动的服务、observer 和 SSH 会话。
