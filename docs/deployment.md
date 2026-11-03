# 当前 GPU 部署

容器：`public-ge108-wlf3-hangu-7` 的 `sjy-0518`。源码：`/home/relay/songjiayun/ContextKV/runs/study-20261103-budget-loop/source`。使用容器自带 Python，安装项目所需 Python 包和固定 SGLang Python 代码。Qwen3-32B 路径为 `/data/models/qwen3-32b`，在 `configs/model_scopes/qwen3-32b.local.json` 中绑定。

固定 SGLang 提交为 `3f36fc08331f64cd63c4fbbdf39cc402f38bb38b`，使用匹配的 PyTorch 2.14.1 / CUDA 13 运行包与 `sglang-kernel` 0.4.10。R0 服务使用 GPU 2、端口 30000、BF16 权重/KV、32768 KV token、16 GiB Host、page64、NGRAM breadth1/draft12、fa3、JSON 空白上限16。实际启动参数和环境记录在本轮 `servers/hangu7-r0-g2/launch.json`。

```bash
python -m pip install --no-build-isolation -e '.[tokenizer,retrieval,query-adapter]'
python -m pip install --no-deps --no-build-isolation -e vendor/sglang/python
python -m pip install 'torch==2.14.1' 'sglang-kernel==0.4.10' \
  'flashinfer-python[cu13]==0.7.2' 'xgrammar==0.2.7' 'nvidia-cutlass-dsl[cu13]==4.8.0'
```

`CONTEXTKV_MODEL_SCOPE_PATH` 与 `CONTEXTKV_FINGERPRINT_JSON` 取当前本地模型记录和实际 tokenizer/template。启动入口为 `scripts/launch_sglang.sh`，原生参数：

```bash
--host 127.0.0.1 --port 30000 --max-total-tokens 32768 \
--mem-fraction-static 0.65 --max-running-requests 8 \
--attention-backend fa3 --chunked-prefill-size 8192 --schedule-policy fcfs \
--speculative-algorithm NGRAM --speculative-ngram-max-bfs-breadth 1 \
--speculative-num-draft-tokens 12 --random-seed 42
```

服务健康后接续当前采集配置。H200 上重新采集原生成本和配对结果；旧机器结果保留在评价参照中。实验结束关闭本任务启动的服务、observer 和 SSH 会话。
