# GPU 部署

使用容器默认 `python3` 与已安装的 SGLang 0.5.23。准备脚本在当前 SGLang 源码内应用项目补丁，依赖沿用容器环境。生成模型目录为 `/data/models/qwen3-32b`，模型声明位于 `configs/model_scopes/qwen3-32b.local.json`；共享辅助模型目录在 `configs/study.yaml` 配置。

```bash
CONTEXTKV_MODEL_SCOPE_PATH=configs/model_scopes/qwen3-32b.local.json bash scripts/bootstrap_sglang.sh
source runs/engine-runtime.env
CUDA_VISIBLE_DEVICES=4 bash scripts/launch_sglang.sh --port 30000
```

启动脚本使用 TP1、BF16、32k 窗口、64-token 页和 HiCache。`CONTEXTKV_GPU_TOKENS`、`CONTEXTKV_PAGE_SIZE` 和 `CONTEXTKV_HICACHE_GB` 分别设置 GPU token 容量、页大小及 Host 缓存大小；控制器从引擎读取实际容量。设备使用范围为 4–7。

辅助进程设置 `CUDA_VISIBLE_DEVICES=4,5,6,7`，查询与证据编码的逻辑设备 `cuda:2/3` 对应物理 GPU 6、7；物理 GPU 5 用于并行标注或训练。按实际就绪阶段并行，训练完成后加载对应的共享资产。

H200 固定题单的辅助编码共用物理 GPU 5。复现时设置 `CUDA_VISIBLE_DEVICES=4,5`，并在运行配置中将查询和证据设备都设为 `cuda:1`。以下命令使用仓库配置的四卡映射。

```bash
CUDA_VISIBLE_DEVICES=4,5,6,7 python3 -m controller --config configs/study.yaml serve --host 0.0.0.0 --port 8000
```

SGLang 的 `/generate` 同次接收输入、原文块和后续需求。`contextkv_compute.mode` 支持 `exact`、`infoflow`、`auto`。`/context/metrics` 返回必要缓存计数和累计 KV 占用；`/context/profile/prepare` 用于队列起点清缓存。

GPU 部署使用小请求校准独立块构建和完整 prefill 的费用；auto 根据包含解码的完整费用选择精确执行或 InfoFlow。

引擎补丁与运行实现见 [SGLang 集成](../engine_patches/README.md)。
