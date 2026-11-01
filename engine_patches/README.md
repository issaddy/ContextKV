# SGLang集成

固定SGLang版本为`3f36fc08331f64cd63c4fbbdf39cc402f38bb38b`。本目录维护原生桥接实现、模型声明和集成补丁；测试与测量工具位于`profiling/`，运行记录位于`runs/`。

## 运行职责

- `contextkv_runtime.py`：提供同批缓存报价，关联业务epoch与原生会话generation，发布完成回执，处理引用退役和资源观测。
- `contextkv_model_scope.py`：加载声明的模型revision、配置和分词身份，供控制器与引擎共用。模型配置直接比较，权重加载交给原生引擎。
- `0001-contextkv-protocol.patch`：把上述桥接接入固定版本的scheduler、Radix/HiCache和HTTP通道；`source_manifest.json`声明上游版本及涉及文件。

报价区分连续GPU前缀、可恢复前缀、Host加载量与在途传输。原生匹配边界和页对齐决定可复用token范围；报价读取状态，实际匹配、准入、加载与淘汰由SGLang执行。

业务epoch通过prepare关联原生会话引用。生成回执记录实际token、分层命中、prefill工作量和完成状态；应用提交后再retire旧epoch。退役撤销业务软引用，物理回收按原生引用计数与淘汰规则执行。共享前缀仍由其他活跃会话使用时保留。

可选Host备份预算在原生备份提交处按请求累计成功准入页，用于业务工作集管理。常规部署使用原生策略；该选项的收益需独立负载评价。

## 准备与运行

完整引擎环境：

```bash
python scripts/prepare_sglang.py --checkout vendor/sglang
SGLANG_BUILD_RUST_EXTS=none python -m pip install -e vendor/sglang/python
```

固定版本的`python/pyproject.toml`提供依赖约束。启动命令、Qwen3-32B配置和设备分配统一见[部署](../docs/deployment.md)。当前运行环境使用单GPU、BF16和Python TreeCore；主生成执行完整KV生命周期，辅助任务解释实例使用原生NGRAM。

集成也支持原生NGRAM的FULL token缓存，使解释与生成可共享一个32B实例及Radix/HiCache。业务epoch、完成回执和引用退役沿用同一流程；NGRAM执行参数进入现有运行配置标识，供阶段成本标定使用。隔离验证已完成追加命中及资料版本更新后的新epoch提交和旧引用退役；完整会话质量与资源结果见评价记录。

仅检查补丁应用与语法：

```bash
python scripts/prepare_sglang.py --source-only \
  --checkout vendor/sglang-source-check --report runs/engine-verification.json
```

实际服务检查入口：

```bash
python -m profiling.smoke --config configs/qwen3_32b_scope.example.yaml
```

`profiling.lifecycle_probe`和`profiling.observation_probe`用于隔离服务的生命周期与观测验证。`profiling.sglang_probe`采集B所需的阶段数据；这些工具的准备操作会改变缓存状态，按各命令的`--help`运行。应用层调用与提交关系见[架构](../docs/architecture.md)，测量方法见[成本标定](../docs/guides/stage_cost_model.md)。
