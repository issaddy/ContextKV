# 矩阵评测的自有服务器入口

`bench.b6_joint_sweep --matrix` 按声明的配置执行 Qwen3 对照，支持 `dev`、`test` 和 `dev_ablation`。默认打印计划，加 `--execute` 执行。固定配置可直接运行 test；从开发集选择过候选的配置使用对应 `--frozen` 记录。

```bash
RAYON_NUM_THREADS=4 .venv/bin/python -m bench.b6_joint_sweep \
  --matrix configs/benchmark_mtrag.yaml --stage test \
  --output runs/mtrag-system-eval
```

输出目录保存各 job 的配置、原始输入与回答、服务器日志、实际运行环境、评分关联信息及结束状态。资源路径相对当前工作目录。每个 job 启动一个自有生成进程，方法之间串行执行；结束或失败时清理该进程组。

## 配置与运行条件

模型、endpoint、语料和 checkout 来自矩阵及共享配置。生成使用本地 `model.scope_path`；controller 使用 patched checkout，native 使用 stock checkout。当前执行器采用 BF16、32768 上下文、page64、GPU0、FA3、chunk8192、FCFS、max running8，Host admission 为 off，HiCache 为 kernel/layer_first/TMA0。

每个 profile 声明 `kind`、`checkout`、`write_policy`、`hicache`、`session_radix`、`gpu_tokens`、`hicache_gb` 和 `mem_fraction_static`。同资源对照共用 GPU 槽数和显存比例，启用 HiCache 的方法共用 Host 配置；B0 为关闭 Host/session 的基线。实际容量来自分配日志与 metrics，启动时记录 GPU、driver、CUDA、PyTorch 和原生 server info。

共享配置决定 prompt、context policy、输出预算、模型和采样。`seed` 控制负载到达顺序，SGLang 采样种子当前固定为42；重复运行另有 repetition ID。`ablation_of` 继承父方法的冻结参数和 server profile，使单项消融保持其余条件一致。

Qwen3-8B/32B 共用 `create_model_template` 和 `LocalModelScope`。scope 记录本地配置、分片清单、tokenizer 与模板身份；实际 HF loading revision 为 None。质量模板中的 revision 存放本地 scope 身份，权重内容验证状态另行记录。32B 的 KV 字节换算使用 config 的 `head_dim=128`。

## 成本与任务执行

- `cost` 和 shadow cost 使用与模型、缓存和采样相符的硬件成本 artifact；加载采用该 job 的实际预算及成本分项设置。
- `warm_idle` 使用原生 quote；固定策略直接执行其配置。
- 模型已测范围以外的请求记录实际固定策略选择，成本覆盖情况进入报告。

任务来自 split/domain/limit_sessions：dev 和 dev_ablation 取开发分区，test 取测试分区。失败、截断和未完成任务保留在结果中。有失败时 CLI 返回非零状态。当前执行器支持既有到达、think-time 和超时配置，截窗负载的组合支持范围以参数检查为准。逐轮deadline从该轮到达开始，P3包含解释/检索与生成；取消会等待已启动的同步检索安全结束，因此清理可晚于deadline。已提交结果的发布不再触发生成超时或重复记账。

P3 默认共用 retriever_config；`comparison_scope: system` 允许方法声明各自的检索与意图机制。两类运行均计入检索排队与执行成本。共享32B解释服务按[部署说明](../deployment.md#多轮检索服务)启动，其进程与资源单独记录。

## 评分与结果复用

开发流程启用 `require_quality_evidence` 时生成 `quality-evidence.template.json` 与分块模板。模板列出实际 job、候选、重复和失败，关联同块参照；实际评分随后填写。test 与 dev_ablation 保存完整结果，使用匹配的参考运行进行离线比较。

输入、实现、配置和评分来源随运行记录保存。固定测试复用已选择候选及声明的重复计划。当前运行使用实际模型身份，历史结果按原运行版本读取。

质量比较见[质量评价](quality_acceptance.md)。报告默认 `response_mode: buffered`，声明 `slo.complete_seconds` 后计算非截断时延达标吞吐；流式交付使用 `streaming` 及 TTFT/TPOT。评价口径进入原始 run 的 `experiment_protocol`，离线报告直接复用。主指标见[项目范围](../project_scope.md#研究目标与指标)。

## 源码失效与归档位置

源码摘要只保留溯源，不再分析 AST、维护依赖清单或强制源码全等。执行语义改变时更新方法声明中的 `revision`；它与数据、采样、预算等一起由已有配置摘要绑定。指标计算改变时更新 `bench.report.METRIC_SCHEMA_VERSION`，`freeze` 从内容摘要匹配的原始 run 重算后重新选择；展示文案改变不升级指标版本，也不使答案失效。协议版本由实现者明确维护，工具不猜测任意代码修改的语义。

归档可以在矩阵和质量配置中声明 `path_map`，例如 `{"/old/results": "/archive/results"}`。它映射已保存评测声明中的文件或目录前缀，不重写原始回答、审计或冻结协议；读取后仍须匹配完成时保存的 SHA256。缺少摘要或内容变化不会因路径映射而放行。运行模型、索引和新任务输出位置仍由执行配置指定。
