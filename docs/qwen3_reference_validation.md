# Qwen3 本地模型：完整开发集参考与同输入对照

2026-10-24。用户提供的 `/media/ssd1/qwen3-8b` 与 `/media/ssd1/qwen3-32b` 已在 H200 上完成 **159＋159次真实生成**。独立审计逐轮重分词和重渲染通过，32B实际收到的消息、证据、渲染文本及输入IDs与8B对应轮一致。**这确认了可用的模型能力比较基础，尚未建立回答质量优劣、缓存策略优势或正式非劣结论。**

## 运行合同

完整dev为22会话/159轮：ClapNQ 6/47、FiQA 5/32、Govt 6/39、Cloud 5/41。8B按P2使用自身已提交的完整QA，每轮REBUILD；所有159轮成功后冻结实际输入，32B逐轮重放这些输入。32B的新答案不写回后续历史，因此它不是32B自有历史闭环。官方top5是固定输入协议，不是oracle，不保证来源充分。

两模型采用Q0、非思考模板、temperature0.7/top_p0.8/top_k20/min_p0、输出cap1024、正常EOS、c1、server seed42和固定会话顺序。不声称逐请求确定性；length答案保留并进入分母，不重试或提高预算。模型分别串行运行，未使用旧cost/length标定、研究planner或host/session缓存；原生Radix前缀缓存仍可使用。

执行源码为`a6b5711d1c82ac0fae8f8c26d8b3d6efd2ba82c1`，远端325个tracked Git blobs在启动前核验。固定stock SGLang为`3f36fc08331f64cd63c4fbbdf39cc402f38bb38b`，未应用研究补丁；BF16权重/KV、TP/DP/PP1、FA3、page64、context32768、GPU pool32768、chunked prefill8192、无HiCache/session radix。独立运行器保留每次真实HTTP、终态、完整input/output IDs和前序QA。此批身份是`generator_capability_reference`，不是正式B0/B6。

本地权重来源只确认实际config、分片库存/字节及tokenizer/template，未确认HF revision或权重内容摘要，保留`local_snapshot_metadata`与`weight_content_verified=false`。不能把本地身份标签写成已验证上游commit。后续主工程[显式模型接入](model_scope.md)是独立的`a87638a`实现，不追改本批源码或借本批stock结果宣称研究补丁已通过GPU验证。

## 完整性与实际观测

[完整公共归档](../runs/remote-h200-qwen3-reference/public-export-manifest.json)包含44文件（含导出清单），archive SHA256为`5e871f02fcaffdeeb0240b7139aec54e1d206752479805a7cd420e980ddfd180`、11,769,080字节；43个payload文件逐一核验。归档/传输在生成结束后进行，未混入私有日志、环境变量或权重。

[独立审计](../runs/qwen3-reference-review/h200-v1/analysis.json)覆盖：固定dev全分母、原始当前问题/top5白名单、8B实际QA因果、冻结文件、32B不反馈自身答案、完整正文span、native实际请求/响应与token IDs、采样及终止原因、实际模型metadata和声明源码。全部318轮机械完整性通过，无遗漏/重试伪补。使用导出的真实backend以及各自保存的Jinja模板，独立重建318份输入；没有以相同tokenizer目录名代替验证。

| 域 | 轮次 | 8B length | 32B length | 8B平均输出tokens | 32B平均输出tokens |
|---|---:|---:|---:|---:|---:|
| ClapNQ | 47 | 1 | 0 | 456.45 | 397.19 |
| FiQA | 32 | 3 | 3 | 659.19 | 600.03 |
| Govt | 39 | 4 | 3 | 553.13 | 453.87 |
| Cloud | 41 | 9 | 7 | 684.61 | 640.54 |
| 合计 | **159** | **17** | **13** | **579.80** | **514.67** |

两组各159成功；8B为142 stop/17 length，32B为146 stop/13 length。平均输入均为4654.48 tokens。截断减少和输出变短均不是质量改善的替代指标。

| 客户端观测口径 | 8B | 32B |
|---|---:|---:|
| 按轮平均HTTP完成时间 | 3.182s | 10.016s |
| 等会话平均HTTP完成时间 | 3.135s | 9.755s |
| 按轮平均请求TTFT | 99.80ms | 396.97ms |
| 按轮平均TPOT | 5.31ms | 18.68ms |

这是一次固定顺序、不同模型/输出的开发观测，不是随机交叉系统实验；没有将原生cache状态/解码长度固定为单变量，也未建立吞吐或尾时延优势。请求TTFT以实际首新增token chunk为准，TPOT是客户端观测，不是CUDA kernel时间。两组虽然同为32768个KV slots，BF16模型几何对应的理论KV大小分别约4.5GiB与8GiB；这不是实测峰值显存，不能称物理内存预算相等。完整质量与成本需同时比较。

原研究7B服务由supervisor恢复为PID71819，独立health HTTP200及`/context/engine/info`模型身份核验通过。首次辅助info探针误用了`/context/info`，该失败与后续正确探针均保留在[恢复记录](../runs/remote-h200-qwen3-reference/runs/qwen3-reference-restore-health.json)，没有因此重启或重跑生成。

## 接续分析

[E_ID成员检查](../runs/qwen3-reference-review/h200-v1/citation-membership.json)只作机械诊断：8B/32B分别92/91轮含显式E_ID，各有4轮引用了当前可见正文清单之外的E_ID。无E_ID不等于无URL来源或错误回答，合法E_ID也不证明支持论断；不能将这些数量转换成引用质量分数。

下一步依据完整原文检查对象、条件、版本、回答性、代码与实际引用，区分模型能力不足、来源不足和支持关系错误。先做冻结输入上的机制定位，再验证自有历史闭环与共同P3。[来源生成方案](source_bound_generation_design.md)和[支持单元原型](support_unit_design.md)提供待检验的分解与强对照；当前没有因完成生成就默认复杂机制有效，也没有选择最终B6或使用heldout。
