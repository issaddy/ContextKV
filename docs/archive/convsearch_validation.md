# ConvSearch-R1 3B 开发重写验证

> 归档记录：保留当时的设计、结果与复用依据；其中“下一步”和执行计划不代表当前任务。当前工作仅以[实现清单](../implementation_plan.md)为准。

本轮完整运行 5 个 Cloud 开发会话、41 个问题，全部返回 `stop`，且能解析为一组完整的 `<think>` / `<rewrite>`。这验证了固定 checkpoint 的调用与作者协议接线；**没有验证检索收益、事实正确性或答案质量准入**。重写中的答案式扩写与历史助手错误传播仍需检索和逐源检查。

## 固定输入与实际运行

原始结果位于 `runs/remote-h200-convsearch/runs/convsearch-owned-development-v1/cqr`；同一导入目录保留冻结输入、stock 服务 runtime、部署证明、模型下载记录与 supervisor。归档 SHA256 为 `3e87eaf2eaeb3ac2836211c44e803778dd9f0f5d1d1f67f70083b7d80cfcefe5`。独立检查脚本与逐条结果是 `runs/convsearch-development-v1-prep/analyze_actual.py`、`actual-validation.json`。

检查覆盖所有 41 条的 task、当前问题、完整实际历史、作者消息渲染、原始输出重解析、结束原因与 token 预算。输入逐字等于原 `source-topic-preparation/frozen-v1` 的 OnlineTurn，以及本次 `frozen-v2` 中的消息；没有使用当前答案、参考答案或 qrels。41 条本地预先计数均等于服务报告的 prompt tokens。已有历史是旧 off 运行实际生成的助手文本，可能含错，不能视为事实标签。

| 项目 | 实际配置或结果 |
|---|---|
| Checkpoint | `BeastyZ/Qwen2.5-3B-ConvSearch-R1-QReCC`，revision `2756459f017585c71a6ca54316cb9acb43bdfee6` |
| 作者代码 | `e7854c89a79c904f331b306d36edde6e9ed12484`，原 prompt、完整 prior QA 序列与 think/rewrite 协议 |
| 项目部署 | `20fef314241c60530775a41f07fdba15c40752fd`；部署记录验证 293 个 tracked files |
| 引擎 | stock SGLang `3f36fc08331f64cd63c4fbbdf39cc402f38bb38b`；独占 GPU0、BF16、TP1 |
| 采样 | temperature 0.7、top_p 1、max_new_tokens 4096、无自定义 stop、chat_template_kwargs `{}`；服务 seed 42 |
| 上下文 | 32768；完整 prior actual QA，5 个首轮也执行模型 |
| 返回状态 | 41/41 complete、41/41 stop；无缺失 usage，无长度截断 |
| Token 用量 | prompt 76822，completion 16556；每条输入 157–5186，输出 150–719 |
| 重写长度 | 462–2976 字符；9/41 含代码围栏 |
| 客户端 completion 时间 | 41 次合计 50.44 秒、平均 1.230 秒；不含服务启动，非并发吞吐指标 |

服务公开参数与下载/部署记录一致，客户端返回模型别名 `convsearch-r1-qrecc-3b`；这些证据不是通过同 tokenizer 推断模型相同。checkpoint 的 FP32 保存与本次 BF16 推理分别记录。全部请求完成后，owned 服务在清理超时后被 SIGKILL（最终返回码 -9）；不能称为优雅退出，也不将其记为生成失败。

作者使用 vLLM 离线批处理，本轮使用 stock SGLang OpenAI 接口；相同 prompt 和采样参数不意味着相同随机抽样。作者源没有固定 seed，本轮显式记录服务 seed。协议及许可出处见 [固定作者 infer.py](https://github.com/BeastyZ/ConvSearch-R1/blob/e7854c89a79c904f331b306d36edde6e9ed12484/src/infer/infer.py) 和仓库内 `adapters/assets/convsearch_r1/protocol.json`。

## 具体重写边界

以下是对当前问题、同条实际历史及已冻结可见材料的只读语义检查，不是人工评分，也不使用当前外部 IBM API 知识替代固定语料。

| Task | 可以确认的变化与边界 |
|---|---|
| `d5b1e735a040853ed361a3dfde1b8ef0<::>4` | 对“Is there a limit on the file size?”补出 IBM Cloudant 与图片/PDF 附件，找回被省略对象。但额外断言文档未说明文件大小限制，并扩写性能、复制和替代存储建议；只凭给定历史不能证明“文档没有限制”。 |
| 同会话 `t5` | 用户纠正“Not documents…attachments”，重写却主要变成如何存储图片/PDF，并附 JSON 与大小数字。Cloud Pak for Data 50MB、Premium/Enterprise 10MB 来自旧助手 t4；旧可见源 `ibmcld_06968-16615-18789` 实为 Discovery 上传/抓取计划限制，且上传表的 Premium 为 50MB。重写继承了旧助手的产品归属与表格误读，不能当新的有源 Cloudant 条件。 |
| 同会话 `t7` | 将历史“individual documents 1MB、attachments 10MB”的对象区别，改写成 Lite 的文档/附件 1MB 与 Standard 附件 10MB 的计划区别。这一映射不由实际历史支持。 |
| `c01c8cf11437e6bb3bc93efac26528c2<::>5`，Carthage | 当前用户明确说 Carthage；重写继续加入前序助手的 IBM Cloud Functions、OpenWhisk Swift SDK `>0.3.0`，并新增历史中没有的 `gem install carthage`。这些不是用户新指定的版本要求。 |
| `c01c8cf11437e6bb3bc93efac26528c2<::>6` | 先前 `carthage update` 用于下载/构建 SDK 依赖；重写将它写成安装最新版 Carthage 工具本身的方法，出现具体操作对象转换。没有编造最新版数字，不应误报为版本号幻觉。 |
| `1c041ce47a81941c26899fdf08bde961<::>6` | “what is Data lakes”仍以 data lake 为主体，没有强加旧 Knowledge Studio 对象；但扩写为解释性答案，并添加 data warehouse 比较和 single source of truth 等历史没有的断言。这是新增目标/答案式扩写，不能归为旧对象污染。 |
| `c6c3b02ca32795af64c903dd76700517<::>6` | 保留创建加密镜像及无法扫描的前提，没有强带旧版本 3/4；随后添加其他工具、云存储替代建议，超出当前创建问题与给定历史支持。 |

这些例子说明完整历史有时能恢复实体，也会把历史助手的数值、操作步骤和关系错误带进检索查询。长扩写是否增加召回或引入干扰，须由后续固定检索资源的排名实验回答，不能仅按字数或是否有代码判胜负。

## 后续固定查询检索对照

准备八方法、各 41 条的同语料排名对照：current、user_topic 单路、user_topic 双路 RRF、v1/v2 的 source/unverified 两类 selector、ConvSearch 单 rewrite。只读取已保存输出；失败或 NA 保留原状态并回退 current，所有尝试完成并落盘之后才读取开发 qrels。复用实际相同 route/rerank 查询的研究进程内结果时，记录来源和逻辑/实际执行量，不把复用行的时间当在线速度收益。

ConvSearch 是 3B、完整 QA、temperature 0.7、4096 输出预算且首轮调用；selector 是 7B、最近三轮 QA、temperature 0、256 输出预算且首轮跳过。因此可以对齐检索语料、BGE 模型与总 rerank 50，不能把两者的生成开销称为“仅方法不同”的公平时延对照。重写协议中的 `model_downloaded: false`、`runtime_or_quality_validated: false` 是冻结适配器准备阶段的静态声明；实际运行事实以本轮 runtime、manifest 完成状态和逐行结果为准，质量未验证的限制继续成立。
