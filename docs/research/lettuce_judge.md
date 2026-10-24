# LettuceDetect v2 的独立本地诊断入口

`bench/lettuce_judge.py` 复用作者固定 QA 输入格式、二分类方向和 span 解码思路，直接运行已有 Transformers token-classification 接口，不引入整个 LettuceDetect 框架。它与 Atomic/native 默认评分互不替换，没有五维 scores、Atomic 三/五分类 verdict 或正式质量通过标志。已完成固定模型的完整 29 条 GPU 开发诊断，全部可运行，但范围漏检和正确换算误伤仍明显；详见 [真实验证记录](../archive/lettuce_judge_validation.md)。

固定模型为 `KRLabsOrg/lettucedect-v2-mmbert-base`，revision `8831ce063dff760b01efe4519b4188f5ee2456f2`，Apache-2.0 权重，模型 config 的上限为 8192。固定作者实现为 `fbce1991f26e210cc49ace3913562ab4f2f31a60`、MIT；英文 QA 模板逐字保存在 `bench/assets/lettuce_judge/qa_prompt_en.txt`，完整 MIT 与适配说明在同目录 LICENSE/NOTICE。模型权重不随本项目分发。[模型卡](https://huggingface.co/KRLabsOrg/lettucedect-v2-mmbert-base/blob/8831ce063dff760b01efe4519b4188f5ee2456f2/README.md)、[作者输入格式](https://github.com/KRLabsOrg/LettuceDetect/blob/fbce1991f26e210cc49ace3913562ab4f2f31a60/lettucedetect/detectors/prompt_utils.py)、[英文模板](https://github.com/KRLabsOrg/LettuceDetect/blob/fbce1991f26e210cc49ace3913562ab4f2f31a60/lettucedetect/prompts/qa_prompt_en.txt)。

## 输入与复用边界

第一段严格按作者公开 QA `predict` 路径格式化：当前 query、所有完整可见原文、按原 manifest 次序排列的 `passage 1/2/...`，包括重复正文。第二段是逐字原答案，保留 Evidence ID、Markdown 和代码。两段一起调用 tokenizer pair 编码，包含特殊 token；不是把 model card 的文字概述自行变成三段输入。

模型实际不接收 `actual_history`，因为作者 QA 接口没有该槽位；输出明确记录未提交的历史轮数。当前 query 不被自动改写或补写。历史 assistant 不能被拼成额外来源，retrieved top-k 也不能替换完整 visible evidence。该输入协议与 Atomic 有真实区别，尤其是依赖对话消歧的案例，不能隐瞒。现有 `judge_payload` 白名单和 `evaluation_binding` 被复用；sidecar 标签、黄金答案、方法臂名不进入 tokenizer 输入。

默认最大长度 8192：先以 `truncation=False` 完整计数，超限不调用模型、返回 not_covered；没有截断、分块或作者包装器的 max-over-chunks 聚合。缺正文、空正文、无来源和空答案分别保留原因。模型加载只读操作者提供的本地 snapshot，`local_files_only=True`、`trust_remote_code=False`，不下载或重新 hash 权重；启动时核对 config 的 0=supported、1=hallucinated 和上下文限制。

## 原生预测与偏移

每个答案 token 用两类 softmax 后的 argmax，平局为类 0，遵循作者预测规则，不在 29 条开发集上另调阈值。连续的类 1 答案 token 合成 span，confidence 仅是该 span 内 token 的最大类 1 概率，不能当作整篇事实错误概率或置信保证。[固定作者推理](https://github.com/KRLabsOrg/LettuceDetect/blob/fbce1991f26e210cc49ace3913562ab4f2f31a60/lettucedetect/detectors/transformer.py)。

偏移采用作者批量路径同样使用的 `sequence_ids==1`，排除添加的特殊 token 和空 offset。Fast tokenizer 的 pair offset 已相对于原答案，故直接使用，不再减去第一个答案 token 的起点；这避免前导空格把全部 span 左移。每个 token/span 保存 `start/end` 与原答案切片，不把 token decode 后的规范化字符串冒充原文字段。Unicode、前导空白、序列 ID、特殊 token 和 offset 越界都有合同测试。

原引用标记先逐字评分，暂不因先前 MiniCheck 的引用误伤而静默删除。每个 token 标记是否与原 citation marker 相交，每个 span 保存交集范围，便于真实运行后识别引用相关误伤；没有把“模型认为某个 ID unsupported”转换成引用错误或答案错误。部分非空白字符未被 token 覆盖时记录 partial。模型异常保存失败行，后续行继续，不能当作无 hallucination。

## 使用与 GPU 开发运行

```bash
# 模型无关导出：只检查材料、产生实际输入文本，不加载权重
python -m bench.lettuce_judge --audit AUDIT.jsonl --output NEW_DRY_RUN.jsonl

# 真实本地 snapshot；需要环境已有 torch 与固定 Transformers
python -m bench.lettuce_judge \
  --audit AUDIT.jsonl --output NEW_LEDGER.jsonl \
  --model-path /absolute/path/to/lettucedect-v2-mmbert-base \
  --device cuda:0 --max-length 8192
```

API：`LettuceScorer.from_local_snapshot(path, device="cuda:0", max_length=8192)`；`evaluate_audit(audit, scorer)`；`evaluate_file(audit_path, output_path, scorer)`。测试可注入 tokenizer 与概率回调，不依赖 PyTorch/GPU。每行保存固定完整协议、实际 prompt/answer、可见来源、旧生成 input_hash 绑定、模型 token 概率与原文 span、长度/覆盖/失败信息。结果文件以排他方式创建，避免覆盖已执行实验。

忽略目录下 `runs/run-lettuce-development-v1.py` 为 root 准备完整冻结 29 条 driver，默认 plan，显式 `--execute` 才加载本地模型。默认从远端 `runs/atomic-development-v3` 逐字复制 audit 和 labels 到新的 `runs/lettuce-development-v1`，labels 只保存，不参与模型。记录实际 Python/torch/transformers/tokenizers、GPU 信息、源码/模板和下载来源记录；它不会停止服务、下载权重或自动选择样本。

```bash
python runs/run-lettuce-development-v1.py --execute
```

首轮必须完整保留 29 条，包括超限、失败、拒答、引用相关 span 与两条代码对照；静态代码语法诊断独立保存。`scored` 只表示分类器完成，不表示断言正确；无 unsupported span 也不是质量通过。与 Selene 的比较按共同可解释的 supported/unsupported 信号及类别错误分析进行，不从二分类结果补造缺失的拒答可答性、引用正确性或五维质量。
