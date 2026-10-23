# 本地独立开源 judge 的实施方案

本轮建议先用 **Selene Mini 8B** 做本地元评测，以 **Prometheus 2 7B** 为原协议对照，再为通过实际检查的 evaluator 接入已保存的 Cloud dev 回答。两者分别来自 Llama、Mistral 家族，与当前 Qwen2.5-7B 生成器不同；无需付费 API 或等用户填写人工分数即可推进。跨家族只能减少一种共同偏差，不能证明评分可靠，也不能把模型评分称为人审。

截至 2026-10-24，已完成官方来源审查、`bench/native_judge.py` 原生薄适配器、`bench/grouse_meta.py` 固定元评测与合同测试；Selene/Prometheus 完整权重均已下载并在 stock SGLang/H200/GPU0 上运行。每模型完成 10 次原生诊断评分及完整 GroUSE train16，真实范围漏判和协议失败见[首轮验证](open_judge_validation.md)，不能据此宣布质量可靠。小文件与访问记录在 ignored `runs/open-judge-research/`，合成请求准备文件在 `runs/open-judge-preparation/`。本方案独立于 B 阶段计时实验。

## 现有链路与最小缺口

仓库没有 `bench/local_judge.py`。原 `bench/judge.py` 保留一次五维 JSON 的接口；新 `bench/native_judge.py` 专门支持 Selene/Prometheus 原生 rubric 格式。它复用既有完整证据白名单、query/input_hash 绑定与规则审核，未改主控制器或生成模板。

`bench/quality_acceptance.py` 已核对真实 run、audit、答案、提示身份和划分，并区分自评、开发集准入、heldout 模型 evaluator 非劣与人工质量结论。保留 `scores` 五维完整键、NA 和失败，新增原生分数/推理文本作为诊断。本轮接线将同 base family 的不同 checkpoint 标记为 `related_model_family_declared`，保留其开发诊断但不称跨家族独立；精确同权重仍是自评。首选跨家族模型，不能靠改路径或 served model 名“变成独立”。详见 [现有质量验收](quality_acceptance.md)。

## 已核实的候选与版本

| 模型 | 固定模型 revision | 可访问与许可记录 | 本次用途 |
|---|---|---|---|
| `AtlaAI/Selene-1-Mini-Llama-3.1-8B` | `427792f1c3e2073cb7da216924fd884b1ba496e0` | API/config/tokenizer 200、gated=false，首个 safetensors 分片未鉴权 HEAD 200；模型卡与作者代码声明 Apache-2.0，保留 Llama 3.1 基座身份 | 首选：官方支持绝对分数、二分类、配对、有/无参考答案 |
| `prometheus-eval/prometheus-7b-v2.0` | `66ffb1fc20beebfb60a3964a957d9011723116c5` | 同上访问成功；模型卡与代码 Apache-2.0；Mistral-7B-Instruct-v0.2 基座 | 对照：rubric 1–5 与 A/B 专用判分 |
| `Unbabel/M-Prometheus-7B` | `030fb74806e4228c466a98706a297d43b31ce5df` | 同上访问成功；模型卡 license=`other`，此次未发现单独许可正文 | 较新多语言备选；Qwen2.5-7B 底座，不作为主要跨家族证据 |
| `Skywork/Skywork-Critic-Llama-3.1-8B` | `825f34599593c0145be91644be233d5c634b2380` | API/config 200；官方仓库为 33 个 `.bin` 权重分片；卡片标 Skywork Community License | 主要面向配对偏好，暂不优先用于五维绝对评分 |

访问成功表示本次元数据/一个权重分片的可访问性，不等于已下载完整模型或实测可加载。固定来源：[Selene 模型](https://huggingface.co/AtlaAI/Selene-1-Mini-Llama-3.1-8B/tree/427792f1c3e2073cb7da216924fd884b1ba496e0)、[Prometheus 2 模型](https://huggingface.co/prometheus-eval/prometheus-7b-v2.0/tree/66ffb1fc20beebfb60a3964a957d9011723116c5)、[M-Prometheus 模型](https://huggingface.co/Unbabel/M-Prometheus-7B/tree/030fb74806e4228c466a98706a297d43b31ce5df)、[Skywork 模型](https://huggingface.co/Skywork/Skywork-Critic-Llama-3.1-8B/tree/825f34599593c0145be91644be233d5c634b2380)。许可栏忠实记录发布者声明，不因基座或仓库代码许可自动补全权重授权条件。

Selene 的作者研究评估通用 rubric 判分；Prometheus 2 研究同时支持直接与相对评分；M-Prometheus 扩展多语言 judge。论文上的总体相关性或通用任务成绩不直接证明本项目的表格条件、代码、历史错位和必要拒答可靠。[Selene 论文](https://arxiv.org/abs/2501.17195)、[Prometheus 2 论文](https://arxiv.org/abs/2405.01535)、[M-Prometheus 论文](https://arxiv.org/html/2504.04953v2)。

GroUSE 官方 README 链接的 `illuin/llama-3-grouse` 本次未鉴权 API 返回 401；不能据此区分私有、受限或不可用，也不能把它列成已就绪依赖。[官方链接所在仓库](https://github.com/illuin-tech/grouse/tree/e4c57ba9eb14548c7642daca3ed4ff24d760c7fa)。

## 显存与已有推理接口

Prometheus 2 为 7.242B BF16 参数，官方权重文件约 14.483 GB；Selene 为 8.030B BF16 参数，约 16.061 GB。权重本身约 13.49 / 14.96 GiB。两者配置均为 32 层、8 个 KV heads、head dimension 128；TP=1、BF16 KV 时，每 token KV 约 128 KiB，32,768 slots 约 4 GiB。**24–32 GiB 是单并发 32K 上下文的初步资源预算，非实测最低显存或峰值**，还需包含激活、图和 allocator。Selene 配置支持 128K，但不能把 32K 预算直接用于 128K。

已有固定 SGLang `3f36fc08331f64cd63c4fbbdf39cc402f38bb38b` 源码包含 MistralForCausalLM 与 LlamaForCausalLM；因此可复用当前隔离 GPU 环境，不必为此再安装 vLLM、FastChat 或完整 Prometheus Python 包。这两个固定架构均已实际加载并完成开发评分，完整启动/运行信息保存在各自 runtime 文件；显存预算仍不能替代具体峰值测量。[固定 Mistral 实现](https://github.com/sgl-project/sglang/blob/3f36fc08331f64cd63c4fbbdf39cc402f38bb38b/python/sglang/srt/models/mistral.py)、[固定 Llama 实现](https://github.com/sgl-project/sglang/blob/3f36fc08331f64cd63c4fbbdf39cc402f38bb38b/python/sglang/srt/models/llama.py)。

已在与性能采样分开的独立时段，用固定 snapshot 启动 stock SGLang，使用独立端口 `30001`、初始 context/total tokens 32768、并发 1。不要直接复用固定 Qwen revision/fingerprint 的 `scripts/launch_sglang.sh`。原生 chat endpoint 按新模型自身 tokenizer chat template 渲染；不要手工再包一层 Mistral `[INST]` 或 Qwen 模板。保留实际 server info、模型 snapshot、启动参数与显存观测。

现存完整 judge JSON 的实测字符长度：correction P3 的 41 份为 22,216–48,885，16K Ours 的 41 份为 19,620–54,510；这不是 token 数。正式评分前用 judge 的固定 tokenizer 做一次入口预算核对。超限时记录失败或显式换更长上下文运行配置，不能静默丢掉旧可见证据。

## 原生判分请求与薄适配

Selene 官方绝对评分模板输出 `**Reasoning:**` 与 `**Result:**`，分数为 1–5；另有 reference/no-reference、Yes/No 和 A/B 模板。原生协议并非五维 JSON。[固定 reference 模板](https://github.com/atla-ai/selene-mini/blob/418c5ce2d8c4d89c5bd971ce60c7589eb8f9d8de/prompt-templates/absolute-scoring-with-reference.yaml)、[固定无参考模板](https://github.com/atla-ai/selene-mini/blob/418c5ce2d8c4d89c5bd971ce60c7589eb8f9d8de/prompt-templates/absolute-scoring.yaml)。

`runs/open-judge-preparation/prepare.py` 已从本轮取得的固定官方模板渲染两个合成样例，生成 `selene-absolute-scoring.request.json` 和 `selene-absolute-scoring-with-reference.request.json`。脚本没有网络操作，样例没有模型分数。请求结构为：

```python
request = {
    "model": "AtlaAI/Selene-1-Mini-Llama-3.1-8B",
    "temperature": 0.0,
    "max_tokens": 1536,
    "stream": False,
    "messages": [{"role": "user", "content": official_template.format(**fields)}],
}
# 服务准备好之后，才发送：
# POST http://127.0.0.1:30001/v1/chat/completions
# 不加 response_format=json_object；保留该模型的原生 Result 格式。
```

原生 HTTP 调用示例（上述准备样例本身仅做离线渲染，实际评分由 CLI 发送）：

```bash
curl --fail-with-body http://127.0.0.1:30001/v1/chat/completions \
  -H 'Content-Type: application/json' \
  --data-binary @runs/open-judge-preparation/selene-absolute-scoring.request.json
```

Prometheus 则使用 `[RESULT]` 后的 1–5 或 A/B。其固定官方 wrapper 默认 `ABSOLUTE_PROMPT_WO_REF`，仅传 `reference_answers` 不会自动切换 reference 模板；RAG 模板还有 `{relevant_context}`，但通用 `absolute_grade` 没有提供该格式变量。应显式渲染对应官方模板，不应误称“传了参考答案就已被模型使用”。[固定 prompts](https://github.com/prometheus-eval/prometheus-eval/blob/dcfb44272d5d0414832f5dbb8c2a05ebc2614234/libs/prometheus-eval/prometheus_eval/prompts.py)、[固定 wrapper](https://github.com/prometheus-eval/prometheus-eval/blob/dcfb44272d5d0414832f5dbb8c2a05ebc2614234/libs/prometheus-eval/prometheus_eval/judge.py)、[固定 parser](https://github.com/prometheus-eval/prometheus-eval/blob/dcfb44272d5d0414832f5dbb8c2a05ebc2614234/libs/prometheus-eval/prometheus_eval/parser.py)。

已实现的代码合同如下：

1. 复用 `judge_payload`、`evaluation_binding`、`rule_audit`、httpx transport 和独占输出创建。每维独立 rubric，`judge.protocol_manifest` 保存完整固定模板、五维 rubric、data policy、解析、参考答案策略与聚合声明，供已有 acceptance 比较；输出仍是 `offline_judge` 绑定格式。模板位于 `bench/assets/native_judge/`，包含固定来源、NOTICE 和 Apache-2.0 文本。
2. Selene/Prometheus 各自严格解析唯一的末尾 Result 字段，保留 `native_results` 中的 raw text、native score、reason、finish reason、API identity 和 usage。有效 1–5 才映射 `(score - 1) / 4`。默认 `na_policy=native_numeric_only` 不接受 NA；可选 `explicit_na_v1` 在原模板后追加明确的项目适用性扩展，合法 NA 记为 `not_applicable`、该维仍为 null。扩展不是作者原协议，尚需元评测。格式错误、HTTP 失败、非 stop 完成（包括 length）记为 `failed`，不是 NA；任一维失败则整行 status=failed，仍保留其余维度的诊断分数。
3. 主评测先用完整可见证据和自身历史、无黄金参考答案。需要 reference 的独立轨道才离线 join targets，并记录 `reference_targets_supplied`；黄金答案不是模型当时可见的 source。脚本历史与实际自身历史不一致仍单列，不把参考答案适用于不同历史当作既定事实。
4. 一次构造完整协议声明；不新增逐请求多层 hash。可配置 `context_window` 并提供固定 tokenizer，按完整 chat prompt tokens + max_tokens 做入口核对，超限不发送、不截断。`base_model`/`base_family` 必填；它们仍是明确的操作者身份声明。`script_history_mismatch` 未单独判分，输出 null，不能当作 false。

可调用 API：

```python
client = NativeJudgeClient(config, transport=optional_httpx_transport,
                           token_counter=optional_exact_chat_token_counter)
try:
    result = await client.evaluate(audit_record, reference=optional_offline_reference)
finally:
    await client.aclose()
# client.complete(messages) 返回原始 transport 结果；不强制套五维 rubric。
```

本地配置示例（model 名需与实际 served-model-name 相同）：

```json
{
  "provider": "local_openai_compatible",
  "model": "AtlaAI/Selene-1-Mini-Llama-3.1-8B",
  "revision": "427792f1c3e2073cb7da216924fd884b1ba496e0",
  "base_url": "http://127.0.0.1:30001/v1",
  "protocol": "selene_absolute_v1",
  "base_model": "meta-llama/Llama-3.1-8B-Instruct",
  "base_family": "llama",
  "na_policy": "native_numeric_only",
  "temperature": 0.0,
  "max_tokens": 1536,
  "context_window": 32768
}
```

```bash
python -m bench.native_judge --audit PATH_TO_SAVED_AUDIT.jsonl \
  --config PATH_TO_CONFIG.json --output NEW_SCORES.jsonl \
  --tokenizer-path PATH_TO_PINNED_LOCAL_SELENE_SNAPSHOT
# 参考答案轨道才添加 --references OFFLINE_LABELS.jsonl。
# tokenizer 仅 local_files_only 加载；不触发权重或 tokenizer 下载。
```

Prometheus 配置使用 `protocol=prometheus_absolute_v1`、模型/revision 表中固定值、`base_model=mistralai/Mistral-7B-Instruct-v0.2`、`base_family=mistral`。API/CLI 都不会安装依赖或启动服务器。针对测试覆盖 23 项协议、输入、完整证据、失败、NA、绑定、预算与文件不覆盖合同；预算计数显式 `return_dict=False`，避免 Transformers 5 默认 BatchEncoding 的字段数被误当 token 数。这些都是工程测试，不能当作真实模型有效性测量。

## GroUSE：先验证 evaluator，再用于本项目

GroUSE 是具有预期行为条件的 GQA 元评测，不是另一套 Cloud 生成质量标签。官方固定 dataset `illuin/grouse` revision 为 `caaaae0dacf5126279ae59609d26926e124b8032`，train 16 个供 prompt 开发、test 144 个供冻结后的检查。本轮没有读取 test 样本正文。[固定数据集](https://huggingface.co/datasets/illuin/grouse/tree/caaaae0dacf5126279ae59609d26926e124b8032)。

论文报告 Prometheus 2 7B 的总体条件通过率为 **52.78%**，并讨论其 1–5 训练格式对 boolean/null 指标的局限。该数字来自原论文，不是本项目复测，也不证明 Selene 必然更好。因此本方案要先实际测，而不是凭通用 judge 名声确定正式 evaluator。[GroUSE 原论文](https://aclanthology.org/2025.coling-main.304/)。

源码并非固定“三个 prompt”，而是 **四个模板、条件执行**：先 relevancy 与 completeness；relevancy 非 null 时再 faithfulness；为 null 时先 usefulness，若 usefulness 仍适用再 faithfulness。通常三次，最多四次调用。positive acceptance 与 negative rejection 是 nullable 状态的确定性推导，不是再问两次模型。relevancy/completeness 为 1–5 或 null，faithfulness/usefulness 为 0/1 或 null；请求失败 `Failed` 与“不适用” null 必须区分。模板将参考答案与候选组成 answer_1/answer_2，读取 answer_2。[固定执行路径](https://github.com/illuin-tech/grouse/blob/e4c57ba9eb14548c7642daca3ed4ff24d760c7fa/grouse/grounded_qa_evaluator.py)、[固定 DTO](https://github.com/illuin-tech/grouse/blob/e4c57ba9eb14548c7642daca3ed4ff24d760c7fa/grouse/dtos.py)。

官方 CLI 的真实参数为以下形式，不存在单独的 `--base-url` 参数：

```bash
# 官方独立环境，MODEL_NAME 为该环境 LiteLLM 已注册的模型路由。
grouse meta-evaluate MODEL_NAME OUTPUT_DIR --train_set --prompts_path PROMPTS_DIR
# evaluator 冻结后，去掉 --train_set 才运行 144 个 meta-test。
```

原 `call_llm` 使用 LiteLLM、默认 temperature=0.01/max_tokens=2048。`load_unit_tests` 当前未固定 dataset revision，且先加载整个 DatasetDict 再取 split。因此不能直接跑未经约束的默认 CLI 并称版本/边界已冻结。[固定 CLI](https://github.com/illuin-tech/grouse/blob/e4c57ba9eb14548c7642daca3ed4ff24d760c7fa/grouse/main.py)、[固定 loader 与推导规则](https://github.com/illuin-tech/grouse/blob/e4c57ba9eb14548c7642daca3ed4ff24d760c7fa/grouse/utils.py)。

最少新依赖的实现是复用固定 MIT prompts、适用性与条件比较规则，使用已有 httpx/Jinja，单独读取固定 revision 的 train 文件；保存许可归属。也可以在隔离官方环境中仅替换 transport 和固定 split loader，保留 `GroundedQAEvaluator` 与 `MetaEvaluator`。不要为 16 个测试把官方 datasets/LiteLLM/diskcache/matplotlib 等整套依赖加入主工程。条件比较只接受源码已有的数值比较与 `==None`，六项通过率及均值保持原口径，Failed 计失败而不是删行。[固定元评测](https://github.com/illuin-tech/grouse/blob/e4c57ba9eb14548c7642daca3ed4ff24d760c7fa/grouse/meta_evaluator.py)、[固定依赖](https://github.com/illuin-tech/grouse/blob/e4c57ba9eb14548c7642daca3ed4ff24d760c7fa/pyproject.toml)。

judge 输入只含 `input`、`actual_output`、`expected_output`、`references`；**conditions、metadata.goal/test_type 不进入模型请求**。原始 GroUSE JSON 协议与 Selene 原生 rubric 协议分开命名：可先在 Selene 上运行原始四模板测指令遵循，也可开发 rubric/适用性适配版，但后者必须注明改动，不能把项目五维分数当作原版 GroUSE 六项。Prometheus 的格式局限同理，不强塞 JSON 后声称保留了作者协议。

## 下一步的有界执行顺序

1. 薄 adapter、固定 GroUSE 元评测和两模型实际服务开发验证已完成。原生输出合同成功但语义漏检明显，不按接口成功认定质量通过。
2. 两模型同预算 train16 原始结果保留。Selene 的6次格式失败可由显式 `normalized_v1` 恢复，但缺少4–7次下游实际响应，随后完整新train已完成51次调用、64/96条件通过，但完整性仍仅5/16，见验证记录；Prometheus 32次均未产生官方可解析JSON，单纯类型规范化无效。模型/原生模板适配与原子条件判断继续分别改进，不把16例结果当可信区间保证。
3. 针对项目已见错误增加明确标为开发诊断的反例：附件 10 MB 与 50 MB、错误概括“1 MB 含附件”、把历史 assistant 断言当来源、无效 curl JSON、合理简短拒答与无依据长答。这些是规则/assistant 构造探针，不伪称新增人工标签。
4. 依据 train 元评测冻结 model/revision、四模板或适配协议、抽样参数、适用性与归一化，再运行 GroUSE test 144 一次。失败如实报告；若后续据此修改，明确它已转为开发证据，不能继续称未见元评测。
5. 对现有三组 P3 共 123 份实际回答完成五维模型评分，完整覆盖失败、NA、截断和困难案例，按现有会话配对验收。与 MiniCheck、规则 flags 的分歧用于定位缺口，不强行融合成“真值”。dev 可以用于继续优化方法；最终冻结的项目 test 保持独立。

此路线推进的是**独立开源模型评分证据**。通过固定 heldout 模型 evaluator 时使用现有 `heldout_evaluator_noninferiority_established`；不能设置仅限实际人工来源的 `quality_noninferiority_established`。它不要求用户先完成 123 份人工表，但也不把模型分数替代人类质量真值。

## 近期方法可复用的部分

2026 年 Autorubric 提供 rubric、分维聚合、agreement 等工程实践；可借鉴原子条件与显式聚合，但当前主工程已有 HTTP/绑定/配对统计，不必立即引入其 LiteLLM、statsmodels、sklearn 等完整依赖。[论文](https://arxiv.org/html/2603.00077v1)、[作者仓库](https://github.com/delip/autorubric/tree/3a85dc182a3b4a5ecf3a8eb4765b936d7645fd3e)。PReMISE 强调 rubric 的结构、偏好适配、可靠性和对抗稳健性不能只由高 agreement 代替；其论文链接的作者仓库本次返回 404，故只作研究启示，不列现成可运行依赖。[PReMISE 论文](https://arxiv.org/abs/2605.30803)。后续按已观测的范围漏判、引用校准和条件分支错误选择具体方法、权重或训练扩展；方法清单本身不算完成效果。
