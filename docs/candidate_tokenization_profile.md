# 保持逐 token 等价的候选分词优化

2026-10-24。在同一组 15 个真实冻结快照上，完整 APPEND/REBUILD 候选构建的本地耗时从 **13.31 降到 4.63 ms**。这里是每个 case 测量中位数的等 case 平均，不是线上 P95。15 对候选的完整字段（实际计时 `build_ms` 除外）、渲染文本、token IDs、证据 span 和输入哈希与原版完全一致。

主要改动位于 `controller/templates.py` 和 `controller/candidates.py`。没有改变 planner、材料选择、required/visible evidence、消息顺序、生成预算、模板文本或默认策略，也没有删除 APPEND/REBUILD 中的任何候选。下面分别记录本机完整候选构建与服务器分词测量，不能把局部收益按比例换算为 control、TTFT 或整轮性能。

## 原路径与小改动

正常请求只调用一次 `build_candidates`，引擎重启后的重新报价才会再构建。原路径在这次调用内分别渲染、完整分词 APPEND 和 REBUILD；其中 system 和多段已提交 assistant 消息内容相同。已有 cProfile 也显示分词占主要时间。`last_input_ids + last_output_ids` 不能直接当成重新渲染的前缀：真实生成 IDs 与已提交 assistant 文本再次编码并不必然相等，本改动没有采用这种复用。

新路径先原样构建两套消息和 spans，再调用 `render_many`。普通 tokenizer 用 Hugging Face 官方完整 prompt batch 编码；固定 Qwen backend 使用下面已审查的特殊 token 边界复用。`render()` 保留单 prompt 的原始编码路径作为等价性参照；只实现 `render()` 的旧 adapter、以及覆盖了 `render()` 的自定义模板仍走原语义。

固定 backend 每次都先由同一个 Jinja 模板生成**完整文本**，然后保留 `<|im_start|>` 分隔符切片，只在本次调用内把完全相同的字符串片段编码一次，最后按原顺序拼回 token IDs。没有按普通文字、证据 body、字节长度或猜测 BPE 边界切分。字面 `<|im_start|>` 出现在证据或回答中时也保持 tokenizer 原来的特殊 token 语义，不被删除或转义。

这个去重表只存在于 `render_many` 的局部变量中，其规模受当前两套候选文本限制；函数返回即丢弃，没有跨请求/跨会话文本缓存或新的持久状态。没有增加请求中的 cryptographic hash。已有每个候选的一次 token hash 和证据验证继续执行；初始化时只比较本来已计算的 tokenizer fingerprint 来选择实现。

`build_ms` 仍是从本次候选构建开始到该候选对象完成的真实累计时间；批量编码后首个候选也包含联合 batch 的等待，不应将两个值相加解释为两次独立成本。service 的 `planning_control_ms` 使用真实 wall time，不累加该字段。

## 窄适用合同与源码依据

快路径仅在已有 tokenizer fingerprint 为 `e360e6f6266884e6667874f3a7d158ca20c4639e2e1dc80427e1edc2c69a38c7`，且 `split_special_tokens=False` 时启用。它对应固定 Qwen2.5-7B-Instruct tokenizer、tokenizers 0.23.2；其 backend 配置包含 NFC、确定性 BPE（dropout=None）和只处理 offset 的 ByteLevel postprocessor。`<|im_start|>` 是非归一化的 AddedToken，single_word/lstrip/rstrip 均为 false。该精确配置的 22 个 AddedToken 中，没有另一个 token 会跨越或覆盖所选切分边界。

官方 `v0.23.2` tag 对应固定 commit `88a4498ad4ea1a9487b0a9b0ff881383fd5a06a3`；其 [Cargo 版本](https://github.com/huggingface/tokenizers/blob/88a4498ad4ea1a9487b0a9b0ff881383fd5a06a3/tokenizers/Cargo.toml#L8) 与本地依赖一致。源码审查单独核对了下面的流程，而非只根据测试推断边界。

[AddedVocabulary](https://github.com/huggingface/tokenizers/blob/88a4498ad4ea1a9487b0a9b0ff881383fd5a06a3/tokenizers/src/tokenizer/added_vocabulary.rs#L488-L524) 先用非归一化 token 表切分原始文本，再只归一化剩余片段；其左侧优先最长匹配及词/空白规则在[匹配实现](https://github.com/huggingface/tokenizers/blob/88a4498ad4ea1a9487b0a9b0ff881383fd5a06a3/tokenizers/src/tokenizer/added_vocabulary.rs#L361-L445)中明确处理。[PretokenizedString](https://github.com/huggingface/tokenizers/blob/88a4498ad4ea1a9487b0a9b0ff881383fd5a06a3/tokenizers/src/tokenizer/pre_tokenizer.rs#L69-L119) 跳过已有 token 的片段，在各剩余片段内操作，随后[按片段次序合并 IDs](https://github.com/huggingface/tokenizers/blob/88a4498ad4ea1a9487b0a9b0ff881383fd5a06a3/tokenizers/src/tokenizer/pre_tokenizer.rs#L198-L244)。

主流程的[提取→预分词→模型顺序](https://github.com/huggingface/tokenizers/blob/88a4498ad4ea1a9487b0a9b0ff881383fd5a06a3/tokenizers/src/tokenizer/mod.rs#L712-L731) 与这一配置的[无 dropout BPE](https://github.com/huggingface/tokenizers/blob/88a4498ad4ea1a9487b0a9b0ff881383fd5a06a3/tokenizers/src/models/bpe/model.rs#L562-L570)、[仅改变 offset/sequence ID 的 ByteLevel 后处理](https://github.com/huggingface/tokenizers/blob/88a4498ad4ea1a9487b0a9b0ff881383fd5a06a3/tokenizers/src/pre_tokenizers/byte_level.rs#L164-L185)一起保证：在这个不可跨越的已注册 token 边界切开、保留分隔符并重用相同片段，得到的 ID 序列与完整编码相同。这里只需要 IDs，不复用 token offset 映射。

这不是任意 BPE/tokenizer 都可分段拼接的结论。源码和配置审查之外，测试还加入会跨该边界匹配的额外 AddedToken，确认新 fingerprint 自动回落到完整 prompt batch；关闭特殊 token 识别也走完整 batch。初始化后 tokenizer/backend/flags 保持不变是现有固定 fingerprint 合同，配置变更应重新构建 Template；没有为此每请求重新扫描或 hash backend。

## 实际本地测量

数据仍为 `runs/remote-h200-expanded/decision-dev-stratified/cases.jsonl` 的 15 个真实快照，原 query/evidence 从对应 Cloud dev benchmark 恢复。每个 case 先暖一次两种实现，再进行十次测量；case 和 before/after 顺序用固定 seed 随机交错。逐字段与原冻结候选比较后，单独采集各 150 次构建的 cProfile。比较脚本保留原 `templates.py`、`candidates.py` 版本，未覆盖历史 benchmark。

本机 macOS 15.7.7、ARM64、Python 3.12.13，14 个逻辑 CPU，transformers 5.19.0/tokenizers 0.23.2。默认组没有设置 `TOKENIZERS_PARALLELISM` 或 `RAYON_NUM_THREADS`，批量编码使用库的自动并行；没有采集每个 batch 的实际 worker 数。另起进程设置 `TOKENIZERS_PARALLELISM=false` 作为串行对照。

| 完整候选构建 | 原版 wall | 新版 wall | 原版 process CPU | 新版 process CPU |
|---|---:|---:|---:|---:|
| 默认分词并行 | 13.31 | 4.63 | 13.31 | 12.88 |
| 禁用分词并行 | 13.05 | 10.04 | 13.05 | 10.04 |

单位 ms，均为 15 个 case 中位数的平均。默认模式 wall 下降约 65.2%，process CPU 仅下降约 3.2%；串行模式约下降 23.1%。因此主要 wall 收益来自批量并行，不能称为同幅算量降低。请求内重复片段字符数平均为总片段字符数的 23.17%（各 case 7.03%–44.13%），与串行复用效果一致。

在固定渲染文本的前置原型里，逐条分词、完整 prompt batch、特殊 token 分段 batch 的 wall 分别约为 11.92、7.86、3.52 ms。这说明只做安全的完整 batch 已有收益，窄范围复用又减少了重复编码。该原型的运行顺序不是上述完整构建的交错协议，只作路径定位，不替代主比较。

默认并行下的 150 次构建 cProfile 总耗时为 2.05→0.73 s，native `encode_batch` 的累计 wall 为 1.79→0.47 s，Python 发起次数从每对两次变为一次；token hash 累计约 0.058→0.057 s，未通过删减哈希或验证获得收益。新路径仍需完整模板渲染、完整 token hash 和证据验证。JSON 中的 P95 是 15 个 case 中位值的分位数，也不是线上请求 P95。

## 验证与复现

真实 tokenizer 测试覆盖 Unicode/NFC 组合、中文、emoji、CRLF/空白、字面特殊 token、残缺标记、共享历史和省略显式 system 的输入；新批量结果同时与原 `render()` 和原生 `apply_chat_template(tokenize=True)` 比较。另验证旧 scalar-only adapter、自定义 render 覆盖、完整 batch 回退、跨边界新增 AddedToken 和 split-special 配置改变。全套启用真实 tokenizer 后 **392 tests + 6 subtests 通过**；之后新增边界组合的专项 4 tests 也通过。仅有原有 Starlette/httpx 弃用提示，Ruff 通过。

```bash
CONTEXTKV_TEST_HF_TOKENIZER=1 python -m pytest -q
PYTHONPATH=. python runs/candidate-tokenization-profile/compare_candidates.py
TOKENIZERS_PARALLELISM=false PYTHONPATH=. \
  python runs/candidate-tokenization-profile/compare_candidates.py
```

证据在 `runs/candidate-tokenization-profile/`：原实现备份、`prototype.py`、`compare_candidates.py`、两组 `candidates-*-comparison.json` 和 before/after cProfile。

## Linux 容器验证与线程预算

同一固定模型、tokenizers 0.23.2、Python 3.12.3，在有 192 个逻辑 CPU 的 Linux 容器中运行 `linux_compare.py`。15 个冻结案例的 30 个候选与原版及原冻结 IDs、文本和哈希全部一致。每案例随机交错前后版本各 20 次；此处测量完整文本渲染与分词，不包含证据组织、数据库、RPC 或 GPU 生成，不能与上表完整候选构建直接相减。

| 分词进程配置 | 原版 wall | 新版 wall | 原版 process CPU | 新版 process CPU |
|---|---:|---:|---:|---:|
| 库默认并行 | 22.39 | 7.51 | 22.28 | 40.12 |
| `RAYON_NUM_THREADS=4` | 22.61 | 7.58 | 22.49 | 17.44 |
| `TOKENIZERS_PARALLELISM=false` | 22.48 | 16.92 | 22.35 | 16.83 |

单位 ms，等案例中位值均值。默认并行虽然 wall 降低，但 CPU 时间增加约 80%；四线程在这次单进程测量中保持相近 wall，并使 CPU 时间下降约 22.5%。因此部署时应明确设置分词线程预算，并在多进程/并发负载下检查 CPU 竞争；这组结果不证明四线程适合所有机器。控制器和原生客户端 manifest 记录请求的线程环境，代码没有隐式修改进程全局线程设置，也不把环境变量声明当作实际 worker 数测量。

原始 `linux-default.json`、`linux-threads4.json`、`linux-serial.json` 与脚本已保存在相同证据目录。工程代码保留请求内复用，没有新增持久 token 缓存。
