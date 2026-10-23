# 回答代码的静态诊断

P3 增强检索的真实回答暴露出两个闭合 curl 请求体语法错误：Cloudant 第 8 轮的 `$or` 数组缺少对象闭合符，另一个 `$exists` 对象出现 `)`。二者均位于闭合代码围栏内，后面还有解释文字，不能把错误只归因于最终回答触及 1024-token 上限。

新增 `bench.code_audit` 对保存的完整 audit 做只读检查：识别普通及列表缩进的 triple-backtick 围栏，解析 JSON 块与 curl 的 `-d`/`--data`/`--data-raw`/`--data-binary`/`--json` 字面参数。记录原始代码偏移、payload 和 JSON 错误位置。shell 只做词法分组，不执行命令；外部文件 payload 不读取，不验证服务 API、认证、字段语义或 shell 变量展开。

```bash
python -m bench.code_audit --audit runs/example/audit-<run>.jsonl \
  --output runs/example/code-audit.jsonl
```

当前 BM25/P3 的 41 份回答未发现此检查范围内的 JSON 解析失败；hybrid/P3 的 41 份中有 4 个任务、9 个解析失败。其中两个是上述完整 curl 请求体，两个是带 `//` 注释的 JSON 示例，五个是单独的对象成员片段。后两类带有不同的展示意图，不能把九个 parser failure 全部称为九个错误 API 调用，更不能由 BM25 零失败推导其整体答案正确。输出的 `presentation_hint` 明确区分这些类型，最终仍须结合上下文审查。

初版未识别列表缩进的代码围栏，真实检查因此误报零失败；已修正正则并加入缩进回归。合同测试还覆盖多行 curl、Mango `$or`/`$exists` 字面键、不同命令参数、外部 payload 和不完整 shell 引号。原回答不改变，旧诊断与修正后的 v2/v3 文件分别保留。

这项能力为后续选择性修订提供具体触发信息。下一步应以冻结初稿、相同来源和完整额外延迟，比较不修订、全部修订及仅语法失败时修订；修复 JSON 形状不代表修复 API 语义或证据支持。当前未将这项诊断自动接入线上回答修改。
