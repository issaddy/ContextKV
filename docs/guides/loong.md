# Loong：完整长资料的P1补充

[Loong论文](https://aclanthology.org/2024.emnlp-main.322/)考察多份论文、财报及法律材料中的定位、比较、聚合和推理。它是单次给定资料任务，用于补充长输入阅读能力。

固定代码提交为`6d2115b8b48a3d19412ccb52a0c9c4ee37869af4`。下载并保存原始文件：

```bash
mkdir -p data/loong/source
curl -fL https://raw.githubusercontent.com/MozerWang/Loong/6d2115b8b48a3d19412ccb52a0c9c4ee37869af4/data/loong.jsonl -o data/loong/source/loong.jsonl
curl -fL https://alibaba-research.oss-cn-beijing.aliyuncs.com/loong/doc.zip -o data/loong/source/doc.zip
```

转换器直接读取所声明的问题文件与资料归档，无需解压；中文ZIP文件名按实际UTF-8编码处理。

```bash
python3 -m adapters.loong --questions data/loong/source/loong.jsonl \
  --documents data/loong/source/doc.zip --output data/loong/set1-v2 --length-set 1
```

公开数据共有1600题。第一长度组323题全部成功转换，包括70题论文、150题法律、103题财报及1839个资料实例。每题是独立会话的第1轮，原始`question`、`instruction`与模板文本保留；仅将`{docs}`位置外置为统一证据来源。重复资料实例和法律文书编号保留，顺序固定为发布文件中的`doc`列表，不采用上游运行时随机打乱。参考答案的字典/列表在标准`targets.text`接口序列化为JSON，原始值另外保留在离线`reference_value`；它们不进入模型输入。

与上游实现的差异明确记录在manifest：

- 所有允许输入的资料均保留，不采用`get_doc_str`中按预算跳过整份文档的行为。预算核查在转换之后执行。
- 法律“Link the Links”任务按论文§3.2.4将事实与候选判决结果分别给出。发布数据的任务标记在`question`，上游代码却检查`instruction`；本适配从实际字段识别任务，保留原问题中的候选结果，文书只读取`content`。法律原始案名可能含待预测的分类信息，只保留在离线元数据；模型可见的来源地址与文档ID使用不含语义的摘要。第一组有35题适用这一协议修正；其他法律题仍读取`content + result`。该组采用上述适配协议。
- 财报文件必须唯一匹配。全数据中的部分名字可能匹配多个归档文件，转换器将整题记入`source_issues.json`，不使用参考答案选择文件。存在未解析任务时保留导出结果并以退出码2结束。第一组无此问题；其他分组使用前检查`manifest.json`的请求、导出及未解析数量。

## 完整预算与运行

完整来源通过 `bench.study` 的 `all_supplied_documents` 阅读路径执行；共享数值阅读与
`numeric_need_isolation` 在相同材料、需求及预算下比较。每题的全部原文保留在来源目录中，
由 READ 分片和有限 REDUCE 完成。
当前执行入口和成本口径见[运行手册](../study/final_runbook.md)。

## 保存答案的评分

`bench.loong_score`复用作者原始准确性/完整性rubric、问题组装及结构化参考值，使用已有本地completion transport。JSON格式的`--judge-config`采用`SelectorConfig`字段，显式声明模型身份、端点、分词器及完整输入/输出预算。解析分数直接使用作者首匹配规则；续跑复用已保存的评分记录。

```bash
python3 -m bench.loong_score --references data/loong/set1-v2/tasks.jsonl \
  --predictions SAVED-OFFICIAL.jsonl --judge-config LOCAL-JUDGE.json \
  --output runs/loong-scores --workers 4
```

缺失答案、输入超限、HTTP失败、输出截断及无有效分数均保留为缺失；报告有效分数均值、满分率及包含缺失项的均值界限。本地模型替代论文的GPT-4，报告结果适用本地评分协议。该rubric比较参考答案，来源支持另行核查。相关归属及协议修改见[Loong说明](../third_party/LOONG_NOTICE.md)。
