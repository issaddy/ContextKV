# Loong：完整长资料的P1补充

[Loong论文](https://aclanthology.org/2024.emnlp-main.322/)考察多份论文、财报及法律材料中的定位、比较、聚合和推理。它是单次给定资料任务，用于补充长输入阅读能力；不提供本课题所需的连续会话、资料版本演化或并发到达过程。

固定代码提交为`6d2115b8b48a3d19412ccb52a0c9c4ee37869af4`。下载并保存原始文件：

```bash
mkdir -p data/loong/source
curl -fL https://raw.githubusercontent.com/MozerWang/Loong/6d2115b8b48a3d19412ccb52a0c9c4ee37869af4/data/loong.jsonl -o data/loong/source/loong.jsonl
curl -fL https://alibaba-research.oss-cn-beijing.aliyuncs.com/loong/doc.zip -o data/loong/source/doc.zip
```

转换器直接读取所声明的问题文件与资料归档，无需解压；中文ZIP文件名按实际UTF-8编码处理。

```bash
python -m adapters.loong --questions data/loong/source/loong.jsonl \
  --documents data/loong/source/doc.zip --output data/loong/set1-v2 --length-set 1
```

公开数据共有1600题。第一长度组323题全部成功转换，包括70题论文、150题法律、103题财报及1839个资料实例。每题是独立会话的第1轮，原始`question`、`instruction`与模板文本保留；仅将`{docs}`位置外置为统一证据来源。重复资料实例和法律文书编号保留，顺序固定为发布文件中的`doc`列表，不采用上游运行时随机打乱。参考答案的字典/列表在标准`targets.text`接口序列化为JSON，原始值另外保留在离线`reference_value`；它们不进入模型输入。

与上游实现的差异明确记录在manifest：

- 所有允许输入的资料均保留，不采用`get_doc_str`中按预算跳过整份文档的行为。预算核查在转换之后执行。
- 法律“Link the Links”任务按论文§3.2.4将事实与候选判决结果分别给出。发布数据的任务标记在`question`，上游代码却检查`instruction`；本适配从实际字段识别任务，保留原问题中的候选结果，文书只读取`content`。法律原始案名可能含待预测的分类信息，只保留在离线元数据；模型可见的来源地址与文档ID使用不含语义的摘要。第一组有35题适用这一协议修正；其他法律题仍读取`content + result`。不能将这组结果当作上游代码逐字复现。
- 财报文件必须唯一匹配。全数据中的部分名字可能匹配多个归档文件，转换器将整题记入`source_issues.json`，不使用参考答案选择文件。存在未解析任务时保留导出结果并以退出码2结束，防止后续命令把部分导出当成全量成功。第一组无此问题；其他分组使用前应先检查`manifest.json`的请求、导出及未解析数量。

## 完整预算与运行

通用`bench.input_profile`使用共同候选装配和实际分词器检查独立P1输入，保留完整作者历史与来源。它不运行模型，也不预测P2/P3历史增长或服务容量：

```bash
python -m bench.input_profile --input data/loong/set1-v2/tasks.jsonl \
  --config configs/loong_input_profile.yaml --output data/loong/set1-v2/model32k
```

仅有分词器元数据的机器可加`--tokenizer-path PATH`，其分词器/模板摘要必须与配置的模型身份匹配。配置使用32K模型窗口和GPU池、4096输出预留及greedy生成；该输出预算用于较长的列表/结构化答案，与短会话1024预算实验分别报告。

当前323题完整输入为9738–56404 tokens，中位数31928；122题同时满足窗口、输出预留和GPU池预算，另外201题完整保留在`input-profile.json`中并标为预算失败。可运行题目的输入为9738–28230 tokens，其中106题超过16K。该子集为法律107题、论文7题、财报8题，不能代表完整Loong的领域分布或总体质量。

预算核查输出完整可运行任务、对应划分和全部逐题长度/输入哈希；没有删除单题中的资料。原323题是场景覆盖分母，报告生成结果时仍须说明201题预算失败。这里的`test`仅是统一runner固定任务的运行入口，不宣称新增作者测试分区。

201题中145题的输入已达到/超过32K，56题仅在加上4096输出预留后超限。122题v2基线及固定32B原rubric评分已全部完成，无生成截断、无缺失评分。平均分64.36/100，法律107题60.39、论文7题84.29、财报8题100.00；平均完整响应3.810秒、P95 5.983秒，平均输出62.14 token。记录目录为`runs/loong-v2-20261101/`。初次v1接入暴露了法律原始案名，108份回答及评分全部作废，v2结果单独保存。

```bash
python -m bench.b6_joint_sweep --matrix configs/benchmark_loong.yaml \
  --stage test --output runs/loong-system --job-timeout 3600
```

确认输入路径与资源配置后加`--execute`执行。该通用矩阵入口沿用普通解码；本次已报告的实测由`runs/loong-v2-20261101/run_long.py`及`matrix_worker.py`启动原生NGRAM（draft 12、breadth 1），保持原矩阵的greedy、4096输出、32K GPU槽位、16 GiB Host与单并发，完整启动参数已归档。当前配置是P1固定重建的长资料基线，没有解释器；不能用它估计B/C在多轮并发中的独立收益。

## 保存答案的评分

`bench.loong_score`复用作者原始准确性/完整性rubric、问题组装及结构化参考值，使用已有本地completion transport。JSON格式的`--judge-config`采用`SelectorConfig`字段，显式声明模型身份、端点、分词器及完整输入/输出预算。默认每份答案只评分一次，无回答重生成或评分修复。

```bash
python -m bench.loong_score --references data/loong/set1-v2/model32k/tasks.jsonl \
  --predictions SAVED-OFFICIAL.jsonl --judge-config LOCAL-JUDGE.json \
  --output runs/loong-scores --workers 4
```

缺失答案、输入超限、HTTP失败、输出截断及无有效分数均保留为缺失；报告有效分数均值、满分率及包含缺失项的均值界限，不把缺失隐藏在有效样本分母中。本地模型替代论文的GPT-4，分数不与原榜单直接比较。该rubric比较参考答案，不读取完整来源，不能代替来源支持核查。相关归属及协议修改见[Loong说明](../third_party/LOONG_NOTICE.md)。
