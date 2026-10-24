# Doc2Dial 原文与多轮监督适配

`adapters.doc2dial`将已经下载的作者文档与对话转为两个独立产物：可供现有检索器读取的完整原文语料，以及只供离线研究的原对话/grounding侧车。它不训练模型、不调用生成、不读取未来轮次作为在线输入，也不把新数据伪装成MTRAG。

## 原理与复用边界

[Doc2Dial](https://aclanthology.org/2020.emnlp-main.652/)提供有文档范围的多轮交互，包含条件查询、澄清及来源跨度。这比只有单轮引用格式更接近本项目的历史与适用条件问题。但其precondition/solution是作者说明中的模糊标注，不是完备逻辑规则或语义真值。[MultiDoc2Dial](https://aclanthology.org/2021.emnlp-main.498/)增加跨文档切换，同时排除了irrelevant场景；它的`respond_no_solution`也不能直接解释为整个语料无答案。

本实现按真实文件格式适配，没有复制官方HF的RC预处理：后者会反转历史拼接、只取紧邻user→agent，并把不连续支持span扩成最小/最大边界包络。这里保留原角色、turn ID、顺序和每一个源范围，不合并发言、不补助手回答、不扩大引用。

版本检查和源码/许可依据见[固定研究记录](../runs/multiturn-grounding-research/README.md)。Doc2Dial官方数据卡标示CC-BY-3.0；代码许可与数据许可分开记录，不因MultiDoc代码为Apache便推定其所有数据如此。真实数据不随wheel分发，使用时保留作者归属及来源记录。

## 可运行入口

先准备作者的documents和**明确训练分区**。以下为已验证的本地路径；其他位置替换路径即可。core CLI不自动联网下载，也不覆盖旧输出：

```bash
uv run python -m adapters.doc2dial \
  --documents runs/multiturn-grounding-research/data/doc2dial_v1.0.1--doc2dial_doc.json \
  --dialogues runs/multiturn-grounding-research/data/doc2dial_v1.0.1--doc2dial_dial_train.json \
  --dataset-version doc2dial-v1.0.1 --source-format v1 \
  --source-uri https://doc2dial.github.io/file/doc2dial_v1.0.1.zip \
  --split train --output runs/doc2dial-v1-prepared
```

v0.9 wOOD使用其独立文档/训练文件，显式传`--source-format v0.9`、`--dataset-version doc2dial-v0.9-wOOD`及另一个output。v1的`references`列表与v0.9的`reference`字典分别解析；后者每个span的多标签完整保存。两套对话有大量重叠，不能直接合并为独立训练样本；也不按文件名猜split或自动选择“最新版本”。

输出：

- `corpus.jsonl`：全部提供的完整`doc_text`，采用可逆的`[domain,doc_id]`身份；source为数据归档及文档fragment，version是数据发布版本，均不冒充原政府网页的有效版本。
- `supervision.jsonl`：每个原对话的完整turn序列、原role/utterance/turn_id/DA、原reference容器及精确grounding范围。连续同role、空reference、多标签保持原样；不是可直接送模型的SFT messages或在线oracle输入。
- `manifest.json`：格式、操作者声明的split、输入文件摘要和转换计数。摘要在文件边界计算一次，不增加请求期hash。

只有实际引用到的span按Python Unicode字符半开区间验证一次并复用。错位或缺失引用明确失败，不用`find`取第一次、不clamp、不归一化。`text_sec`在真实数据中存在遗漏嵌套正文的情况，故不使用它替换原文或拼支持范围。标签不进入corpus，不生成静态dependencies，也不把空reference自动标成全语料无支持。

## 接到已有检索与HTTP

```bash
uv run python -m adapters.retrieval build \
  --corpus runs/doc2dial-v1-prepared/corpus.jsonl \
  --index runs/doc2dial-v1-bm25 --snapshot doc2dial-v1.0.1
uv run contextkv --config configs/demo.yaml --state-dir runs/doc2dial-demo-state serve \
  --retriever-index runs/doc2dial-v1-bm25 \
  --evidence-corpus runs/doc2dial-v1-prepared/corpus.jsonl --corpus-id doc2dial
```

示例使用CPU mock；实际Qwen3改用对应scope和已经启动的引擎。HTTP只收到当前问题，检索器读取corpus；监督侧车不会加载。不得把ground-truth文档或reference spans送给检索器再称真实检索。完整文档较长时沿现有预算拒绝溢出，不静默裁掉正文。当前未为这些非交替对话扩展MTRAG的P1/P2/P3脚本gate。

## 实际验证与剩余问题

| 分区 | 原文档 | 对话/轮次 | 精确引用出现次数 | 边界 |
|---|---:|---:|---:|---|
| v1.0.1 train | 488 | 3474 / 44149 | 63950 | 2222个对话有连续同role；该版本没有空引用轮次 |
| v0.9 wOOD train | 488 | 3471 / 49972 | 102972 | 2772对OOD user/agent加431个no_solution，共5975个空引用轮次 |

独立逐条核对原role、文本、DA、reference容器、turn ID及字符范围通过；原始验证/测试对话未提取解析。v1完整488文档实际BM25建盘重载后，以固定首个用户问题取得5个完整文档，Qwen3输入5424tokens且可行；没有提供gold文档、标签或预期回答，0模型调用。这是接线证明，不是检索或回答质量评测。原件与复算脚本在`runs/doc2dial-integration/`。

OOD只能说明问题不适用于给定文档：例如在DMV材料下问VA事务，完整语料可能另有VA答案。真实target也有条件否定反转等噪声；地址正确不授予target语义资格。固定开发材料的有限字面重叠检查无命中，不代表全量去污染完成。

[FaithDial/CHARP对照研究](../runs/multiturn-grounding-alternatives/README.md)补充了另一约束：人工修订与history字段不保证模型需要利用历史；CHARP同知识、不同末问的成对设计更能隔离历史消歧。当前保留这些有效原理和数据接口，后续按真正的历史、条件及证据充分性问题设计机制；没有因适配成功就启动训练或重跑大矩阵。
