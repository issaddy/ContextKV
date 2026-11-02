# READ/REDUCE/ANSWER 机制的离线审计

`bench.audit_reading_mechanisms` 读取 `bench.reading_mechanisms` 已保存的执行文件；输入和输出使用不同目录。

```bash
python -m bench.audit_reading_mechanisms \
  --input-dir runs/study/mechanisms-r1 \
  --output-dir runs/study/mechanisms-r1-literal-audit
```

每个 case 单独保存审计 JSON，manifest 保存原执行状态和字面质量状态。原始 case 文件的 SHA256 写入审计。原输入目录存在 `state/sessions.sqlite3` 时，只读加载真实 Attempt，记录最终 prompt 的 input hash 与已有 Computed 行；不会重新构建或修改提示。

观察量分开保存：

- `execution`：completed/failed、READ/REDUCE 次数、QA 和版本递增。
- `coverage`：所需单元、实际物理覆盖、来源文档、仍开放的需求。
- pair/correction：表单、平台、命令和资格条件中的字面词项；引用 ID 是否属于实际允许范围。
- long：原文逐条登记及坐标、重复 ID、原文不同值冲突；READ 质量量值与原文字面量的核对；实际 REDUCE 根 claim；最终 ANSWER 中全部计数和质量量值。质量统一换算为 g，冲突或多个不同量值保持可见。
- multihop：最终年份字面量；实际 claim 的父边、Unicode 原文跨度和三个关系的来源证据；年份 claim 是否包含完整三源祖先链。

这是 controlled fixture 的 literal/provenance audit，`benchmark_metric=false`。例如，执行完成、物理覆盖完整、REDUCE count=36 与最终 ANSWER=3 可以同时出现在同一条记录。来源缺失、未形成三跳链和错误量值保持原样。该命令不调用模型，也不补答案或 claim。

本地实现测试覆盖单位归一化、重复登记、computed 与最终回答分离、有年份但无完整三源链，以及输入文件不变。真实 R0/R1 的这份离线报告仍需在存放原始执行文件的容器内运行。

# 计算结论型 ANSWER 课程

`training.reader_computed_curriculum` 已接入 `training.reader_curriculum` 的 build/review/audit 三步。配置新增 `maximum_computed_cases` 和 `computed_excerpt_characters`，当前固定为8个 native train 父样本和256字符背景。每个父样本构造 control/changed 两个 ANSWER 输入，单独存放在 round2 目录。

```bash
python -m training.reader_curriculum build --config configs/reader_curriculum.yaml
python -m training.reader_curriculum review --config configs/reader_curriculum.yaml --concurrency 8
python -m training.reader_curriculum audit --config configs/reader_curriculum.yaml
```

父样本必须是 native 训练分区中已复核、原文支持且任务绑定正确的 ANSWER。实际历史、父 prompt/hash、父来源 Unicode 坐标保留。新的 `constructed_train_computed_scope` 来源使用独立命名的 dispatch 登记，附带明确的构造文字及坐标；不读取 development fixtures 或公开 test。typed reduction 从全部构造登记计算 count 和 sum，同 ID 重复登记不新增计数。结论明确标记为离线构造 oracle，不记作真实模型 READ/REDUCE。

最终输入由现有 `build_plan_candidates` 编译，保留全范围根结论和一个代表原文单元。修改组在未展示的登记中改变一个明确质量数字，其原始/修改来源 key、坐标、文本和 hash 都保留；完整范围 count 不变、sum 随之变化。每个 ANSWER 目标均由教师在新 prompt 下重新生成并复核，不使用旧 ANSWER。

配对审计要求 control/changed 各自明确回答完整范围的 count 和归一化质量总量；只复述代表摘录或混合局部与全局总量的目标被排除。READ 按输入与输出预算增加包数时，每个计划包都需独立的 fresh accepted 复核，包含无新事实窗口和中间 hop；任一包未完成、被拒或绑定不符都不能以终端窗口正确代替整对通过。`fresh_step_review_statuses` 保存逐包状态。

通过的新增样本与既有 native train 合并，父验证集保持不变，constructed 样本继续使用原 native task ID，不增加1000实际训练轮门槛的计数。

主 agent 已报告在指定容器完成这批课程构造；fresh教师目标待完成，尚无round2训练资产或效果结果。已完成的首轮 R1 训练数据和权重不受该构造流程影响。
