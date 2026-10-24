# 来源计划、回答实现与宿主引用的集成合同

2026-10-24。本文件将[来源绑定生成设计](source_bound_generation_design.md)接到现有会话、回执、完整 QA 与缓存协议。**A2 不能通过修改 `GenerationResult.text` 后直接走原来的单次生成提交来实现。** 最小实验先重放冻结输入、独立保存两次模型原件和最终发布文本；生产首片再增加一个有限两阶段事务，并先强制下一轮从已发布 QA 重建。是否恢复 APPEND 优化应晚于语义及恢复合同验证。

本次仅代码审查与 CPU 反例，没有运行模型、改引擎、改核心或访问远端。审查基线为 `faba5b150f7f493f073f5d66f6d35d49db8d1060`。原生版本仍为 `3f36fc08331f64cd63c4fbbdf39cc402f38bb38b`；Qwen3 scope 的新增启动校验不改变以下 token/会话语义。

## 1. 当前单次生成合同实际在哪里

| 当前接点 | 已有语义 | A2 不可沿用的隐含等同 |
|---|---|---|
| [`SGLangEngineClient.generate`](../controller/engine_client.py#L196) | 一份完整 `input_ids` 对应一次原生请求；返回原生 text、output_ids、receipt；流式必须有完整 terminal 与 DONE | 最终宿主答案不一定是这次原生 text，JSON/分组包装和宿主引用不在 raw output IDs 中 |
| [`Controller._validate`](../controller/service.py#L112) | 检查 rid/attempt/epoch/restart/input hash/finish、token 类型和 cap，拒绝不完整生成 | 该检查不证明任意替换后的 text 对应 output_ids，也不证明两阶段全完成 |
| [`commit_turn_and_outbox`](../controller/session_store.py#L234) | 同一事务将 `result.text` 写入 TurnRecord 和 ExecutionLedger.messages，把 `result.output_ids` 写入两者，并在成功后排旧 epoch 退休 | 目前假定这些字段来自同一次未改写的引擎结果；没有 PublishedAnswer 或两个 stage 的概念 |
| [`ExecutionLedger` / `TurnRecord`](../controller/schema.py#L71) | 分别保留执行 input/output IDs、已提交消息、完整用户/助手历史 | 宿主添加引用后，published answer、native output 和下一轮模板编码是三个对象 |
| [`build_candidates`](../controller/candidates.py#L55) | APPEND 使用 execution.messages；REBUILD 使用 raw_history；LCP 与上次真实 input+output 比较 | 不能把未实际 forward 的 citation token 伪装成 last_output_ids 或已缓存长度 |
| [`verify_rendered_evidence`](../controller/templates.py#L203) | 在候选边界检查所有消息原文按序完整进入实际模板；不只检查证据正文 | A2 的 published QA 必须通过同一检查，不能删除 `</think>` 等内容来迁就模板 |
| [`_recover`](../controller/service.py#L326) | RESULT_READY 可重验并提交；SUBMITTED 取原生 status.result；不允许结果未知时盲目重发 | 若宿主渲染只在正常 HTTP 返回路径执行，恢复路径会遗漏或重新选择计划 |
| [`OnlineSession.commit`](../adapters/mtrag.py#L172)、[`run_sessions`](../bench/run_sessions.py#L118) | 只有 handler 成功返回后才将助手文本放进下一轮；失败保留当前任务，依赖后续轮为 unfinished | 第一次计划不能当业务助手回合提交；第二次实现也须等发布物持久化才能返回成功 |

这里不要求 `encode(native_text) == native_output_ids` 对所有 tokenizer 恒成立：原生 detokenization、停止 token、特殊 token 可使回编码不同。要求是**保留原始生成轨迹，不将宿主再加工的 token 假称为那条轨迹**。下一轮输入始终由完整消息经实际模板重编码，不能做普通 BPE 字串拼接；当前 [`render_many`](../controller/templates.py#L124)也明确不拼接 generated IDs。

### 已复现的最小错误接线

[CPU witness](../runs/source-bound-integration-review/witness.py)用真实 Controller/SessionStore/ByteTemplate/MockEngine，刻意做一个错误适配器：原生输出 `A raw sentence.`，仅把返回 text 换成 `A raw sentence. [E_f00d]`，原生 output IDs 与 receipt 不变。[记录](../runs/source-bound-integration-review/witness.json)显示：

- 正常返回被现有 `_validate` 接受并 COMMITTED；TurnRecord.text 已带引用，assistant_output_ids 仍解码为无引用文本；下一轮 APPEND 仍 feasible。
- 模拟原生已完成但 HTTP 响应丢失后，现有恢复取得原生保存结果，提交的却是无引用文本。
- 因此同一宿主“功能”因返回/恢复路径不同产生不同发布物。该反例证明新接线缺少发布事务，**不证明当前原生实现会用错误 KV 计算**，也不是现有未加工单次生成协议的故障。

## 2. 原生 session 与 KV 的边界

固定原生 [`GenerateReqInput.session_id`](https://github.com/sgl-project/sglang/blob/3f36fc08331f64cd63c4fbbdf39cc402f38bb38b/python/sglang/srt/managers/io_struct.py#L181)注明：顶层 session_id 是稳定身份，与会重建 prompt 的 session_params 不同。scheduler 的 radix-native 分支直接用请求的完整 input_ids 创建 Req；研究 guard 还拒绝 `session_params`。不能把这里的 APPEND 理解为“引擎无条件在旧输出之后接几个新 token”。

原生 cache 按实际请求 token key 匹配；[`checkpoint_into_tree`](https://github.com/sgl-project/sglang/blob/3f36fc08331f64cd63c4fbbdf39cc402f38bb38b/python/sglang/srt/mem_cache/unified_radix_cache.py)从 `full_untruncated_fill_ids[:up_to]` 取得 token，按 component/page 的有效长度写入 RadixKey。研究 [`finish`](../engine_patches/contextkv_runtime.py#L478)的 `input_hash` 是收到的输入、`inserted_cache_tokens` 是真实 Req 计数；[`publish_result`](../engine_patches/contextkv_runtime.py#L726)保存原生 text/output_ids 供恢复，不知道外层宿主渲染。

所以：

1. 已缓存的 raw realization 后缀不会因宿主补引用而自动变成 published 后缀；旧 inserted count 不能改名为已缓存最终答案。
2. 若下一轮真的发送重新渲染的完整 published QA，原生可重用它与旧请求**实际相同且仍驻留**的前缀，分歧之后重新 prefill。理论 LCP、soft ref、retire ACK 均不是实际命中证明；仍看新请求 receipt 的 device/host/prefill。
3. 单独将 `output_ids` 换成 `encode(published_text)` 更糟：那不是实际采样轨迹，不匹配原生恢复结果，也不能证明对应 KV 存在。清空 IDs 以绕开检查同样不能当已有执行账本完整。
4. native completed/epoch COMMITTED 是一次原生请求终态，不等于整个业务 A2 已发布。宿主仍须等两阶段与渲染的业务事务完成。
5. 不需要为首片新增 KV 改写、树扫描、强制 token 回灌或第三次“补齐缓存”调用。原生没有当前项目已验证的 post-render KV 修改合同，不能用 dummy generation 或零输出请求冒充它。

## 3. 最简隔离实验：先不碰业务 ledger

复用现有冻结的 8B 全开发输入（22 会话/159 轮）、真实模型 scope、HFTemplate 和 NativeSGLangClient；不调用 `Controller.serve_turn` 来装载中间步骤，也不以 test_mode 绕模型身份。实验身份明确为 frozen-history generation diagnostic，不是 B0/B6/APPEND 性能实验。

每个 case 保留原当前 query、完整实际前序 QA、完整当前五源及稳定引用表。两次调用都携带完整材料；计划、目录和说明只是本轮私有指令，不成为外部 EvidenceBlock，也不进入任何下一轮业务 QA。当前完整消息验证、证据 span 验证及预算溢出失败规则继续复用。历史助手事实不能升级为可选 source unit。

| arm | 调用与发布 | 有界预算 |
|---|---|---|
| A0 | 先保留已经取得的同冻结输入原稿作历史质量锚 | 原始 cap1024；旧耗时不能当本次受控性能基线 |
| A1 | 一次普通提纲，再一次自由回答；提纲仅本轮私有 | cap384 + cap1024，不补生成重试 |
| A2 | 一次合法 unit 分组计划；一次返回与 group 一一对应的 realization；宿主附该组引用 | cap384 + cap1024，第二阶段是一遍调用，不能每组另给1024 |
| Extractive | 同一个 A2 计划由宿主复制完整所选单元与引用 | 无额外模型调用；超最终预算标不适用/失败，不能截断原文伪成同预算成功 |

这相当于最多 636 个新增模型请求（A1/A2 各 2×159），实际因失败提前停止而减少的调用仍记为未执行，不从任务分母删掉。若要比较性能，另预声明同时执行的新 A0、顺序/资源控制与重复，不能将历史 A0 时间同新 A1/A2 时间直接相减。第一轮不立即扩模型×缓存写策略的大网格。

A2 的第二次输入应使用既定 plan 对应的本轮 unit 内容/IDs，不让 realization 再替换 source 选择。一个 JSON 数组或对象将句段绑定到预定 group 序号；宿主验证组数/序号、内容类型、literal E_hex 禁入后进行一次确定性渲染。具体输入 schema 和失败条件须在生成前冻结；不能从生成答案挑组、猜错 ID 或修坏 JSON。现 [CPU 原型](../runs/support-unit-prototype/README.md)只实现机械出处，不是完整 A2 runner，也没有 `uncovered` 的业务语义。

### cap、空计划与 length 必须先定义

`max_new_tokens=1024`控制实际生成 token，JSON包装占这份预算。宿主再加引用可能让最终发布文本超过1024；这两个数量必须分别记录。若沿设计的“最终答案 cap1024”作共同限制，则另以实际 tokenizer 对**完整 published text**编码一次，检查发布预算，超限保留原件并失败，不删引用/删句/补发。不把宿主字符数叫生成 token，也不暗增模型 cap 补偿 JSON 开销。

合法空计划表示没有所选来源，不自动等于空答案成功。必须预先声明一个可见的不足/拒答输出，或把该结果标未覆盖；社交回应与实质事实段分开但都保留。`coverage=supported`只是模型自报，不是质量门槛。计划 length/坏结构不能当成功空计划；实现无法得到完整合法输出时不发布半段。A0原稿的原有length输出仍作为截断结果保留，不能为凑齐成功率静默把A2失败改成A0 fallback。

每 case 至少保存：两阶段实际 messages/input_ids、raw HTTP/text/output_ids、finish/error、sampling/grammar、plan原件与解析对象、selected unit原key/spans、realization原件、完整 published text、发布文本编码及渲染协议、各阶段时间/预算。发布文本编码是宿主产物，字段不得命名为 native output IDs。只在输出工件绑定已有源身份，不引入每句新 hash 体系。

## 4. 后续生产首片：两个有限 stage，一个业务 commit

只有隔离实验值得继续时，才做以下小接线。不是新的通用 DAG/任务系统，也不改变原生 admission。

- **原生结果保持原义。** `GenerationResult` 与 `CompletionReceipt`原件不改。一个业务 Attempt 关联固定的 `support_plan` / `realization` 两份 stage result，分别保存实际输入身份与原生 RID。RID可从已有父 attempt ID 加固定 stage 后缀派生，不另建身份哈希。
- **增加明确发布物。** 保存 `published_text`、宿主渲染版本、选定 plan/realization 关系与发布预算数据；不能将它塞回 raw GenerationResult.text。业务成功响应使用发布物，审计仍能取两份 raw 结果。
- **仅一次语义提交。** 两个 stage 不能分别调用现在的业务 `serve_turn` 或 `OnlineSession.commit`。业务 version/raw_history_version只增长一次，TurnRecord.assistant_text记录用户真正收到的 published 文本。
- **首次不造执行 ledger。** A2完成后不把 realization prompt（含私有plan）写成下一轮业务 ExecutionLedger.messages。首片用明确的“需要重建/无可用于APPEND的业务执行账本”状态，下一轮从 published raw_history构造REBUILD。旧的单阶段路径及字段解释保持不变。
- **输出IDs分离。** 不能将 published编码放入原始 `assistant_output_ids` 冒充生成，或把 raw实现IDs说成publishedIDs。实现时须显式表示 host-rendered出版物与raw stage输出；若首片让旧assistant_output_ids留空，必须有新的明确来源状态/阶段记录，且禁用依赖这个字段的旧长度预测，不能用零长度获得成本优势。更合适的是独立出版物记录，旧字段仍只表示确实对应的native输出。
- **按真实阶段清理。** 中间 native session 是本轮临时执行资源，成功/失败确定后用原有 retire outbox清理。它们的原生completed不是业务完成。下一轮新session仍可能从共享radix复用相同前缀；退休释放soft refs不等于物理KV清空，不宣称重建等于冷启动。

最小原子边界是在现有SQLite事务中同时持久化最终发布物、业务TurnRecord、版本CAS与待退休操作，再向调用方返回成功。两份raw结果应早于此持久化；格式错误也要可追溯。若阶段1完成而阶段2失败/未知，不得前移业务历史，也不得为了继续会话省略这一轮。

现成本与长度接口也需要范围约束：[`output_prior`](../controller/cost_model.py#L63)和[`candidate_features`](../controller/output_length.py#L31)读历史 assistant_output_ids长度，单阶段decode误差不能直接解释A2最终文本/两次推理。首片用固定重建、禁旧拟合做自适应选择；待新采样分清计划长度、实现长度、发布长度后再谈B联合优化，不将Qwen2.5或单调用标定改hash冒充覆盖。

## 5. 恢复、流式与下一轮历史

| 业务状态 | 允许恢复动作 | 不允许动作 |
|---|---|---|
| plan已提交、结果未知 | 查该stage原生状态；若完整raw已保存则复用并按冻结协议解析 | 把native COMPLETED当业务commit；重发计划以选更好来源 |
| plan持久化、realization未提交 | 从同一不可变plan继续唯一realization提交 | 重新选unit、改变已冻结输入、提交plan为助手回答 |
| realization结果未知 | 查该stage status，恢复同一raw result；无法找回则按现显式abandon语义失败 | 另采样一次答案冒充幂等恢复 |
| raw实现已保存、published未保存 | 按同render版本确定性重建、验证预算，再原子commit | 正常路径有引用、恢复路径提交无引用raw文本 |
| published与commit已完成、客户端丢响应 | 相同request返回同一持久化发布物；不再调用模型/重渲染为新版本 | 返回status中的raw实现文本代替业务发布物 |
| stage终止/解析失败 | raw/error保留，业务历史不变，清理已知终态资源；后续依赖轮计unfinished | 将失败/空计划/超预算删行，或自动补A0成功 |

生产首次A2应缓冲最终发布物，**不要把模型实现的SSE token直接作为最终用户答案流出**。当前stream统计的是原生新增token到达，不含后续宿主引用和校验；若已经对用户部分发布，就不能再声称失败没有发布。要支持逐组流式需另定义组级有效性、不可撤销发布和恢复协议，当前首片不扩它。

下一轮 P2/P3 的两个历史持有者必须一致：store.raw_history与 `OnlineSession` 均使用 published文本，不能一个使用raw realization、另一个使用补引用版本。完整QA仍须通过实际HF模板检查；Qwen3的合法模板转换若删去历史内容，应像现合同一样失败，不清洗已发布答案来伪称保留。

未来恢复APPEND时，应分别保存 **业务消息前缀** 与 **最近真实原生执行 token轨迹**：按完整published消息渲染，计算与真实轨迹的LCP只作诊断，再对真实新candidate进行原生quote/generate。当前执行messages一字段混两种用途，不能仅改一个flag就打开。若要把支持计划长期留在输入以提高命中，它会改变跨轮输入/证据与成本协议，须独立声明；默认不把私有推理加入历史。

## 6. 全阶段成本与评价绑定

业务完成时间从业务arrival到发布物持久提交/实际响应完成的较晚者，沿现 `max(t_response_done,t_commit)-t_arrive` 口径。分别记录计划请求、实现请求、grammar等待/编译、宿主解析/渲染、两次完整prefill、生成token、序列化/持久化和任何必要quote/prepare/retire等待。两阶段串行的wall总时间不由两个server_elapsed相加替代；跨client/server时钟不相减。

- 缓冲A2的可用答案首字节在发布阶段；计划first token和实现first token另列，不能冒作用户TTFT。
- native TPOT仅对应各阶段raw输出，宿主复制引用不计decode；published长度另列。
- grammar/JSON失败、结果丢失、length、预算失败、未覆盖、后续unfinished均进入固定分母。重复调用、缓存/池占用若有都计入，不用成功子集报便宜。
- 质量评价对象必须是**用户实际published答案**，并能回到输入五源、完整实际QA、plan、raw实现与宿主投影。现 [`export_evaluation`](../bench/quality.py#L127)和 [`quality_acceptance`原run绑定](../bench/quality_acceptance.py#L81)默认一份answer/input_ids/output_ids；不能给published答案配raw实现IDs，或把宿主重新编码伪成原生IDs让旧gate通过。
- 隔离原型先用明确的pipeline审计侧车、不声称已有B6资格；正式接入时在工件/评分边界一次绑定出版物与两阶段来源，保留旧单阶段规则。机械引用、visible support、cited support、对象/条件、回答性、未覆盖和代码真实性继续分开，不能用host渲染成功代替质量。

## 7. 最少验证与停止条件

| 验证 | 必须观察到的结果 |
|---|---|
| 当前CPU反例 | 正常/丢响应两路径暴露text与raw IDs/恢复差异；不误报GPU错误 |
| 两阶段事务CPU合同 | 中间成功不增长turn/version；任一步失败无业务commit；published之后丢响应只回同发布物；每个stage唯一RID且不重采样 |
| 输出/完整QA合同 | 宿主ID复制来自固定plan；raw text/IDs原样保留；新history为完整publishedQA；实际模板省略内容即失败；最终发布超cap失败不截文 |
| 有限真实native合同 | 同真实输入跑A2两stage，再提交一轮published-history重建；逐token核新输入、原生receipt与实际prefix/device/host/prefill，无虚造已缓存citation token |
| 质量/成本实验 | 全159冻结输入，各阶段失败保留；先对A0/A1/A2与同plan extractive看对象/条件/出处/回答性，再做自有published历史闭环 |

若只是引用合法率升高，语义/回答性没有改善；若extractive同样有效且更便宜；或实际收益来自额外调用而非A2支持关系，则不直接接B/C或扩大状态机。若只读/CPU合同未满足，先停在隔离实验，不用更多guard、输出hash或补引用规则掩盖执行账本不一致。
