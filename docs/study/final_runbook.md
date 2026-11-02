# 最终复现与产物读取手册

所有运行命令位于既存主机 `public-ge108-wlf3-hangu-5.idchb2az3.hb2.kwaidc.com` 的 `sjy-0518` 容器。项目根目录为：

```bash
sudo -n nerdctl --namespace default exec -it sjy-0518 bash
cd /home/relay/songjiayun/ContextKV/runs/study-20261102/project
```

权重、原始数据、实际历史、SQLite、评分和连续采样原地读取；本手册不安排模型传输、容器创建或服务重启。执行环境为 `.venv-sglang/bin/python`，官方评分依赖按已存 study 配置使用。源码提交及源码 SHA、模型 scope、run config SHA 和实际引擎 fingerprint 分别登记，容器父仓库的 HEAD 不替代解包源码身份。

## 固定入口与资产

下列路径均相对上述项目根。

| 范围 | 实际产物 |
|---|---|
| R0 基座 | `/media/ssd1/qwen3-32b`；`configs/model_scopes/qwen3-32b.local.json` |
| R0 实际 revision | `local-snapshot:7442623e99dd0b456628ddc48805a4ea11249ee2ee66484b5d1ab44d2940beb8` |
| R0 成本 v3 | `models/study-plan-r0-bridge-v3-execution.json`；`models/study-plan-r0-bridge-v3-native/native-assets.json` |
| R0 当前 READ 收益 v4 | `models/study-plan-gain-r0-bridge-v4.json` |
| B 开发与冻结 | `runs/study/rolling-r0-bridge-final-v4/{dev/matrix.json,dev/results.jsonl,frozen.json}` |
| B 已存共同开发集汇总 | `runs/study/b-final-common-dev-v1` |
| B 独立预登记测试 | `runs/study/b-prespecified-test-v1/{registration.json,study.yaml,matrix.yaml}` |
| C 当前配对 v4 | `runs/study/c-current-r0-g5-v4/{paired-workflow.json,train/paired-rows.jsonl,dev/paired-rows.jsonl}` |
| C scope-only 资产 | `runs/study/c-current-r0-g5-v4/value-model-scope-only.json` |
| 真实联合负载 | `runs/study/capacity-r0-joint-v1` |
| A 公开 V4 bundle | `runs/study/a-public-final-v4/{study.yaml,matrix.yaml,cohort_manifest.json,retune_manifest.json,schedule.json}` |
| A 最终冻结 | `runs/study/a-public-final-v4/frozen.json`；胜出方法和测试方法只取该文件 |
| 训练来源与 R1/R2 | `docs/study/training_assets.json`；`models/study-reader-r1`、`models/study-reader-r2` 及各自 merged 目录、scope 和 manifest |

R0 bridge 的配置标签为 `qwen3-r0-source-first-windows-32k-nonthinking`；成本资产仍核对实际 reader、cache config、物理设备与 resident identity。当前输入/cache/concurrency 桶中的统计预测保留 `observed_input_range` 与 `prediction_bin`；`native_statistical_extrapolation` 可参与选择，但不改称 `calibrated=true`。当前 ANSWER 独立估计剩余时间及平均驻留积分，已付 head 保留在 paid 中。成本-only 容量数据不进入 QA 质量和 C value 监督。

## 冻结及协议边界

B 开发共同 6 QA 的 RB_llm：B3 为 0.8，S3 λ0 为 0.633333，S3 λ1e-5 为 0.566667；实际开发胜出方法为 **B3**。`b-prespecified-test-v1/registration.json` 独立登记 B3、B6 和两档 S3 测试对照；这些结果不回流选择 λ 或覆盖 B3 开发冻结。

C v4 有 4 个配对窗口，train/dev 各 2；scope-only 的 dev 覆盖为 2/2，MAE 为 4452.695606 ms，开发未选择正收益留存动作，native incumbent 保留。完整特征和 count-only 的 0/2 是未覆盖，不是零收益预测。 signed `Native−Retain`、首次控制和 collateral 成本从原始配对表读取。

A 的 `a_finalcell_readers` 声明落在 `study.yaml` 的 `policy_comparison.cell_readers`，冻结后也保存在 `frozen.config`。每格可为相同 reader checkpoint 声明不同服务；端口不同不构成新 reader 身份。逐格 run config 和服务映射在冻结前确定，冻结保存其 SHA。并行测试使用这些预先声明的独立服务；不能在冻结后通过改端口或改模型文件替换某格。

V4 `provided_full_direct_when_fits` 只适用于单轮 supplied-originals 请求：保留全部原文、Unicode 空白和 distractors，先验证同一 8192 body 预算、passage 限制及真实 32K 模型窗口（ANSWER 1024）下每份原文完整出现在候选中。都满足时直接 ANSWER、READ=0，P0/P1 选择头被绕过，属于同完整输入的 reader 对照；超预算继续既有 READ 规划路径，不裁短原文来强行取得 direct 资格。实际选择与 token 数见 `provided_full_direct` trace。

2Wiki `final_line` 是预先声明的格式投影：恰好一个终端 `Final answer: ...` 行、已知未截断、字段非空；只去除字段内 E_ 引用标记。缺行、多行、非终端或截断均记录失败，不做语义抽取或按 gold 修补。每格 `scoring/2wiki/` 同时保留 `answer_projection.jsonl`、`raw_full_answer.scores.jsonl` 和 `manifest.json`。原始全文 EM/F1 是辅助结果，不能替代投影主指标；字段失败分母、原始答案、SP/evidence/joint 和费用分别保留。

以下只读查看既有声明及源码身份，不运行新协议测试或模型调用：

```bash
.venv-sglang/bin/python - <<'PY'
import json
from pathlib import Path
import yaml
from bench.study_checkpoint import source_inventory
print('source', source_inventory(Path.cwd())['sha256'])
for name in ('runs/study/b-prespecified-test-v1/registration.json',
             'runs/study/a-public-final-v4/retune_manifest.json'):
    print(name, json.loads(Path(name).read_text()))
cfg = yaml.safe_load(Path('runs/study/a-public-final-v4/study.yaml').read_text())
print('declared_cell_readers', cfg['policy_comparison'].get('cell_readers', {}))
PY
```

## A：按已冻结方法并行测试，再只读汇总

由 root 在 `frozen.json` 已生成且逐格独立服务已声明后执行。每个 worker 的 `--methods` 只允许冻结测试方法子集；同一方法不重复启动。

```bash
task_a_root=runs/study/a-public-final-v4
task_a_pids=()
while IFS= read -r task_a_method; do
  .venv-sglang/bin/python -m bench.a_final_batch run \
    --output "$task_a_root" --split test --methods "$task_a_method" \
    > "$task_a_root/test-${task_a_method}.worker.log" 2>&1 &
  task_a_pids+=("$!")
done < <(.venv-sglang/bin/python -c \
  "import json; print('\n'.join(json.load(open('runs/study/a-public-final-v4/frozen.json'))['test_methods']))")
task_a_failures=0
for task_a_pid in "${task_a_pids[@]}"; do
  wait "$task_a_pid" || task_a_failures=$((task_a_failures+1))
done
.venv-sglang/bin/python -m bench.a_final_batch collect --output "$task_a_root" --split test
.venv-sglang/bin/python -m bench.a_final_batch report --output "$task_a_root"
```

`collect --split dev|test` 只从现有每格评分产物重建聚合文件和来源 manifest；不生成或重试答案。失败 worker 的原始日志、输出、费用和缺失题仍保留。

## B：共同 QA 与真实预测误差

以下输出目录须是新目录；方法 ID 是预登记的固定 ID，不从测试分数挑选。

```bash
.venv-sglang/bin/python -m bench.plan_result_summary \
  --matrix runs/study/b-prespecified-test-v1/matrix.yaml \
  --root runs/study/b-prespecified-test-v1 \
  --methods B3 B6 S3-lambda0 S3-lambda1e-05 --split test \
  --output runs/study/repro-final/b-common-test

.venv-sglang/bin/python -m bench.plan_prediction_audit \
  --root runs/study/b-prespecified-test-v1 \
  --output runs/study/repro-final/b-prediction-audit
```

共同 task IDs、同问题、同实际 QA 历史三个 cohort 分开；历史只比 turn ID 与真实文本。步骤数及在线 token 取真实 response；成本缺项保持 null。预测审计读取 COMPLETE StepRecord、实际阶段窗口和同范围物理采样，扣除 paid 与未来 tail 后再比较当前步，导出 count/MAE/bias。共同子集不冒称完整公开 benchmark；独立校准观测重叠和未匹配原因保留。

## 总成本并集与最终只读索引

以下显式 glob 只读取已存费用与采样；输出登记未匹配文件、未知时间洞及覆盖范围。原始 job 积分、逐请求份额和连续采样重叠不能再次相加。全程 observer 的实际路径可另作为 `--manifest`、`--physical` 追加；不据缺采样区间补造费用。

```bash
.venv-sglang/bin/python -m profiling.study_cost_ledger \
  --manifest 'runs/study/plan-gain-r0*/**/capture-entry.json' \
  --manifest 'runs/study/c-current-r0-g5-v4/**/capture-entry.json' \
  --manifest 'runs/study/rolling-r0-bridge-final-v4/workflow.json' \
  --manifest 'models/study-reader-r*/training_cost*.json' \
  --physical 'runs/study/b-prespecified-test-v1/**/physical_resources.json' \
  --physical 'runs/study/a-public-final-v4/**/physical_resources.json' \
  --physical 'runs/study/c-current-r0-g5-v4/**/*resource*.json' \
  --physical 'models/study-reader-r*/*physical_resources*.json' \
  --gpu-devices 0,1,2,3,4,5,6,7 \
  --output runs/study/repro-final/full-cost-ledger.json

.venv-sglang/bin/python -m profiling.final_study_index \
  --artifact 'B-cost=models/study-plan-r0-bridge-v3-execution.json' \
  --artifact 'B-native=models/study-plan-r0-bridge-v3-native/native-assets.json' \
  --artifact 'B-gain=models/study-plan-gain-r0-bridge-v4.json' \
  --artifact 'B-freeze=runs/study/rolling-r0-bridge-final-v4/frozen.json' \
  --artifact 'B-registration=runs/study/b-prespecified-test-v1/registration.json' \
  --artifact 'A-retune=runs/study/a-public-final-v4/retune_manifest.json' \
  --artifact 'A-freeze=runs/study/a-public-final-v4/frozen.json' \
  --artifact 'A-projection=runs/study/a-public-final-v4/**/scoring/2wiki/manifest.json' \
  --artifact 'capacity=runs/study/capacity-r0-joint-v1/*.json' \
  --records 'Bdev=runs/study/rolling-r0-bridge-final-v4/dev/results.jsonl' \
  --records 'Btest=runs/study/b-prespecified-test-v1/*/*/*/results.scored.jsonl' \
  --records 'Adev=runs/study/a-public-final-v4/dev/results.scored.jsonl' \
  --records 'Atest=runs/study/a-public-final-v4/test/results.scored.jsonl' \
  --baseline Bdev=B3 --baseline Btest=B3 --baseline Adev=P0_R0 --baseline Atest=P0_R0 \
  --paired 'Cv4=runs/study/c-current-r0-g5-v4/*/paired-rows.jsonl' \
  --retention-model 'scope=runs/study/c-current-r0-g5-v4/value-model-scope-only.json' \
  --cost-ledger runs/study/repro-final/full-cost-ledger.json \
  --output runs/study/repro-final/index
```

索引只使用已有数值指标与 caller 标注的 cohort；不同版本、公开轨道、构造诊断、容量数据保持独立。尚无最终评分、冻结或完整物理窗口的项目保持 missing/null 或列入未匹配清单，不能由训练 loss、字段投影成功率、成本预测或另一轨道分数代填。原始产物及权重继续保留在指定容器。
