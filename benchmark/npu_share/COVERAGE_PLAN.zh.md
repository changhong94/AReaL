# 代表性 benchmark 测试覆盖方案

[English version](COVERAGE_PLAN.md)

这是测试覆盖计划，不是已通过测试的记录。每个用例本身就执行一次**固定组与共享组对照**，无需再手动补跑一对。各用例串行运行，单项失败后继续执行其余项；全部尝试结束后，只要有失败项，脚本就返回非零退出码。

本计划选取 28 个典型配置，不穷举所有参数组合。建议先运行四个 core 用例，再选择与改动相关的套件。这是按风险选择的覆盖，**不是数学上完备的两两组合覆盖数组**。

## 执行方法

在已配置 benchmark 的主机上，从任意目录定义 `BENCH_DIR`。可先用 `plan` 查看命令，再直接执行 `run`。无需先执行 `prepare`，因为
`run` 自带准备步骤。`plan` 不准备数据、不创建目录或容器，也不使用 NPU。命令及代码注释与英文版保持一致。

```bash
BENCH_DIR="$(python3 - <<'PY'
import json
from pathlib import Path
p = json.loads((Path.home() / '.config/areal-benchmark/profile.json').read_text())
print(Path(p['repo']) / 'benchmark/npu_share')
PY
)"

# Inspect commands only: no data preparation, containers or NPU use.
bash "$BENCH_DIR/coverage_bench.sh" --suite core --mode plan
```

正式测试可直接执行下面的命令，它自身包含准备步骤，无需先运行 `prepare`：

```bash
bash "$BENCH_DIR/coverage_bench.sh" --suite core --mode run
```

其他选择如下，沿用第一个代码块定义的 `BENCH_DIR`：

```bash
# Investigate inference load; compare C05 against C01 and C06 against C02.
bash "$BENCH_DIR/coverage_bench.sh" --suite load --mode run

# Single-batch terminal path and off-policy boundaries.
bash "$BENCH_DIR/coverage_bench.sh" --suite boundary --mode run

# 4/8-card instance routing and TP-group readiness; ensure these devices are available.
bash "$BENCH_DIR/coverage_bench.sh" --suite topology --mode run

# Unequal inference:training ratios, including TP2 and Megatron.
bash "$BENCH_DIR/coverage_bench.sh" --suite ratio --mode run

# A focused pair, rather than every topology.
bash "$BENCH_DIR/coverage_bench.sh" --cases 'T01 T02' --mode run

# Workload size and answer-group coverage (six selected combinations).
bash "$BENCH_DIR/coverage_bench.sh" --suite workload --mode run

# Long-run stability; only after shorter runs pass.
bash "$BENCH_DIR/coverage_bench.sh" --suite long --mode run

# Inspect the complete representative matrix without creating output directories.
bash "$BENCH_DIR/coverage_bench.sh" --suite all --mode plan
```

`prepare` 是独立、可选的预检，不是 `run` 的前置步骤。每次执行 `prepare` 或 `run` 都会创建一个新的时间戳目录；`run`
**不会接续或复用此前的 prepare 目录**，而是冻结当前代码并重新准备输入。因此先后执行两者会生成两个目录，但只有 `run` 会训练。`plan`
不创建目录。不需要单独预检时，直接运行 `run`，即可避免重复准备及额外目录。

用 `--profile /path/to/profile.json` 指定其他主机配置。支持 `all --mode run`，但它不是默认项，因为会涉及代码沙箱、8
卡实验及 100 批长实验。

## 统一测试矩阵

`Offpolicy` 对应 `--max-head-offpolicyness`，`TP` 表示每个 vLLM 实例使用的张量并行卡数。默认每批 32 题、每题 8
个回答，每个用例运行一个 epoch。

workload 套件包含 C01 和 W01–W05，覆盖每批 16／32／64 题及每题 4／8／16 个回答。单独运行 core 时保持 32/8；使用
`--suite workload` 或 `--suite all` 才包含这些变化。

两组接收相同的命令行参数。除 G01/G02 使用 eager 外，所有用例均使用推理图模式；训练执行模式不变。套件名和枚举值保留英文，便于与脚本对应。

| ID  | 套件      | RL   | 训练引擎 | 推理模式 | 题目并发数 | 批次数 | NPU 数 | 推理:训练 |  TP | Offpolicy | 每批题数 | 每题回答数 | 主要目的                                            |
| --- | --------- | ---- | -------- | -------- | ---------: | -----: | -----: | --------- | --: | --------: | -------: | ---------: | --------------------------------------------------- |
| C01 | core      | math | fsdp     | graph    |         16 |      6 |      2 | 1:1       |   1 |         2 |       32 |          8 | 默认基准；权重发布与最后一批清理                    |
| C02 | core      | math | megatron | graph    |         16 |      6 |      2 | 1:1       |   1 |         2 |       32 |          8 | 与 C01 比较训练后端                                 |
| C03 | core      | code | fsdp     | graph    |         16 |      3 |      2 | 1:1       |   1 |         2 |       32 |          8 | E2B/Kuasar 执行及奖励计算路径                       |
| C04 | core      | code | megatron | graph    |         16 |      3 |      2 | 1:1       |   1 |         2 |       32 |          8 | 代码工作流与另一种训练后端                          |
| C05 | load      | math | fsdp     | graph    |         32 |      6 |      2 | 1:1       |   1 |         2 |       32 |          8 | 相对 C01 增大并发                                   |
| C06 | load      | math | megatron | graph    |         32 |      6 |      2 | 1:1       |   1 |         2 |       32 |          8 | 相对 C02 增大并发                                   |
| E00 | boundary  | math | fsdp     | graph    |         16 |      6 |      2 | 1:1       |   1 |         0 |       32 |          8 | 同步准入；权重发布后才允许下一批执行                |
| E04 | boundary  | math | megatron | graph    |         16 |      3 |      2 | 1:1       |   1 |         0 |       32 |          8 | 另一训练后端上的零前瞻权重发布                      |
| E01 | boundary  | math | fsdp     | graph    |          1 |      1 |      2 | 1:1       |   1 |         1 |       32 |          8 | 最小正并发与前瞻值；不经过普通中间批次发布          |
| E02 | boundary  | math | fsdp     | graph    |         16 |      6 |      2 | 1:1       |   1 |         1 |       32 |          8 | 与 E00、C01 比较最小正前瞻范围                      |
| E03 | boundary  | math | fsdp     | graph    |         16 |      6 |      2 | 1:1       |   1 |         3 |       32 |          8 | 相对 C01 扩大前瞻；观察过时及未使用工作             |
| T01 | topology  | math | fsdp     | graph    |         16 |      3 |      4 | 1:1       |   1 |         2 |       32 |          8 | 多个 TP1 推理实例与两个训练 rank                    |
| T02 | topology  | math | fsdp     | graph    |         16 |      3 |      4 | 1:1       |   2 |         2 |       32 |          8 | 相同设备和工作量下，与 T01 比较 TP2                 |
| T03 | topology  | math | megatron | graph    |         32 |      3 |      8 | 1:1       |   1 |         2 |       32 |          8 | 四个训练 rank；四个专用及四个借用的 TP1 实例        |
| T04 | topology  | math | megatron | graph    |         32 |      3 |      8 | 1:1       |   2 |         2 |       32 |          8 | 两个专用及两个借用的 TP2 实例；与 T03 比较          |
| R01 | ratio     | math | fsdp     | graph    |         16 |      3 |      3 | 2:1       |   1 |         2 |       32 |          8 | 2:1 分配；两个专用及一个借用的 TP1 实例             |
| R02 | ratio     | math | fsdp     | graph    |         16 |      3 |      4 | 3:1       |   1 |         2 |       32 |          8 | 3:1 分配；三个专用及一个借用的 TP1 实例             |
| R03 | ratio     | math | megatron | graph    |         16 |      3 |      6 | 2:1       |   2 |         2 |       32 |          8 | 2:1 分配；两个专用及一个借用的 TP2 实例             |
| R04 | ratio     | math | megatron | graph    |         16 |      3 |      8 | 3:1       |   2 |         2 |       32 |          8 | 3:1 分配；三个专用及一个借用的 TP2 实例             |
| G01 | execution | math | fsdp     | eager    |         16 |      6 |      2 | 1:1       |   1 |         2 |       32 |          8 | 与图模式 C01 配对的 eager 对照                      |
| G02 | execution | math | megatron | eager    |         16 |      6 |      2 | 1:1       |   1 |         2 |       32 |          8 | 与图模式 C02 配对的 eager 对照                      |
| W01 | workload  | math | fsdp     | graph    |         16 |      6 |      2 | 1:1       |   1 |         2 |       16 |          8 | 相对 C01 减少每批题数                               |
| W02 | workload  | math | fsdp     | graph    |         16 |      6 |      2 | 1:1       |   1 |         2 |       64 |          8 | 相对 C01 增加每批题数                               |
| W03 | workload  | math | fsdp     | graph    |         16 |      6 |      2 | 1:1       |   1 |         2 |       32 |          4 | 相对 C01 减少每题回答数                             |
| W04 | workload  | math | fsdp     | graph    |         16 |      6 |      2 | 1:1       |   1 |         2 |       32 |         16 | 相对 C01 增加每题回答数                             |
| W05 | workload  | math | fsdp     | graph    |         16 |      6 |      2 | 1:1       |   1 |         2 |       64 |         16 | 较大题目批次与较大回答组的组合高负载                |
| L01 | long      | math | fsdp     | graph    |         16 |    100 |      2 | 1:1       |   1 |         2 |       32 |          8 | 3,200 道不同题目；反复发布、清理及资源增长          |
| L02 | long      | math | megatron | graph    |         32 |    100 |      2 | 1:1       |   1 |         2 |       32 |          8 | 较高并发下 Megatron 长时间稳定性；推理:训练保持 1:1 |

E01 故意组合多个边界，以较低成本发现最后一批处理路径的问题，不是单变量性能消融。core/load/off-policy 配对以及 T01/T02、T03/T04
保持其他命令行设置不变，但生成的回答仍可能不同。

L01、L02 均使用两张 NPU，推理:训练为 `1:1`。L02 同时改变训练后端和并发，是补充性稳定性覆盖，不是单变量消融。每组在 100 批内处理 3,200
题、25,600 个回答。

R01–R04 覆盖不等比例分配，其余用例使用 `--inference-training-ratio 1:1`。ratio 用例均为算术
RL，只有一个分布式训练组。固定模式只使用专用推理卡，共享模式还可借用训练卡。两组卡数都必须能被 TP 整除，训练组不会动态缩放。R01/R03、R02/R04
分别覆盖相同实例数、不同 TP 和训练后端，属于功能覆盖。

所有用例都为共享组显式设置 `--adaptive-concurrency capacity`。固定组保留请求的基线并发。R025 按就绪实例相对于专用实例的数量缩放：2:1
配置最多可达基线的 1.5 倍，3:1 最多可达 4/3 倍，仍受就绪状态、队列及 off-policy 限制。这些是计划，不能当作硬件性能已经验证的声明。

## 各套件的验收内容

- \*\*Core：\*\*两组完成；题目与回答数量符合预期，输入哈希一致；检查点、摘要及四类图可用。阶段图同时有中英文版，其他图保持英文。
- \*\*Code：\*\*实际沙箱创建、执行、删除预检及奖励计算成功；全局奖励并发一致。仅准备配置不能验证服务、认证及代码执行。基础设施失败不能作为零奖励训练样本接受。
- \*\*Load/off-policy：\*\*检查循环时间、token 数、吞吐、每批收齐情况、保留或迁移的工作及显存压力。R 标签中的 `saved`
  是固定组区间耗时减去共享组区间耗时，不是累计领先时间。
- \*\*Topology：\*\*确认设备/rank 映射、各 rank 预算、集合通信就绪及发布成功。**TP>1
  不等于覆盖了显存压力回退**；修改卸载/映射代码时，应另做可控的非对称压力测试，不要在优化器更新中人为制造 OOM。
- \*\*Ratio：\*\*核对专用/借用 rank 集合、训练设备归属、向每个专用实例发布权重，以及不等比例下 R025 的目标并发。
- \*\*Long：\*\*完成全部 100 次更新；观察
  RAM/HBM、文件描述符、后台任务及发布延迟是否随步数增长。默认监测并不覆盖所有资源，判断泄漏需要额外测量，不能只看成功退出。

每组题数为 `batches × batch_size`，回答数为 `batches × batch_size × answers`。默认 32/8 时：3 批为 96
题／768 回答；6 批为 192／1,536；1 批为 32／256；100 批为 3,200／25,600。W01–W05
按统一矩阵中的参数计算。分词及数据准备需确认有足够的合格、不重复题目。多个回答是分别采样的结果，不是复制参考答案。

功能通过不以固定加速百分比为门槛。变慢是待调查的结果，不自动等于实现错误。要提出性能结论，针对选定对照至少重复三次，报告所有轮次及 token
数，并区分模型质量和系统吞吐。当前脚本固定种子且总是先固定组、后共享组；重复不提供种子多样性，也不消除顺序偏差。隔离计算成本需要使用相同输入回放。

## 覆盖缺口与可选后续测试

- 八卡 TP4 是可选项，此前缺少端到端验证。先准备，再做**单批**检查，之后再考虑长实验：

```bash
  python3 "$BENCH_DIR/run_bench.py" --rl math --engine megatron \
    --npu-count 8 --inference-tp 4 --concurrency 32 --batches 1 --prepare-only
  # Remove --prepare-only only when explicitly ready to test this topology.
```

- 本方案未穷举 code × 所有拓扑 × 所有 off-policy 值。修改多实例沙箱路由时，需将 T02 改为 `--rl code` 再测，才能声称支持该组合。
- 并发 64 以上、更大模型、不同上下文长度、多节点、分布式故障恢复及训练质量评估需要单独测试。
- 非法参数（如并发 0、offpolicy -1、1:1 比例配 3 卡、总共两卡却配置 TP2）应在 CPU 参数/配置检查中覆盖，而非硬件实验。
- 100 批成功不证明可以无限稳定运行，应如实报告验证范围。

## 结果与失败处理

脚本在 profile 的持久化 `workspace_root` 下创建 `coverage-<日期时间与时区偏移>/`，不放入代码目录。每个子实验生成独立时间戳目录，包含
`settings.json`、`summary.json`、`summary.md`、`summary.zh.md`、`index.html`、日志及图片。`C01.log`
等日志中有 `WORKSPACE=` 行；`cases.tsv` 记录选定参数及退出码。

准备模式只生成数据、配置及 `prepared`
状态，不生成性能图或实测加速比。任一用例失败时保留日志并继续其余项。全部尝试结束后，有失败项则返回退出码1，否则返回0；检查对应 `state.json`
和日志，修复后可只重跑失败 ID。脚本不会自动重试，也不会自动从检查点恢复。

`Offpolicy=0` 是受支持的同步准入场景，不是非法参数测试。版本 v 时，累计最多提交/接收 `(v + 1)`
批；仅完成当前批不足以放行下一批，必须等发布版本推进。两张卡仍可在训练阶段之间共同推理。PPO 期间不重新开启推理，因为没有未来批次可执行。E00、E04 分别覆盖
FSDP、Megatron 的此边界。

## 已记录的零前瞻冒烟验证

2026-10-15 的 FSDP 双卡实验 `20261015-224950-0700_math_fsdp_c16_b2_1f7359` 完成固定与共享两组，各两批、每批
16 题、每题 6 个回答。每组完成 192 个回答：第一批全部 96 个片段使用版本 0，第二批全部 96 个片段使用版本
1。轨迹审计未发现版本违规。这只验证两批边界路径，不等于完整 E00 六批或 Megatron E04 已通过。完整日志及
`offpolicy-zero-audit.json` 保留在 `/root/areal-benchmarks/` 下上述运行目录中。

## 推理执行模式

默认 `--inference-mode graph` 选择 vLLM Ascend 的 `FULL`，覆盖 prefill、混合批次及
decode；训练执行不变。捕获桶包含调度器 token 上限（默认 2048），不仅是小型 decode 批次。worker 对每个阶段/模式记录一次
`AREAL_GRAPH_PREFILL_MIXED` 和 `AREAL_GRAPH_DECODE`，标明实际调度模式；eager 回退显示为 `mode=NONE`。

`--inference-mode eager` 使用原执行路径。固定、共享接收相同模式，并在 settings 的 `inference_mode` 中保存。现有
workload/topology/ratio/long 用例默认使用 graph；`--suite execution` 选择 C01/C02 图模式与 G01/G02
eager 对照。`coverage_bench.sh --inference-mode eager` 会覆盖所有选中用例的模式。

共享图模式支持范围为现有 Qwen2 BF16 全注意力模型及 uni/mp 的相同 TP 实例。稀疏 KV
保持虚拟地址稳定；图内存池常驻，不属于可回收存储。发布权重时，已捕获的权重/缓冲区存储不能更换。不支持的图模式或存储替换会明确失败，不能静默重放失效地址。

## 批量大小与回答组覆盖

`--suite workload` 比较 C01 与五个附加用例。均采用算术 RL、FSDP、推理图模式、并发 16、6 批、双 NPU、TP1、off-policy
2、推理:训练 1:1。四个用例分别改变一个工作量因素，W05 检查组合高负载；不是九种组合全遍历。

所有 workload 用例已列于统一矩阵。每批回答数为 `每批题数 × 每题回答数`，每次运行题数为 `批次数 × 每批题数`。命令显式传入两个参数，`cases.tsv`
同时记录。它们比较系统性能，不证明哪种工作量形状能取得最佳 RL 学习效果。历史结果仍按原有参数解释，不套用新默认值。

共享组默认开启 R027
返回推理收益判断及经过验证的常驻缓存预算（`AREAL_PROFIT_AWARE_RETURN=1`、`AREAL_RESIDENT_BUDGET=1`），固定组显式关闭两项。Coverage
继承这些默认值。缓存抵扣保留显存安全及拓扑检查，默认池抵扣仅限单 rank FSDP。

### Coverage 总览页

带时间戳的 coverage 总目录会生成英文 `index.html`、中文 `index.zh.html`、`summary.md`、`summary.zh.md`
和机器可读的 `summary.json`。每项运行前后、正常退出或可捕获的中断时均更新。总览列出所有所选用例的参数、状态、固定/共享训练循环耗时、节省比例、token
数以及单项图表、汇总、状态和日志链接。循环耗时不含初始化；缺失数据不填零，失败项不计算性能收益。SIGKILL 等强制终止无法执行退出汇总。

已有目录也可单独补生成汇总，不运行实验：

```bash
python3 "$BENCH_DIR/coverage_summary.py" /path/to/coverage-directory --final
```

旧目录若没有 `planned-cases.tsv`，只能恢复有记录的用例，不会猜测缺失的计划项。查看仍在运行的 suite 时，不要加 `--final`。

### 暂停、恢复与手动重跑

先找到正在运行的 **Bash coverage_bench.sh 的 PID**，不要填 driver、run_bench.py 或训练进程的 PID。

```bash
python3 "$BENCH_DIR/coverage_control.py" pause --pid <coverage-bash-pid>
python3 "$BENCH_DIR/coverage_control.py" status --pid <coverage-bash-pid>
# Wait until paused before manually rerunning selected cases.
python3 "$BENCH_DIR/coverage_control.py" rerun --pid <coverage-bash-pid> --cases 'T01 T02'
python3 "$BENCH_DIR/coverage_control.py" resume --pid <coverage-bash-pid>
```

`pause_pending` 表示当前对照仍在正常运行、清理资源；`paused` 表示主进程已暂停且没有
活动的直属子进程。重用卡之前仍须检查容器是否清理完成，尤其是失败项。当前项的 TSV 记录及套件汇总可能要到恢复后才更新。仅暂停启动后续用例的 Bash，不暂停整个进程组。
这是保留远端进程的暂停，不支持远端重启恢复；笔记本断开不影响。`pause --wait` 可等待当前子任务结束。

手动重跑保留原结果，新建带时间戳的 coverage 目录，采用当前代码和用例矩阵、原启动 命令的 profile 及 inference-mode
覆盖值。不覆盖原失败记录，不改变暂停套件的队列。 如果矩阵曾修改，应核对旧 planned-cases.tsv。没有存活的暂停套件时，可独立运行：

```bash
bash "$BENCH_DIR/coverage_bench.sh" --mode run --cases 'T01 T02' --profile /path/to/profile.json
```

### 随机项目顺序

使用 `--shuffle` 随机排列所选用例；`--shuffle-seed 42` 指定种子并启用随机排列，可复现同一顺序。默认仍使用原顺序。种子和顺序打印到日志并保存到
`case-order.json`，`planned-cases.tsv` 按实际顺序排列。暂停后恢复原进程不会重新打乱。单项失败仍继续后续项。

```bash
bash "$BENCH_DIR/coverage_bench.sh" --suite all --mode run --shuffle
bash "$BENCH_DIR/coverage_bench.sh" --suite all --mode plan --shuffle-seed 42
```
