# 固定、Colocated、共享三组比较

`run_bench.py` 默认请求 fixed、colocated、latest 三组；latest 继续表示共享，兼容旧产物。`--arms two`
显式保留旧两组模式。部署校准包装器仍为两组专项，不会偷偷扩大为三组。

Colocated 不支持或失败时，记录 `arm-status.json` 的状态和原因，继续固定与共享。报告只计算真实完成组的耗时，不填造 colocated
数据。若失败容器无法停止则停止后续占卡工作，避免资源冲突；不停止无关容器。

社区来源：areal-project/AReaL 的 ascend 分支
`examples/experimental/awex/gsm8k_grpo_awex_npu_colocate_sample.yaml`。基线使用全卡 Megatron
与全卡 vLLM 交替运行、Ray、AWEX、HCCL、CPU IPC，`npu_share=False`。输入、offpolicy、评分策略保持相同，不偷偷使用示例的
offpolicy=0。三组使用同一个镜像。Colocated 关闭本研究的借卡、空闲训练引擎和提前恢复开关。

默认 colocated 训练 TP=1，可用 `--colocated-actor-tp` 明确设置。DP 数必须整除每批回答数；六卡 DP6
与256回答不整除时记录不支持，不擅自填充、丢数据或改变并行度。FSDP 的 AWEX 路径尚未验证，当前只运行固定和共享并记录原因。

## 环境

官方 AWEX 仓库 `inclusionAI/Awex` 的 `main-ascend` 分支，验证锁定提交
`95265f85fede81ce8717bf715d57c2ad856e8171`。已有镜像缺少 AWEX，使用独立派生镜像，原镜像不修改。源码打包配置只包含顶层包，普通
wheel 安装缺少 reader/meta 等子包，因此使用源码 editable 安装；不改上游源码。

`colocated-image/Dockerfile` 提供可复现配方。将指定提交的源码放入构建上下文 `awex-source/`，显式提供
`BASE_IMAGE`、`BENCH_PYTHON`、`AWEX_COMMIT`。不升级其他依赖。镜像中必须能导入 `awex.meta.meta_server` 和
`awex.vllm_plugin`，不能只检查顶层包存在。

## 输出与验证

四类图均按实际完成组绘制。phase 图列出全部训练卡及 actor/rollout offload/onload 阶段。summary 列出各组 RL
秒数、共享相对固定、colocated 相对固定、共享相对 colocated。历史两组报告仍可读取。

CPU 测试包含三组报告、历史两组兼容、colocated 失败后继续共享、配置不修改原输入、非法布局拒绝。图表 fixture
使用历史日志验证布局，明确标注非真实性能结果。实际三组 smoke 结果需单独核验，不能把预检或 fixture 当作真实完成。

## 单独验证 colocated

使用 `run_bench.py --arms colocated` 可以只执行
colocated；不会启动固定、共享，也不会生成缺失参照组的性能比较。资源前后钩子仍正常执行。成功退出仅表示运行完成，仍需核对批次和权重更新日志。

当前 vLLM 将 `/sleep`、`/wake_up` 注册在开发路由中，因此 colocated 容器必须设置
`VLLM_SERVER_DEV_MODE=1`。未设置时，即使初始化成功，首次卸载仍会返回 HTTP 404。该环境变量只由 colocated
分支设置，不改变固定和共享分支。

同卡训练进程和推理进程均创建 HCCL 通信组，colocated 环境同时设置 `HCCL_NPU_SOCKET_PORT_RANGE=auto`，避免设备侧默认端口
16666 冲突。首轮仅修复 HTTP 路由后已确认卸载和恢复成功，但权重发布首次 HCCL barrier
因端口冲突失败；后续重复通信组报错是重试结果。该失败记录保留，不作为有效性能结果。

colocated 配置显式启用 `vllm.enable_sleep_mode=True`，并合并
`additional_config.weight_nz_mode=0`。前者让权重与 KV 使用可卸载内存池；后者遵守 vLLM-Ascend wake_up 的 RL
权重格式检查，避免 FRACTAL_NZ 恢复精度问题。与历史固定/共享结果比较前应明确记录格式差异，当前仅做 colocated 功能验证。

## 独立源码基线（2026-10-31 修订）

此前以研究分支关闭开关运行的 fixed/colocated 仅为接入调试，不能作为独立社区基线。

118、200 均设置 `/mnt/nvme2/areal-development/AReaL-ascend-baseline`，来源
`areal-project/AReaL:ascend`，commit `b1f7893a9026fde67de822871f3e6102f5d10837`，保持
clean。固定与 colocated 可共用此社区源码，通过原生配置区分；共享使用研究库。profile 的 `baseline_repo` 指向该目录，也可用
`arm_repos.fixed`、`arm_repos.colocated` 分别指定独立目录，绝不退回研究库。

bench 为每组冻结 `source-<arm>`，记录 commit、分支、文件哈希和镜像 ID；基线入口 `baseline_entry.py`
不安装请求租约、共享校准、KV slab 或共享发布补丁。入口验证并记录实际导入的 `areal`、trainer 路径。配置在对应社区源码中再次校验，剔除字段记录在
`baseline-config-audit.json`。

社区详细 trace 尚未适配时，报告仅使用真实 train.begin/end，不伪造微阶段、请求或 token 图。需要后续补齐独立社区 trace
的读取器。代码任务评分适配器属于公共工作负载组件，使用相同版本，不应改写训练或推理调度。

## 独立基线镜像兼容层

社区源码保持 clean，固定/colocated 使用独立镜像。镜像增加社区日志所需 trackio，配套固定 huggingface-hub 1.12.0、datasets
4.8.5、fsspec 2026.2.0 和 pandas 2.3.3；torch、torch_npu、vLLM、CANN 不替换。构建保存安装前后包清单。

当前 vLLM 的启动模块仍位于 openai 路径，而社区 AReaL 使用 launchers/serve
路径。`colocated-image/vllm_module_aliases.py`
只注册三个模块路径别名，保持原模块及函数对象相同；不引入研究训推调度实现。镜像移除原有两项多模态 HTTP 扩展补丁，保留已编译 Ascend 算子构建适配。构建配方为
`Dockerfile.community`，wheel 文件需按 `wheels.sha256.json` 校验。

数据使用社区支持的 scheduling_spec=null 本地加载模式，与共享使用相同的冻结题目；预检比较全部训练题输入 token 的 SHA256。code
基线通过外部 CPU 评分适配器调用同一评分代理，沿用上游生成/训练逻辑，不调用研究版生成完成通知。

新 coverage 仅选择计划中的19项 Megatron 配置。旧41项套件保持暂停；修复后的失败项使用同一新 coverage 的 --output
追加尝试，不覆盖原始失败。
