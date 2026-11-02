# R031：部署校准、实跑替代与可追溯性

研究问题：能否在不跨机器、不跨模型或并行配置的前提下摊薄部署成本，并用正式执行中测得的同口径成本提高预测准确性？训练布局（R030）和返回推理（R027）是两个独立消费者。研究指标包括冷启动校准时间、命中节省时间、实测替代前后的预测误差以及端到端墙钟；不能把校准成本或离线回放选择当作实际性能收益。

## 缓存与优先级

118 的 run_bench 在主机用户的 ~/.cache/areal-deployment-calibration/
下保存数据，仅把本次匹配的记录副本交给容器。机器身份、模型路径和文件内容哈希、实际训练/推理 backend（含
TP/DP/PP）、设备数、工作负载配置、精度、驱动、镜像、运行源码/模板和策略环境参与匹配；物理卡编号不参与。总批数不用于区分同一配置。老格式、损坏或不匹配记录不得作为本机测量。只有成本标量可以复用，模型权重、优化器、版本、请求队列、选卡序列和历史收益均不导入。

- 训练布局无记录时做版本0的部署探针，校验训练状态；完整命中后跳过测量，仍正常初始化和建立两个引擎。按布局和指标独立选择最近的有效实跑值，否则用该布局部署值。计算按
  padded token positions 归一化；驻留复用和未完成的后台转移不替代已就绪入口成本；未执行的发布不能作为零成本样本。
- 推理返回沿用已有部署预热入口，兼容记录命中时不再启动预热训练。普通 bench 可使用已有记录并从实际完成的 publish/restore
  对学习；正式测量首次到达时清除导入的部署样本，不把二者求平均。
- 只在共享组成功且报告完成后晋升本机记录。版本链保留原始部署值、每批样本及生效来源；源码或模板变化保守失效。匹配范围过宽/过窄和探针代表性是后续研究项目。

## 追查与调整

运行目录中查看 calibration-inputs.json（实际输入
key/revision）、layout-calibration-audit.json（布局、指标、批次、单位、总秒数、工作量和来源）、calibration-commits.json（本次产出
revision）。推理对记录在 return-calibration-observed.json。每次正式选卡另有
measurement_origins。修正公式应保留测量值并使旧算法缓存失效，不应手改 JSON 覆盖原始证据。

主机上用 python3 benchmark/npu_share/calibration_admin.py list 查 key；history --kind KIND
--key KEY 看版本；show 查看具体数据；invalidate --reason 原因 使下次重新测量；rollback --revision REV
--reason 原因 重新启用有证据的版本。管理操作同样追加修订；运行持锁时禁止调整。固定/colocated 独立基线不读写此缓存。尚未验证 NPU
冷/热两次正式运行的墙钟收益。

训练探针可用 adjust-probe --kind training-layout --key KEY --layout 1 --field compute_seconds
--seconds VALUE --reason 原因 修订；仅修改后备探针，不覆盖已有同口径实跑值。修订记录标为
research_adjustment，保留父版本、调整前后秒数和理由。未对现有实验数值作人工调整。
