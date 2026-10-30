# AUX-S01：多机评分辅助研究

性质：辅助研究，代码保留在当前 feature 分支，不纳入论文、不向上游 AReaL 提交。

## 当前范围

目标已由四机调整为 alice 与 108 两台 AgentEnv，每台最多 8 个评分生命周期，总上限 16。 alice 的 AgentEnv v0.2.4 已完成同一份
768 回答的 8/16 并发回放；均无错误、重试、奖励差异和删除失败。 8 并发 48.930 秒，16 并发 48.904 秒。原到达序列最后一个回答在 48.759
秒出现，不能用此结果断言极限吞吐。 108 已通过 huawei SSH 核实：运行 6.5.0-28，6.8.0-138 与 ublk_drv
已安装但尚未启动。仅准备切换/回退方案，待维护窗口批准，不得视为 AgentEnv 已部署。 118 和 200 为 Linux 5.10，缺少 ublk_drv，不满足
AgentEnv v0.2.4 的 Linux 6.8+ 要求；不改宿主内核。 CubeSandbox 不再投入排查；Kuasar 原服务不变。

## 显式接入

单服务 KUASAR_CONFIG 格式和行为不变。辅助实验把该文件换为包含 backends 的配置：

```json
{
  "ca_bundle": "/private/all-backend-ca.pem",
  "lock_dir": "/work/sandbox-pool-slots",
  "backends": [
    {"name": "alice", "machine": "alice", "provider": "agentenv", "capacity": 8,
     "access_file": "/private/alice.json", "user": "root"},
    {"name": "108", "machine": "108", "provider": "agentenv", "capacity": 8,
     "access_file": "/private/108.json", "user": "root"}
  ]
}
```

每个私有 access_file 包含 api_key、api_url、template_id；AgentEnv 的数据入口沿用 api_url。
所有文件、路径必须在实际评分进程中可读，凭据不得提交到仓库。 只允许 HTTPS 或通过 SSH 隧道访问的 loopback
HTTP；证书在初始化时使用统一信任包，不逐请求切换。 使用单一协调主机，所有 rollout 进程必须共享同一个本地 lock_dir。文件锁不是跨主机分布式锁。
独立主机或实验若使用不同锁目录，不保证共同后端的总限额；本研究不支持这种部署。 评分全局槽位必须另设为 16，后端槽位各为 8；默认全局评分配置未改变。

## 调度和失败语义

扫描所有后端空闲槽位，不把请求排在满载后端上；一次评分在同一后端创建、上传、执行和删除。 槽位覆盖整个生命周期。失败不当零奖励，不自动跨服务重试。
创建结果不明确或删除失败时槽位保留 180 秒，等待 120 秒沙箱租期加请求余量结束；进程崩溃后仍保留该记录。 后端错误、排队时间、服务时间和隔离状态写入
request_trace。正常测试非零退出才是零奖励。 模板必须包含相同 Python 和依赖，运行同一测试脚本和资源限制。 仅辅助适配器和 CPU
调度测试已实施；真实两机联合回放仍待 108 就绪。

C04 单后端对照按用户要求保持原评分并发 16，独立配置 alice 容量 16；两机池目标仍是每机 8。配置支持显式上限 1–32，默认路径不变。
