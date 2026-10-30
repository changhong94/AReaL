# AUX-S01：多机评分辅助研究

性质：辅助研究，代码保留在当前 feature 分支，不纳入论文、不向上游 AReaL 提交。

## 当前范围

目标已由四机调整为 alice 与 108 两台 AgentEnv，每台最多 8 个评分生命周期，总上限 16。 alice 的 AgentEnv v0.2.4 已完成同一份
768 回答的 8/16 并发回放；均无错误、重试、奖励差异和删除失败。 8 并发 48.930 秒，16 并发 48.904 秒。原到达序列最后一个回答在 48.759
秒出现，不能用此结果断言极限吞吐。 108 已启动 6.8.0-138 内核并部署 AgentEnv v0.2.4。相同镜像摘要、E2B 2.25.1 和同一份 768
回答回放：8 并发 48.882 秒，16 并发 48.894 秒；均无错误、清理失败、奖励差异或重试。P95 分别 0.418、0.690
秒。结果受原始到达节奏限制，不代表饱和吞吐。 118 和 200 为 Linux 5.10，缺少 ublk_drv，不满足 AgentEnv v0.2.4 的 Linux
6.8+ 要求；不改宿主内核。 CubeSandbox 不再投入排查；Kuasar 原服务不变。

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
调度测试已实施；真实两机联合回放尚未执行。

C04 单后端对照按用户要求保持原评分并发 16，独立配置 alice 容量 16；两机池目标仍是每机 8。配置支持显式上限 1–32，默认路径不变。

## 2026-10-30 开发验证

200 上补充 create/upload/execute/delete 分阶段计时；删除异常单独记录
sandbox_pool_error(stage=delete)，仍传播异常并隔离槽位。新增基础设施失败不重试、清理异常隔离及成功计时测试。

108 回放目录：`/home/huawei/areal-agentenv-preparation/replay-c8` 和
`replay-c16`。两机联合真实负载尚未运行，避免额外占用当前118
coverage正在使用的alice服务。联合接入仍需为200配置108的独立安全隧道与私有access文件；代码不内置地址或凭据。

200 CPU 单元测试：`test_sandbox_pool.py` 6 项通过（0.16
秒），涵盖满载后端绕行、跨进程限额、隔离槽位、成功计时及创建/删除失败传播。未提交或推送本轮改动；118代码与coverage配置未改。

## 评分代理（2026-10-30，私有辅助研究）

本节继续属于 AUX-S01：只用于私有研究分支，不向上游 AReaL 提交、不进入论文。 现有直连模式不变。新增 `agentenv-worker` 模式：200 一次
POST 提交脚本、超时和任务 ID， 沙箱侧代理本机调用 AgentEnv 完成创建、上传、执行、删除，清理完成才返回评分。

依据：从 200 到 alice/108 的 RTT 约 350ms；本机健康检查约 0.4ms，200 复用 HTTP 连接仍约
350ms。四次串行远程调用带来显著网络等待。代理仅减少远程往返，不改变评分 脚本、CPU/内存/超时限制或沙箱隔离。性能收益尚待真实同输入对照验证。

### 协议与错误语义

- `POST /v1/score`：JSON `task_id`、`script`、`timeout`（1–10 秒）。
- `GET /v1/tasks/{task_id}`：查询已接受任务，断线后不要换 ID 或跨后端重跑。
- 请求必须携带私有 Bearer token。服务仅监听 loopback，由独立 SSH 隧道转发。
- 同 ID 同输入只执行一次；同 ID 不同输入返回 409。满载返回 429，未接受新任务。
- SQLite 在执行前记录身份，完成后保存结果。重启遗留任务标记 unknown，不自动重放。 同一数据库通过进程文件锁禁止启动两个
  worker。数据库需要持久保存，不自动清理去重记录。
- 普通测试失败返回 completed、passed=false；基础设施故障返回 error，不转换成零奖励。
- 未确认清理的任务保留槽位隔离。客户端无自动重试，任务 ID 写入 request trace。
- server 返回 worker_seconds；本机分阶段计时沿用 sandbox_pool 的 request trace。
- 生成代码只在 AgentEnv 沙箱内执行，不在代理宿主机执行。

### 部署配置

服务端使用单一 `provider=agentenv` 的本机配置（API 必须是 loopback），设置 `lease_timeout: 0`、独立持久
`lock_dir`、capacity 32 和只读私有 access 文件。 代理端与直连端如果并存，必须协调共同的机器容量；独立锁目录不能限制绕过代理的调用。
因此不在正在运行的 C04 期间切换或施加真实代理回放。

```bash
PYTHONPATH=/path/to/templates:/path/to/repo \
python /path/to/templates/scoring_worker.py \
  --config /private/local-agentenv-pool.json \
  --token-file /private/worker-token \
  --database /persistent/tasks.sqlite --port 18010 --capacity 32
```

200 的 backend 改用 `provider: agentenv-worker`；access 文件只含代理 `api_url` （SSH 隧道 loopback
地址或 HTTPS）和代理 `api_key`，不需要 AgentEnv 凭据或 template_id。 继续使用复合池按机器分流，每机 32、总评分并发
64。查询任务须回到最初接受它的后端。

实现位于 `templates/scoring_worker.py` 与 `templates/sandbox_pool.py`。 验证涵盖 HTTP
鉴权、状态查询、重复提交、输入冲突、容量、重启不重放、基础设施错误与 客户端适配器真实 HTTP 路径。当前 C04 冻结快照不包含代理，仍是直连复合池。

### 代理部署验证与 C04 对照启动

alice、108已部署独立 `areal-scoring-worker` 服务（各32），200经独立隧道访问。
真实沙箱验证已通过：正确测试、失败测试、删除完成、同ID同结果去重、状态查询。 代理不替换原AgentEnv服务；原直连入口保留。源码12项测试通过。
200代理C04：`/mnt/nvme2/areal-development/c04-agentenv-worker64-20261030`，固定/共享各3批，评分总64。
对照基线为此前直连复合池C04：固定256.136秒、共享239.238秒。代理版仍在运行，尚无性能结论。
本轮继续属于私有辅助研究，不进上游、不进论文；未自动commit/push。
