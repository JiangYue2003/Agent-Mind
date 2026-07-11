# EchoMind Skill 使用说明

`skills/` 是 EchoMind 的声明式 Agent Skill 目录。一个 Skill 用于给已有的
General、Technical、Billing 或 Escalation Agent 追加特定业务约束、答复风格和
工具绑定条件，无需重启 FastAPI 服务即可生效。

Skill 不是 Python 插件：不能在 ZIP 中携带代码，不能动态注册新的 MCP 工具，不能
绕开工作流的槽位校验、超时、熔断或权限边界。真实工具调用仍由现有
`MCPToolManager` 和 Workflow 执行。

## 目录与来源

运行时读取两个活动来源：

```text
skills/
  catalog/
    <skill-id>/
      <version>/
        skill.json
        prompt.md

data/skills/
  drafts/
    <draft-id>/<skill-id>/<version>/
  published/
    <skill-id>/<version>/
```

`skills/catalog` 用于 Git 管理的正式包，应用对它只读。`data/skills/drafts`
保存后台上传但尚未发布的草稿；`data/skills/published` 保存已经发布的上传包。

同一个 `id@version` 不能同时出现在 catalog 和 published 中。运行时会拒绝整个
新快照并保留上一份有效版本，因此发布新版本时必须递增版本号。

## Skill 包格式

上传 ZIP 或直接放入 catalog 的目录都必须只包含两个文件，且都位于包根目录：

```text
skill.json
prompt.md
```

上传 ZIP 不允许子目录、隐藏文件或额外文件。ZIP 本身和解压后的总大小均不得超过
128 KB；`skill.json` 不得超过 64 KB，`prompt.md` 不得超过 32 KB，且必须是
UTF-8 文本。

### `skill.json`

下面是一个用于退款规则问答的完整示例：

```json
{
  "id": "refund-policy",
  "name": "Refund Policy",
  "version": "1.0.0",
  "description": "规范退款规则类答复",
  "targets": {
    "agent_roles": ["billing"],
    "intent_categories": ["billing"],
    "goals": ["knowledge"],
    "tools": ["knowledge_search"]
  },
  "allowed_tools": ["knowledge_search"],
  "prompt_file": "prompt.md",
  "priority": 20
}
```

字段说明：

| 字段 | 必填 | 说明 |
| --- | --- | --- |
| `id` | 是 | 小写字母开头，只能包含小写字母、数字和 `-`。 |
| `name` | 是 | 管理接口和排障页面中展示的名称。 |
| `version` | 是 | 严格三段语义版本，例如 `1.0.0`。 |
| `description` | 是 | 简短描述用途，不会注入 Agent 提示词。 |
| `targets.agent_roles` | 是 | 可选值：`general`、`technical`、`billing`、`escalation`。 |
| `targets.intent_categories` | 是 | 当前意图识别结果的值，例如 `billing`。 |
| `targets.goals` | 是 | 可选值：`knowledge`、`live_record`、`action`。 |
| `targets.tools` | 否 | 只有当前 Workflow 已计划这些工具时才激活。 |
| `allowed_tools` | 否 | 绑定到当前 MCP 工具目录；当前计划必须包含其中全部工具才会激活。 |
| `prompt_file` | 是 | 只能是根目录下的 `prompt.md`。 |
| `priority` | 否 | 整数，范围 `-1000` 到 `1000`，默认 `0`；值越大越早注入。 |

`targets.tools` 与 `allowed_tools` 都必须引用应用启动时已经注册的 MCP 工具，例如
当前的 `knowledge_search`、`order_lookup`、`shipment_track`、`refund_create`、
`human_handoff`。它们是 Skill 的选择/绑定条件，不会自动让 LLM 执行工具，也不能
增加一个未在代码中注册的新工具。

### `prompt.md`

此文件会在命中后以 `[Skill: <id>]` 区块追加到目标 Agent 的系统提示词。写法应当
具体、短小且可执行，例如：

```md
只依据知识库和工作流提供的证据说明退款规则。
不要把会话记忆、用户画像或历史摘要描述为实时退款结果。
知识库没有明确给出的时效、金额或例外条件，直接说明信息不足。
```

不要在 Skill 中要求模型伪造订单、物流、退款等实时事实。实时事实只能来自既有工具
结果；Skill 不能覆盖 Agent 的基础安全约束。

## 激活规则

每个 `/chat` 请求在工作流执行前捕获一个不可变的 Skill 快照。随后按下面条件选择
Skill：

1. 当前 Agent 角色在 `targets.agent_roles` 中；
2. 当前意图在 `targets.intent_categories` 中；
3. 当前 Workflow 目标在 `targets.goals` 中；
4. `targets.tools` 和 `allowed_tools` 中的工具都已被当前 Workflow 计划；
5. 同时命中多个 Skill 时，按 `priority` 从高到低、再按 `id` 排序。

Primary Agent、Supporting Agent 与独立的 `final_writer` 都分别选择自己的 Skill，
不会把 Billing Skill 传给 Technical Agent。热加载只影响之后的新请求；已经开始的
请求继续使用它开始时捕获的快照。

## 上传与审核

所有管理接口都要求请求头：

```text
X-Skill-Admin-Token: <SKILL_ADMIN_TOKEN>
```

先在运行应用的环境中设置：

```dotenv
SKILL_ADMIN_TOKEN=replace_with_a_long_random_value
```

未配置令牌时，管理接口返回 `503`；令牌不正确时返回 `403`。修改环境变量后需要
重启应用容器或进程。

### 开发态：上传即发布

当 `APP_ENV=development` 且未显式设置 `SKILL_REVIEW_REQUIRED` 时，审核默认关闭。
有效 ZIP 会经历校验、写入草稿、原子发布、刷新快照，并在同一个上传请求中返回：

```json
{
  "status": "published",
  "auto_published": true
}
```

这就是本地“即插即用”的路径。审核关闭不代表绕过包校验、令牌校验或工具白名单。

### 生产态：先草稿，后发布

在非 development 环境中，审核默认开启。上传成功后返回：

```json
{
  "status": "draft",
  "auto_published": false,
  "draft_id": "..."
}
```

审核通过后，由管理员使用该 `draft_id` 发布。也可以通过
`SKILL_REVIEW_REQUIRED=true|false` 显式覆盖默认行为；该设置在应用启动时读取，修改
后应重启应用。

### PowerShell 上传示例

以下命令假设 API 在本机 `8000` 端口：

```powershell
$token = $env:SKILL_ADMIN_TOKEN
$zip = "C:\work\refund-policy-1.0.0.zip"

curl.exe -sS -X POST "http://localhost:8000/admin/skills/upload" `
  -H "X-Skill-Admin-Token: $token" `
  -F "file=@$zip;type=application/zip"
```

生产态发布草稿：

```powershell
$draftId = "replace_with_draft_id"

curl.exe -sS -X POST "http://localhost:8000/admin/skills/$draftId/publish" `
  -H "X-Skill-Admin-Token: $token"
```

### 打包 ZIP

压缩时传入两个文件本身，而不是包含它们的目录，否则 ZIP 会带一层目录并被拒绝：

```powershell
$package = "C:\work\refund-policy\1.0.0"
Compress-Archive -Force `
  -Path "$package\skill.json", "$package\prompt.md" `
  -DestinationPath "C:\work\refund-policy-1.0.0.zip"
```

## 自动热加载

`SkillRuntime` 在 FastAPI lifespan 中启动轮询任务。默认每秒检查一次活动来源，可用
下面的环境变量调整，最小有效值为 `0.1` 秒：

```dotenv
SKILL_RELOAD_INTERVAL_SECONDS=1
```

对 `skills/catalog/<id>/<version>/skill.json` 或 `prompt.md` 的新增、修改、删除，都会
触发重新校验和快照替换。加载器会在读取前后比对文件指纹；如果文件正被编辑，会重试
三次。包无效、版本重复或目录持续变化时，上一份有效快照保持服务，新错误可从状态接口
查看。

上传发布无需等待轮询：成功发布会在请求内立即刷新。发布后，只有该版本确实进入活动
快照，接口才返回成功；否则新发布目录会回滚到草稿状态。

## 手动加载与状态检查

查看活动 Skill、草稿、最后成功刷新时间和最近加载错误：

```powershell
$token = $env:SKILL_ADMIN_TOKEN
curl.exe -sS "http://localhost:8000/admin/skills" `
  -H "X-Skill-Admin-Token: $token"
```

立即触发一次全量重新加载：

```powershell
curl.exe -sS -X POST "http://localhost:8000/admin/skills/reload" `
  -H "X-Skill-Admin-Token: $token"
```

如果返回的 `generation` 增加，并且目标 Skill 出现在 `active` 中，说明新快照已生效。
`active` 不返回 `prompt.md` 正文，避免管理状态接口泄露完整提示词。

运行 Trace 的 `orchestrator.run` 阶段会记录：

```json
{
  "skill_snapshot_generation": 12,
  "skill_ids_by_role": {
    "billing": ["refund-policy"]
  }
}
```

这些字段用于确认某次响应实际使用了哪些 Skill，不包含提示词正文。

## Git 管理的手动目录加载

如果 Skill 需要和业务代码一起评审，直接创建目录：

```text
skills/catalog/refund-policy/1.0.0/skill.json
skills/catalog/refund-policy/1.0.0/prompt.md
```

提交 Git 后，让运行主机上挂载的 `skills/catalog` 同步到该版本。生产 Compose 已把它以
只读方式挂载到 `/app/skills/catalog`。文件落盘后等待轮询，或调用 `/admin/skills/reload`
立即加载。

不要直接编辑 `data/skills/published` 或 `data/skills/drafts`：前者由发布流程维护，后者
需要保留完整的草稿目录结构和校验结果。

## 日常开发注意事项

1. 本地调试前确认实际运行进程有 `APP_ENV=development` 和
   `SKILL_ADMIN_TOKEN`；仅修改 `.env.example` 不会改变已运行容器。
2. 上传 Skill 前先在测试中构造相同 manifest，至少覆盖命中条件、无效工具、重复版本和
   热加载后的快照行为。相关测试位于 `tests/test_skill_runtime.py` 和
   `tests/test_skill_admin_api.py`。
3. 新业务工具必须先经代码方式注册到 `MCPToolManager`、补齐 Workflow 槽位/执行器和
   测试，之后 Skill 才能引用它。只上传 Skill 不会让未知工具可执行。
4. 修改已发布的上传 Skill 时递增 `version`。直接改写同一已发布版本会被拒绝；catalog
   中的同版本文件可以在 Git 变更下热加载，但不能和 published 同版本共存。
5. `priority` 只决定提示词拼接顺序，不应用于解决业务冲突。多个 Skill 有冲突时，应先
   收敛为一个职责更清晰的 Skill。
6. 每次变更后检查 `/admin/skills` 的 `generation`、`active`、`last_errors`，并在
   `/traces/{trace_id}` 中确认 `skill_ids_by_role`。
7. Skill 提示词应保持短小、可审查，避免复制基础 Agent 的整段系统提示词；基础安全与
   实时事实约束由 Agent 和 Workflow 统一负责。

## 管理接口速查

| 方法 | 路径 | 用途 |
| --- | --- | --- |
| `POST` | `/admin/skills/upload` | 上传 ZIP；开发态自动发布，生产态创建草稿。 |
| `POST` | `/admin/skills/{draft_id}/publish` | 显式发布生产草稿。 |
| `GET` | `/admin/skills` | 查看 generation、活动包、草稿、刷新时间和错误。 |
| `POST` | `/admin/skills/reload` | 手动重新扫描并刷新活动来源。 |
