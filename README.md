# EchoMind

EchoMind 是一个面向客服场景的 Agent 智能助手后端。它把受控工作流、混合 RAG、实时工具调用、三层记忆、SSE 流式输出和评测链路组合在一起，目标不是做“完全自治”的通用 Agent，而是做一个回答路径清晰、证据链明确、工程上可验证的智能客服系统。

## Highlights

- Bounded Agent Workflow: 先判定 `knowledge / live_record / action`，再做槽位校验、动作规划和工具执行，而不是把所有控制权直接交给模型
- Hybrid RAG Pipeline: 父子块切分、Query Rewrite、向量召回、BM25、RRF 融合、全局 rerank、父块上下文回填
- Real-time Tooling: 支持订单查询、物流查询、退款提交、人工转接，并把实时结果和知识库证据统一注入回答上下文
- Resilience by Default: LLM 总预算重试、工具超时与熔断、Fallback、写操作幂等、流式失败不落半截回答
- Evaluation Built In: `/search` 与 `/chat` 都可评测，支持检索层 Gold 数据校验、RAGAS、Trace 和 Prometheus 指标
- Hot-reload Skills: 不重启服务即可为指定 Agent 角色追加声明式业务约束

## What It Can Handle

| Request Type | Example | Runtime Behavior |
| --- | --- | --- |
| Knowledge QA | `退款多久到账？` | 走知识检索链路，返回规则或时效说明 |
| Live Record Lookup | `帮我查一下订单状态` | 检查订单号槽位，必要时追问，再调用实时工具 |
| Action Execution | `我要退款` | 进入动作工作流，提交退款或引导补充信息 |
| Mixed Questions | `这个订单能不能退，多久能到账？` | 同时融合订单实时状态和退款规则 |

## Why This Repo Exists

很多 AI 客服 Demo 停留在“接一个向量库，再调一次大模型”。EchoMind 更关注的是完整闭环：

- 用户问题如何被路由到合适的处理模式
- 什么时候该查知识库，什么时候必须查实时系统
- 缺少关键槽位时如何优雅追问，而不是直接胡答
- 模型、工具、检索、记忆、流式输出如何协同
- 回答质量如何被评测，而不是只靠主观感觉

## Quick Try

复制环境变量模板：

```powershell
Copy-Item .env.example .env
```

启动本地 reranker：

```powershell
.\tools\install_local_reranker.ps1
.\tools\start_local_reranker.ps1
```

启动服务：

```powershell
docker compose up -d
```

最小验证：

```powershell
curl.exe -X POST "http://127.0.0.1:8000/chat" `
  -H "Content-Type: application/json" `
  -d "{\"message\":\"退款多久到账\",\"user_id\":\"u1001\"}"
```

更多启动与评测说明见下文的“快速开始”和“评测与测试”。

## 系统总览

```text
User / Frontend
        |
        v
  FastAPI API Layer
        |
        +--> MemoryManager
        |      |- Redis: working memory
        |      |- ChromaDB: episodic memory
        |      `- ChromaDB: user profile
        |
        +--> WorkflowIntentDecider
        +--> SlotManager
        +--> ActionPlanner
        +--> ActionExecutor
        |      `- ToolRegistry
        |             |- knowledge_search
        |             |- order_lookup
        |             |- shipment_track
        |             |- refund_create
        |             `- human_handoff
        |
        +--> AgentOrchestrator
        |      |- General Agent
        |      |- Technical Agent
        |      |- Billing Agent
        |      `- Escalation Agent
        |
        +--> Trace / Metrics / Evaluation
        |
        `--> Response / SSE Stream
```

### 一次 `/chat` 请求的大致生命周期

1. 读取会话记忆，拼出当前用户上下文
2. 识别粗粒度意图和实体
3. 判断这轮请求要解决的是知识问答、实时记录，还是业务动作
4. 做槽位校验，缺少订单号等关键参数时优先追问
5. 生成工作流计划，决定要不要查知识库、查订单、查物流、发起退款或转人工
6. 执行动作计划，并把结果整理为证据块
7. 交给主 Agent 生成最终答复
8. 写回工作记忆，并异步更新用户画像
9. 保存 Trace，供回放和评测使用

## RAG 设计

EchoMind 的知识检索不是单一向量搜索，而是一条完整的检索优化链路。

### 文档入库

- 知识文档可以通过接口直接导入，也可以从本地文件上传
- 文档会先切成父块，再切成子块
- 当前默认切分为：
  - 父块 `300` 字，重叠 `60`
  - 子块 `100` 字，重叠 `20`
- 向量由 DashScope `text-embedding-v4` 显式生成，ChromaDB 只负责存储和近邻检索

### 查询链路

- 对用户问题先做 Query Rewrite
- 多个子查询并行执行：
  - 向量召回
  - BM25 召回
- 用 RRF 融合候选结果
- 对融合后的 child chunk 做全局 rerank
- 返回精确命中的 child chunk，同时补回 parent context，供最终回答使用

### 这样设计的原因

- 子块负责“找得准”
- 父块负责“上下文更完整”
- 向量召回负责语义相似
- BM25 负责关键词兜底
- RRF 降低单一路径偏差
- rerank 改善最终排序质量

项目里还配有检索层 Gold 数据和问答层 RAGAS 评测，方便验证“检索做了优化”不是口头描述。

## Agent 与工作流设计

### Agent 角色

当前项目内置了四类 Agent 角色：

- `general`
- `technical`
- `billing`
- `escalation`

它不是完全开放式的多 Agent 社会模拟，而是一个静态角色分工的受控编排系统。复杂问题可以让主 Agent 和辅助 Agent 分工处理，最终由主角色做汇总。

### 工作流规划

项目将请求处理抽象为显式状态路径：

- `INTAKE`
- `PLAN`
- `CLARIFY`
- `RETRIEVE`
- `ACT`
- `EXECUTE`
- `VERIFY`
- `RESPOND`
- `HANDOFF`
- `CLOSE`

工作流计划由 `WorkflowPlan` 表达，动作由 `ActionItem` 表达。每个动作都带有：

- 动作类型
- 依赖关系
- 需要的槽位
- 超时
- 重试次数
- 失败策略

这意味着项目不是“模型想到什么做什么”，而是“模型参与决策，系统负责约束执行”。

### 动作执行

`ActionExecutor` 会按依赖和输入条件执行 DAG 风格动作：

- 依赖满足才执行
- 可并行的读操作支持并发执行
- 写操作是否允许重试，要看是否具备幂等键
- 支持失败策略：
  - `FAIL_FAST`
  - `CONTINUE`
  - `HANDOFF`
  - `ASK_USER`

这套机制让“退款提交失败后怎么办”“工具返回异常后是否要追问用户”这类问题有了工程化落点。

## 记忆设计

项目采用三层记忆结构：

- 工作记忆：Redis，保存当前会话最近消息，TTL 24 小时
- 情景记忆：ChromaDB，保存跨会话可检索历史片段
- 用户画像：ChromaDB，保存由对话提炼出的长期偏好与实体

当工作记忆增长到阈值后，会触发摘要压缩：

- 旧消息总结成摘要
- 旧内容转存到情景记忆
- 当前会话只保留最近少量消息

这样可以控制 prompt 膨胀，避免多轮对话越来越重。

## Skill 机制

EchoMind 支持声明式 Skill 热加载。Skill 的作用不是动态加代码，而是为既有 Agent 增加可审查的业务约束和答复风格。

Skill 具备几个特点：

- 只能命中既定角色、意图、目标和工具条件
- 只能引用系统里已经注册的工具
- 不能越过槽位校验、超时控制和安全边界
- 支持上传草稿、审核发布、自动热加载

如果你希望给 Billing Agent 加一个“退款规则回答风格包”，这套机制比直接改主 Prompt 更可控。

详细说明见 [skills/README.md](skills/README.md)。

## 流式输出与可观测性

### SSE 流式返回

`/chat/stream` 支持阶段化 SSE 输出：

- `understanding`
- `planning`
- `retrieving`
- `answering`

如果流式过程异常中断，系统不会把半截回答写入记忆，这一点对真实对话体验很重要。

### Trace

每次请求都可以记录 Trace，包括：

- 意图识别
- 工作流决策
- 槽位检查
- Planner
- 状态迁移
- 工具调用
- LLM 调用
- 记忆读写

这让你可以回看“为什么这轮回答用了知识库”“为什么这轮直接追问订单号”“为什么这轮走了转人工”。

### Metrics

项目暴露了 `/metrics`，并通过 Prometheus 抓取运行指标。

## 评测体系

EchoMind 不只提供应用能力，也提供评测能力。

### 问答层评测

- 支持通过 `/chat` 跑真实工作流
- 支持通过 RAGAS 评估回答质量
- 会校验 deployed 知识库是否和 Gold 文件一致

### 检索层评测

- 独立调用 `/search`
- 验证 Recall、MRR、nDCG 等指标
- 支持对原始召回和 rerank 后结果分别评估

### 当前数据资产

- `docs/customer-service-kb/` 下内置了 `10` 篇客服知识文档
- `evaluation/datasets/` 下提供了问答、工作流、检索三类数据集与 Gold 文件

详细流程见 [docs/ragas-evaluation.md](docs/ragas-evaluation.md)。

## 目录结构

```text
EchoMind/
├─ api/                 FastAPI 入口与接口定义
├─ agents/              Agent 编排与角色执行
├─ core/                韧性控制、Embedding 客户端、意图识别
├─ memory/              三层记忆管理
├─ mcp/                 工具管理、知识库、外部服务适配
├─ workflow/            工作流决策、状态机、动作规划与执行
├─ telemetry/           Trace 采集与存储
├─ monitor/             Prometheus 与运行监控
├─ evaluation/          RAGAS 与检索评测
├─ local_reranker/      宿主机 GPU reranker 服务
├─ skills/              声明式 Skill 目录与运行时
├─ docs/                评测说明与客服知识库文档
├─ tests/               单元测试 / 集成测试
├─ data/                Chroma、评测产物、Skill 数据
├─ docker-compose.yml   主运行编排
├─ docker-compose.dev.yml 开发态热更新编排
└─ README.md
```

## 技术栈

### 应用与推理

- FastAPI
- Uvicorn
- Anthropic-compatible LLM API
- DashScope `text-embedding-v4`

### 存储与检索

- Redis
- ChromaDB
- BM25（进程内索引）

### 评测与监控

- Prometheus
- RAGAS
- 自定义 Retrieval Runner

### 推理增强

- 宿主机 FastAPI reranker 服务
- `BAAI/bge-reranker-v2-m3`

## 快速开始

### 1. 前置要求

推荐环境：

- Python `3.12`
- Docker / Docker Compose
- Windows PowerShell（项目当前脚本以此为主）
- 可用的 Anthropic-compatible LLM 接口
- 可用的 DashScope Embedding Key

如果要启用本地 GPU reranker，还需要：

- NVIDIA GPU
- CUDA 可用的 Python 环境

### 2. 配置环境变量

复制配置模板：

```powershell
Copy-Item .env.example .env
```

至少需要填写这些关键项：

- `ANTHROPIC_API_KEY`
- `ANTHROPIC_BASE_URL`（如使用兼容接口）
- `ANTHROPIC_MODEL`
- `DASHSCOPE_API_KEY`
- `DASHSCOPE_HTTP_BASE_URL`
- `DASHSCOPE_WORKSPACE`

如果只是先把服务跑起来，不做评测，可以暂时不配置 RAGAS 专用 embedding 环境变量。

### 3. 启动本地 reranker（推荐）

项目当前默认推荐宿主机本地 reranker，而不是直接使用 Compose 内 TEI。

```powershell
.\tools\install_local_reranker.ps1
.\tools\start_local_reranker.ps1
```

健康检查：

```powershell
Invoke-RestMethod http://127.0.0.1:18080/health
```

更多说明见 [local_reranker/README.md](local_reranker/README.md)。

### 4. 启动主服务

```powershell
docker compose up -d
```

检查健康状态：

```powershell
Invoke-RestMethod http://127.0.0.1:8000/health
```

打开接口文档：

- Swagger UI: `http://127.0.0.1:8000/docs`

### 5. 最小验证

知识检索：

```powershell
curl.exe -X POST "http://127.0.0.1:8000/search?query=退款多久到账&top_k=5&include_debug=true"
```

单轮聊天：

```powershell
curl.exe -X POST "http://127.0.0.1:8000/chat" `
  -H "Content-Type: application/json" `
  -d "{\"message\":\"退款多久到账\",\"user_id\":\"u1001\"}"
```

## 开发模式

项目提供了开发态热更新配置，适合频繁改 Python 代码时使用：

```powershell
docker compose -f docker-compose.yml -f docker-compose.dev.yml up -d
```

这个模式会：

- 使用 `Dockerfile` 的 `development` target
- 挂载本地源码
- 用 `uvicorn --reload` 自动热更新
- 保持 Redis / ChromaDB / Prometheus 常驻

当你修改了 `Dockerfile`、依赖或主 Compose 编排时，再回到常规 `docker compose up -d --build`。

## 常用接口

| 方法 | 路径 | 说明 |
| --- | --- | --- |
| `GET` | `/health` | 健康检查 |
| `POST` | `/chat` | 主对话接口 |
| `POST` | `/chat/stream` | SSE 流式对话接口 |
| `POST` | `/search` | 检索链路调试接口 |
| `POST` | `/knowledge/add` | 批量追加知识文档 |
| `POST` | `/knowledge/replace` | 全量替换知识库并重建索引 |
| `POST` | `/knowledge/upload` | 上传文件导入知识库 |
| `GET` | `/knowledge/stats` | 查看知识片段数量 |
| `GET` | `/knowledge/chunks` | 导出 child chunk 清单 |
| `POST` | `/eval/run` | 跑内置评测 |
| `GET` | `/metrics` | Prometheus 指标 |
| `GET` | `/traces/recent` | 查看最近 Trace |
| `GET` | `/traces/{trace_id}` | 查看单次 Trace 详情 |
| `POST` | `/admin/skills/upload` | 上传 Skill |
| `POST` | `/admin/skills/{draft_id}/publish` | 发布 Skill 草稿 |
| `GET` | `/admin/skills` | 查看 Skill 状态 |
| `POST` | `/admin/skills/reload` | 手动重载 Skill |

## 知识库管理

项目内置了一批客服知识文档示例，位于：

- `docs/customer-service-kb/`

你可以：

- 直接通过 `/knowledge/upload` 上传 `.md`、`.txt` 或 `.json`
- 通过 `/knowledge/add` 传结构化 JSON
- 通过 `/knowledge/replace` 在切分参数变更后整体重建知识库

如果你修改了父块 / 子块切分策略，务必同步重建知识库，否则评测结果会和 Gold 文件失配。

## 评测与测试

### 运行单元测试

```powershell
python -m unittest discover -s tests
```

### 运行问答层 RAGAS 评测

```powershell
python -m dotenv run -- python -m evaluation.ragas_runner `
  --base-url http://127.0.0.1:8000 `
  --dataset evaluation/datasets/customer_service_v1.jsonl `
  --workflow-dataset evaluation/datasets/customer_service_workflow_v1.jsonl `
  --output-dir data/eval `
  --top-k 5 `
  --recall-k 20 `
  --require-rerank true `
  --required-rerank-provider tei `
  --max-concurrency 3
```

### 运行检索层评测

```powershell
python -m evaluation.retrieval_runner `
  --base-url http://127.0.0.1:8000 `
  --dataset evaluation/datasets/customer_service_search_v1.jsonl `
  --gold evaluation/datasets/customer_service_search_gold_v2.json `
  --output-dir data/eval `
  --top-k 5 `
  --recall-k 20 `
  --require-rerank true `
  --required-rerank-provider tei
```

### 汇总评测结果

```powershell
python -m tools.summarize_ragas_report data/eval/<run_id>.json
```

## 工程韧性设计

这部分不是功能清单，而是项目在工程实现上真正有区分度的地方：

- LLM 调用统一走预算式重试：总超时、指数退避、full jitter
- 工具层支持缓存、超时控制、熔断与 fallback
- 工作流动作支持失败策略，而不是简单 `try/except`
- 写操作（如退款、转人工）只有拿到幂等键时才允许重试
- 流式输出失败时不持久化半截 Assistant 回复
- 工作记忆自动压缩，控制多轮上下文膨胀
- 检索评测会验证 active chunk 与 Gold 内容哈希，避免对错版本知识库做“伪评测”

## 当前范围与非目标

为了避免误解，这里明确项目当前的边界：

- 不是通用自治型 ReAct Agent 平台
- 不是完整标准化的 MCP Server 实现，当前更接近“类 MCP 的工具注册与调用框架”
- 不是生产级分布式高可用系统
- 订单、物流、退款、人工转接外部系统目前主要通过 mock 接口演示集成方式
- 当前 `/chat` 请求仍主要依赖上游传入的 `user_id`，没有完整的统一身份认证和工具级 scope 控制

## 相关文档

- [docs/ragas-evaluation.md](docs/ragas-evaluation.md)
- [skills/README.md](skills/README.md)
- [local_reranker/README.md](local_reranker/README.md)
- [日常开发流程.md](日常开发流程.md)

## 后续可演进方向

- 增加统一身份认证与工具级 scope 控制
- 将 mock 外部系统替换为真实 OMS / TMS / 售后服务
- 引入更细粒度的权限、限流和审计
- 为 Prompt 组装增加统一 token budget 管理
- 将 Trace、评测、知识库版本管理做成更完整的运营闭环
