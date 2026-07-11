"""
EchoMind 智能客服系统 — FastAPI 入口

启动时打印小熊饼干图案。
所有核心组件在 lifespan 中初始化，通过环境变量配置。
"""
import asyncio
import hashlib
import logging
import os
import pathlib
import sys
import time
import uuid
from datetime import datetime, timezone
from contextlib import asynccontextmanager
from typing import Any, Dict, List, Optional

# 将项目根目录加入 sys.path，确保无论从哪里执行都能找到 agents/core/memory 等模块
# 这一行必须在所有项目内部 import 之前执行
_ROOT = str(pathlib.Path(__file__).parent.parent.resolve())
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Response, UploadFile, File, Header
from fastapi.middleware.cors import CORSMiddleware
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest
from pydantic import BaseModel

load_dotenv()

logging.basicConfig(
    level=getattr(logging, os.getenv("LOG_LEVEL", "INFO")),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

BANNER = r"""
    ʕ•ᴥ•ʔ  ʕ•ᴥ•ʔ  ʕ•ᴥ•ʔ
   ╔══════════════════════╗
   ║   EchoMind  v2.0     ║
   ║   智能客服 AI 系统    ║
   ╚══════════════════════╝
    ʕ•ᴥ•ʔ  ʕ•ᴥ•ʔ  ʕ•ᴥ•ʔ
"""

# ── 全局组件（lifespan 中初始化）─────────────────────────────────────────────
_orchestrator = None
_memory       = None
_tool_manager = None
_monitor      = None
_evaluator    = None
_chat_intent_recognizer = None
_trace_store  = None
_action_planner = None
_slot_manager = None
_state_machine = None
_workflow_intent_decider = None
_skill_runtime = None
_skill_store = None
_skill_reload_task = None
_mock_handoff_records: List[Dict[str, Any]] = []
_mock_refund_records: Dict[str, Dict[str, Any]] = {}
_mock_idempotency_records: Dict[str, tuple[Dict[str, Any], Dict[str, Any]]] = {}

_MOCK_ORDER_DATA: Dict[str, Dict[str, Any]] = {
    "ORD20250701001": {
        "order_id": "ORD20250701001",
        "user_id": "u1001",
        "status": "运输中",
        "amount": "99.00",
        "payment_status": "已支付",
        "shipment_status": "运输中",
        "refund_status": "",
        "updated_at": "2026-07-02T10:20:00+08:00",
        "source": "mock_oms",
    },
    "ORD20250701002": {
        "order_id": "ORD20250701002",
        "user_id": "u456",
        "status": "已退款",
        "amount": "199.00",
        "payment_status": "已支付",
        "shipment_status": "未发货",
        "refund_status": "退款成功",
        "updated_at": "2026-07-02T09:00:00+08:00",
        "source": "mock_oms",
    },
    "ORD20250701003": {
        "order_id": "ORD20250701003",
        "user_id": "u1001",
        "status": "已支付",
        "amount": "59.90",
        "payment_status": "已支付",
        "shipment_status": "待发货",
        "refund_status": "",
        "updated_at": "2026-07-02T11:15:00+08:00",
        "source": "mock_oms",
    },
}

_MOCK_SHIPMENT_DATA: Dict[str, Dict[str, Any]] = {
    "ORD20250701001": {
        "order_id": "ORD20250701001",
        "user_id": "u1001",
        "carrier": "顺丰速运",
        "tracking_no": "SF1234567890",
        "shipment_status": "运输中",
        "events": [
            {"time": "2026-07-02 12:00:00", "status": "商家已发货"},
            {"time": "2026-07-03 09:30:00", "status": "快件运输中"},
        ],
        "source": "mock_tms",
    },
    "ORD20250701002": {
        "order_id": "ORD20250701002",
        "user_id": "u456",
        "carrier": "圆通速递",
        "tracking_no": "YT9988776655",
        "shipment_status": "未发货",
        "events": [
            {"time": "2026-07-02 09:00:00", "status": "订单已取消，无需发货"},
        ],
        "source": "mock_tms",
    },
}


def _anthropic_cfg() -> Dict[str, Any]:
    key = os.getenv("ANTHROPIC_API_KEY", "")
    if not key:
        raise RuntimeError("未设置 ANTHROPIC_API_KEY")
    cfg: Dict[str, Any] = {
        "api_key":  key,
        "model":    os.getenv("ANTHROPIC_MODEL", "claude-3-5-sonnet-20241022"),
    }
    base_url = os.getenv("ANTHROPIC_BASE_URL", "").strip()
    if base_url:
        cfg["base_url"] = base_url
    return cfg


def _ensure_trace_store():
    global _trace_store
    if _trace_store is None:
        from telemetry.runtime import TraceStore
        _trace_store = TraceStore(capacity=int(os.getenv("TRACE_STORE_CAPACITY", "200")))
    return _trace_store


def _skill_review_required() -> bool:
    configured = os.getenv("SKILL_REVIEW_REQUIRED", "").strip().lower()
    if configured:
        return configured in {"1", "true", "yes", "on"}
    return os.getenv("APP_ENV", "production").strip().lower() != "development"


def _create_skill_runtime(tool_manager: Any):
    from skills.runtime import SkillRuntime

    skill_catalog_dir = pathlib.Path(os.getenv("SKILL_CATALOG_DIR", str(pathlib.Path(_ROOT) / "skills" / "catalog")))
    skill_data_dir = pathlib.Path(os.getenv("SKILL_DATA_DIR", str(pathlib.Path(_ROOT) / "data" / "skills")))
    return SkillRuntime(
        catalog_dir=skill_catalog_dir,
        published_dir=skill_data_dir / "published",
        drafts_dir=skill_data_dir / "drafts",
        known_tools=tool_manager._tools.keys(),
    )


async def _watch_skill_catalog(runtime: Any, interval_s: float) -> None:
    while True:
        await asyncio.sleep(interval_s)
        try:
            if runtime.refresh_if_changed():
                logger.info("Skill 目录已热加载到 generation=%s", runtime.snapshot.generation)
        except Exception as ex:
            logger.warning("Skill 目录热加载失败: %s", ex)


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _orchestrator, _memory, _tool_manager, _monitor, _evaluator, _chat_intent_recognizer, _trace_store
    global _action_planner, _slot_manager, _state_machine, _workflow_intent_decider
    global _skill_runtime, _skill_store, _skill_reload_task

    print(BANNER, flush=True)

    from agents.agent_orchestrator import AgentOrchestrator, Request
    from core.intent_recognizer import IntentRecognizer
    from evaluation.evaluator import EndToEndEvaluator
    from mcp.handoff_service import HumanHandoffService
    from mcp.knowledge_base import KnowledgeBase
    from mcp.order_lookup import OrderLookupService
    from mcp.refund_create import RefundCreateService
    from mcp.shipment_track import ShipmentTrackService
    from mcp.tool_manager import MCPToolManager, Tool
    from memory.conversation_memory import MemoryManager
    from monitor.performance_monitor import PerformanceMonitor
    from skills.runtime import SkillRuntime, SkillStore
    from telemetry.runtime import TraceStore
    from workflow.action_planner import ActionPlanner
    from workflow.intent_decider import WorkflowIntentDecider
    from workflow.slot_manager import SlotManager
    from workflow.state_machine import WorkflowStateMachine

    cfg = _anthropic_cfg()
    logger.info(f"模型: {cfg['model']}  base_url: {cfg.get('base_url', '(官方)')}")

    # 意图识别器（Orchestrator 内部也会创建，这里单独暴露给 Evaluator）
    recognizer = IntentRecognizer(
        api_key=cfg["api_key"],
        base_url=cfg.get("base_url"),
        model=cfg["model"],
    )
    _chat_intent_recognizer = recognizer
    _workflow_intent_decider = WorkflowIntentDecider(
        api_key=cfg["api_key"],
        base_url=cfg.get("base_url"),
        model=cfg["model"],
    )
    _trace_store = TraceStore(capacity=int(os.getenv("TRACE_STORE_CAPACITY", "200")))
    _slot_manager = SlotManager()
    _action_planner = ActionPlanner(slot_manager=_slot_manager)
    _state_machine = WorkflowStateMachine(max_action_steps=3)

    # Agent 编排器
    _orchestrator = AgentOrchestrator(
        api_key=cfg["api_key"],
        base_url=cfg.get("base_url"),
        model=cfg["model"],
    )

    # 记忆管理器（Redis 工作记忆 + ChromaDB 情景记忆/用户画像）
    _memory = MemoryManager(
        redis_url=os.getenv("REDIS_URL", "redis://redis:6379/0"),
        chroma_host=os.getenv("CHROMA_HOST", "chromadb"),
        chroma_port=int(os.getenv("CHROMA_PORT", "8000")),
        chroma_path=os.getenv("CHROMA_PERSIST_DIRECTORY", "/app/data/chroma"),
        api_key=cfg["api_key"],
        base_url=cfg.get("base_url"),
        model=cfg["model"],
    )

    # MCP 工具管理器 + RAG 知识库（基于 ChromaDB 的真实检索）
    _tool_manager = MCPToolManager(
        api_key=cfg["api_key"],
        base_url=cfg.get("base_url"),
        model=cfg["model"],
    )
    kb = KnowledgeBase(
        chroma_host=os.getenv("CHROMA_HOST", "chromadb"),
        chroma_port=int(os.getenv("CHROMA_PORT", "8000")),
        chroma_path=os.getenv("CHROMA_PERSIST_DIRECTORY", "/app/data/chroma"),
    )
    order_lookup = OrderLookupService(base_url=os.getenv("ORDER_LOOKUP_BASE_URL", "http://localhost:8000"))
    shipment_track = ShipmentTrackService(base_url=os.getenv("SHIPMENT_TRACK_BASE_URL", "http://localhost:8000"))
    refund_create = RefundCreateService(base_url=os.getenv("REFUND_CREATE_BASE_URL", "http://localhost:8000"))
    handoff_service = HumanHandoffService(base_url=os.getenv("HANDOFF_BASE_URL", "http://localhost:8000"))
    logger.info(f"知识库已加载: {kb.doc_count} 个文档片段")

    def knowledge_fallback(params: Dict[str, Any], context: Optional[Dict[str, Any]], error: str):
        query = params.get("query", "")
        return [{
            "title": "知识库降级结果",
            "content": f"知识库暂时不可用，未能完成对“{query}”的语义检索。请稍后重试，或转人工客服确认。",
            "score": 0.0,
            "fallback": True,
            "error": error,
        }]

    _tool_manager.register(Tool(
        name="knowledge_search",
        description="搜索知识库（基于 ChromaDB 向量检索）",
        handler=kb.search_handler,
        schema={
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "top_k": {"type": "integer"},
            },
            "required": ["query"],
        },
        cache_ttl=300.0,
        supports_rerank=True,
        rerank_handler=kb.rerank_handler,
        fallback=knowledge_fallback,
    ))
    _tool_manager.register(Tool(
        name="order_lookup",
        description="查询订单状态（通过外部 OMS 接口）",
        handler=order_lookup.lookup_handler,
        schema={
            "type": "object",
            "properties": {
                "user_id": {"type": "string"},
                "order_id": {"type": "string"},
            },
            "required": ["user_id", "order_id"],
        },
        cache_ttl=30.0,
    ))
    _tool_manager.register(Tool(
        name="shipment_track",
        description="查询物流轨迹（通过外部物流接口）",
        handler=shipment_track.track_handler,
        schema={
            "type": "object",
            "properties": {
                "user_id": {"type": "string"},
                "order_id": {"type": "string"},
            },
            "required": ["user_id", "order_id"],
        },
        cache_ttl=15.0,
    ))
    _tool_manager.register(Tool(
        name="refund_create",
        description="提交退款申请（通过外部售后接口）",
        handler=refund_create.create_handler,
        schema={
            "type": "object",
            "properties": {
                "user_id": {"type": "string"},
                "order_id": {"type": "string"},
                "reason": {"type": "string"},
            },
            "required": ["user_id", "order_id"],
        },
    ))

    _tool_manager.register(Tool(
        name="human_handoff",
        description="转人工并写入会话上下文到外部客服系统",
        handler=handoff_service.handoff_handler,
        schema={
            "type": "object",
            "properties": {
                "user_id": {"type": "string"},
                "conv_id": {"type": "string"},
                "latest_message": {"type": "string"},
                "intent": {"type": "string"},
                "urgency": {"type": "string"},
                "reason": {"type": "string"},
                "summary": {"type": "string"},
                "recent_messages": {"type": "array"},
                "user_profile": {"type": "object"},
                "order_snapshot": {"type": "object"},
                "knowledge_context": {"type": "array"},
            },
            "required": ["user_id", "conv_id", "latest_message"],
        },
    ))

    _skill_runtime = _create_skill_runtime(_tool_manager)
    _skill_runtime.refresh()
    _skill_store = SkillStore(runtime=_skill_runtime, review_required=_skill_review_required())
    reload_interval_s = max(0.1, float(os.getenv("SKILL_RELOAD_INTERVAL_SECONDS", "1")))
    _skill_reload_task = asyncio.create_task(_watch_skill_catalog(_skill_runtime, reload_interval_s))

    # 性能监控（可选启动 Prometheus）
    prom_port = int(os.getenv("PROMETHEUS_PORT", "0")) or None
    _monitor = PerformanceMonitor(
        orchestrator=_orchestrator,
        tool_manager=_tool_manager,
        interval_s=float(os.getenv("MONITOR_INTERVAL", "10")),
        webhook_url=os.getenv("ALERT_WEBHOOK_URL") or None,
        prometheus_port=prom_port,
    )
    await _monitor.start()

    # 评测器
    _evaluator = EndToEndEvaluator(
        orchestrator=_orchestrator,
        recognizer=recognizer,
        api_key=cfg["api_key"],
        base_url=cfg.get("base_url"),
        model=cfg["model"],
        baseline_path=os.getenv("EVAL_BASELINE_PATH", "/app/data/eval/baseline.json"),
    )

    logger.info("EchoMind 已就绪")
    yield

    if _skill_reload_task is not None:
        _skill_reload_task.cancel()
        try:
            await _skill_reload_task
        except asyncio.CancelledError:
            pass
        _skill_reload_task = None
    await _monitor.stop()
    logger.info("EchoMind 已关闭")


# ── FastAPI ───────────────────────────────────────────────────────────────────
app = FastAPI(
    title="EchoMind 智能客服",
    version="2.0.0",
    lifespan=lifespan,
    docs_url="/docs",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── 请求/响应模型 ─────────────────────────────────────────────────────────────
class ChatRequest(BaseModel):
    message:     str
    user_id:     str = "anonymous"
    conv_id:     Optional[str] = None
    operation_id: Optional[str] = None


class ChatResponse(BaseModel):
    conv_id:     str
    response:    str
    intent:      str
    agent_type:  str
    escalated:   bool
    latency_ms:  float
    knowledge_used: bool = False
    trace_id:    str = ""


def _require_skill_admin(token: Optional[str]) -> None:
    expected = os.getenv("SKILL_ADMIN_TOKEN", "").strip()
    if not expected:
        raise HTTPException(503, "Skill 管理接口未配置")
    if token != expected:
        raise HTTPException(403, "Skill 管理权限不足")


@app.post("/admin/skills/upload", status_code=201, tags=["Skills"])
async def upload_skill(
    file: UploadFile = File(...),
    x_skill_admin_token: Optional[str] = Header(default=None),
):
    _require_skill_admin(x_skill_admin_token)
    if _skill_store is None or _skill_runtime is None:
        raise HTTPException(503, "Skill 运行时未初始化")
    if not (file.filename or "").lower().endswith(".zip"):
        raise HTTPException(400, "Skill 上传文件必须是 ZIP")

    from skills.runtime import SkillValidationError

    try:
        result = _skill_store.upload_zip(await file.read())
    except SkillValidationError as exc:
        raise HTTPException(400, str(exc)) from exc
    return {
        "draft_id": result.draft_id,
        "status": result.status,
        "auto_published": result.auto_published,
        "skill": {
            "id": result.skill.id,
            "version": result.skill.version,
        },
        "generation": _skill_runtime.snapshot.generation,
    }


@app.post("/admin/skills/{draft_id}/publish", tags=["Skills"])
async def publish_skill(
    draft_id: str,
    x_skill_admin_token: Optional[str] = Header(default=None),
):
    _require_skill_admin(x_skill_admin_token)
    if _skill_store is None or _skill_runtime is None:
        raise HTTPException(503, "Skill 运行时未初始化")

    from skills.runtime import SkillValidationError

    try:
        skill = _skill_store.publish(draft_id)
    except SkillValidationError as exc:
        raise HTTPException(400, str(exc)) from exc
    return {
        "status": "published",
        "skill": {"id": skill.id, "version": skill.version},
        "generation": _skill_runtime.snapshot.generation,
    }


@app.get("/admin/skills", tags=["Skills"])
async def skill_catalog(x_skill_admin_token: Optional[str] = Header(default=None)):
    _require_skill_admin(x_skill_admin_token)
    if _skill_store is None or _skill_runtime is None:
        raise HTTPException(503, "Skill 运行时未初始化")
    payload = _skill_runtime.describe()
    payload["drafts"] = _skill_store.list_drafts()
    return payload


@app.post("/admin/skills/reload", tags=["Skills"])
async def reload_skills(x_skill_admin_token: Optional[str] = Header(default=None)):
    _require_skill_admin(x_skill_admin_token)
    if _skill_runtime is None:
        raise HTTPException(503, "Skill 运行时未初始化")
    _skill_runtime.refresh()
    return _skill_runtime.describe()


# ── 路由 ──────────────────────────────────────────────────────────────────────
@app.get("/health")
async def health():
    if _orchestrator is None:
        raise HTTPException(503, "服务未就绪")
    return {"status": "ok", "agents": _orchestrator.get_stats()}


@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    """
    主对话接口。完整流程：
      记忆读取 → 意图识别 → Agent 路由 → 执行 → 记忆写入
    """
    if _orchestrator is None or _memory is None:
        raise HTTPException(503, "服务未就绪")

    from agents.agent_orchestrator import Request as OrcReq
    from agents.agent_orchestrator import AgentType
    from memory.conversation_memory import MsgRole
    from telemetry.runtime import TraceContext
    from workflow.state_machine import WorkflowState

    conv_id = req.conv_id or str(uuid.uuid4())
    trace = TraceContext(user_id=req.user_id, conv_id=conv_id, message=req.message)
    trace_store = _ensure_trace_store()
    response_text = ""
    response_agent_type = AgentType.GENERAL
    response_intent = None
    response_escalated = False
    knowledge_used = False
    response_latency_ms = 0.0
    skill_snapshot = _skill_runtime.snapshot if _skill_runtime is not None else None
    t0 = time.monotonic()

    try:
        with trace.stage("chat.total"):
            # 1. 读取记忆上下文
            with trace.stage("memory.read"):
                mem_ctx = await _memory.get_context(req.user_id, conv_id, query=req.message)

            # 2. 构建编排请求（含对话历史，用于意图识别上下文）
            history = [
                {"role": m.role.value, "content": m.content}
                for m in mem_ctx.recent_messages[-5:]
            ] if mem_ctx.recent_messages else None

            with trace.stage("intent.recognize"):
                intent_result = await _recognize_chat_intent(req.message, history)

            workflow_decision = None
            if not _should_handoff(req.message, intent_result):
                with trace.stage("workflow.decide"):
                    workflow_decision = await _decide_workflow(
                        req.message,
                        intent_result,
                        history,
                    )
                    if workflow_decision is not None:
                        trace._stages[-1].meta["decision"] = workflow_decision.to_dict()

            entities = _entities_with_resolved_order_id(intent_result, workflow_decision)
            if intent_result is not None and entities is not getattr(intent_result, "entities", None):
                intent_result.entities = entities

            with trace.stage("workflow.slot_check"):
                slot_assessment = _get_slot_manager().assess(
                    message=req.message,
                    intent=intent_result.intent if intent_result else None,
                    entities=entities,
                    decision=workflow_decision,
                )
                trace._stages[-1].meta["slot_check"] = slot_assessment.to_dict()

            with trace.stage("workflow.planner"):
                action_plan = _get_action_planner().plan(
                    message=req.message,
                    intent=intent_result.intent if intent_result else None,
                    entities=entities,
                    slot_assessment=slot_assessment,
                    intent_confidence=getattr(intent_result, "confidence", 0.0) if intent_result else 0.0,
                    intent_reasoning=getattr(intent_result, "reasoning", "") if intent_result else "",
                    decision=workflow_decision,
                )
                trace._stages[-1].meta["plan"] = action_plan.to_dict()

            with trace.stage("workflow.state_transition"):
                workflow_path = _get_state_machine().build_path(action_plan)
                trace._stages[-1].meta["path"] = workflow_path.to_dict()

            response_intent = intent_result.intent if intent_result else None
            response_agent_type = _agent_type_for_intent(response_intent, action_plan)
            response_escalated = action_plan.need_handoff

            if workflow_path.includes(WorkflowState.CLARIFY):
                response_text = action_plan.clarify_prompt or slot_assessment.clarify_question or "为了继续帮你处理，请补充关键信息。"
            else:
                context_parts = [_final_answer_memory_context(mem_ctx)]
                execution_result = await _execute_workflow_plan(
                    action_plan,
                    req,
                    conv_id,
                    mem_ctx,
                    intent_result,
                    trace=trace,
                )
                knowledge_used = execution_result.knowledge_used
                if execution_result.context_blocks:
                    context_parts.extend(execution_result.context_blocks)

                if execution_result.handoff_required:
                    failure_handoff_key = _make_idempotency_key(
                        operation="failure_handoff",
                        command_id=str(req.operation_id or conv_id),
                        user_id=req.user_id,
                        resource_id=conv_id,
                    )
                    handoff_result = await _maybe_handoff(
                        req,
                        conv_id,
                        mem_ctx,
                        intent_result,
                        None,
                        force=True,
                        trace=trace,
                        idempotency_key=failure_handoff_key,
                    )
                    response_escalated = True
                    if handoff_result and str(handoff_result.get("status", "")).lower() != "failed":
                        context_parts.append(_format_handoff_context(handoff_result))
                    else:
                        context_parts.append(
                            "[工作流执行状态]\n"
                            "- 退款请求的最终状态暂时无法确认，请不要重复提交。"
                        )

                if execution_result.user_input_required:
                    response_text = "当前操作状态暂时无法确认，请不要重复提交；请稍后重试或联系人工客服。"
                    response_escalated = response_escalated or action_plan.need_handoff
                    response_latency_ms = (time.monotonic() - t0) * 1000
                else:
                    full_context = "\n\n".join(part for part in context_parts if part)

                    orch_req = OrcReq(
                        message=req.message,
                        user_id=req.user_id,
                        conv_id=conv_id,
                        context=full_context,
                        history=history,
                        intent=intent_result.intent if intent_result else None,
                        urgency=intent_result.urgency if intent_result else None,
                    )

                    # 3. 执行
                    with trace.stage("orchestrator.run"):
                        result = await _run_planned_orchestrator(
                            orch_req,
                            action_plan,
                            trace=trace,
                            skill_snapshot=skill_snapshot,
                        )
                    response_text = result.response
                    if action_plan.follow_up_prompt:
                        response_text = f"{response_text.rstrip()}\n\n{action_plan.follow_up_prompt}"
                    if not action_plan.need_handoff:
                        response_agent_type = result.agent_type
                    response_intent = result.intent
                    response_escalated = response_escalated or result.escalated or action_plan.need_handoff
                    response_latency_ms = result.latency_ms

            # 4. 写入记忆
            with trace.stage("memory.write"):
                await _memory.add_message(req.user_id, conv_id, MsgRole.USER, req.message)
                await _memory.add_message(req.user_id, conv_id, MsgRole.ASSISTANT, response_text)

            # 5. 异步更新用户画像（不阻塞响应）
            with trace.stage("profile_update.schedule"):
                asyncio.create_task(_memory.update_profile(req.user_id, conv_id))
    except Exception as ex:
        trace.finalize(success=False, error=str(ex))
        trace_store.save(trace)
        raise

    trace.finalize(success=True)
    trace_store.save(trace)
    if response_latency_ms <= 0:
        response_latency_ms = (time.monotonic() - t0) * 1000

    return ChatResponse(
        conv_id=conv_id,
        response=response_text,
        intent=response_intent.value if response_intent else "other",
        agent_type=response_agent_type.value,
        escalated=response_escalated,
        latency_ms=round(response_latency_ms, 1),
        knowledge_used=knowledge_used,
        trace_id=trace.trace_id,
    )


async def _build_knowledge_context(
    message: str,
    top_k: Optional[int] = None,
    context: Optional[Dict[str, Any]] = None,
) -> tuple[str, bool]:
    """
    为 /chat 主链路构建 RAG 知识上下文。

    这里复用 MCPToolManager 的查询改写、并行召回、重排、fallback 能力。
    """
    if _tool_manager is None:
        return "", False
    if not _should_use_knowledge(message):
        return "", False
    resolved_top_k = _answer_knowledge_top_k() if top_k is None else top_k
    try:
        result = await _tool_manager.search_with_rewrite(
            "knowledge_search",
            message,
            top_k=resolved_top_k,
            context=context,
        )
        if not result.success or not isinstance(result.data, list) or not result.data:
            return "", False

        parts = ["[知识库检索结果]"]
        used = False
        prompt_blocks = []
        seen_parent_ids = set()
        for item in result.data[:resolved_top_k]:
            if not isinstance(item, dict):
                continue
            parent_id = str(item.get("parent_id", "")).strip()
            title = str(item.get("title", "未命名文档"))
            content = str(item.get("matched_child_content") or item.get("content", "")).strip()
            parent_content = str(item.get("parent_content", "")).strip()
            heading_path = str(item.get("heading_path", title))
            score = item.get("score", "")
            if not content:
                continue
            if parent_id and parent_id in seen_parent_ids:
                continue
            if parent_id:
                seen_parent_ids.add(parent_id)
            used = True
            i = len(prompt_blocks) + 1
            block = [
                f"{i}. 标题: {title}",
                f"   相关路径: {heading_path}",
                f"   相关度: {score}",
            ]
            block.append(f"   命中片段: {content[:320]}")
            if parent_content and parent_content != content:
                block.append(f"   所属段落: {parent_content[:420]}")
            prompt_block = "\n".join(block)
            prompt_blocks.append(prompt_block)
            parts.append(prompt_block)

        if not used:
            return "", False
        trace = (context or {}).get("trace")
        if trace is not None:
            for stage in reversed(trace._stages):
                if stage.name == "knowledge_context.build":
                    stage.meta["retrieved_contexts"] = prompt_blocks
                    break
        parts.append("请仅依据以上知识库内容回答；若信息不足，请明确说明知识库未提供该信息。")
        return "\n".join(parts), True
    except Exception as ex:
        logger.warning(f"构建知识库上下文失败: {ex}")
        return "", False


def _answer_knowledge_top_k() -> int:
    value = os.getenv("RAG_ANSWER_TOP_K", "5")
    try:
        top_k = int(value)
    except ValueError:
        logger.warning("RAG_ANSWER_TOP_K=%r 无效，使用默认值 5", value)
        return 5
    if top_k < 1:
        logger.warning("RAG_ANSWER_TOP_K=%r 必须为正数，使用默认值 5", value)
        return 5
    return top_k


def _final_answer_memory_context(mem_ctx: Any) -> str:
    """Keep cross-conversation memory out of answer prompts that may state live facts."""
    renderer = getattr(mem_ctx, "to_prompt_text", None)
    if renderer is None:
        return ""
    try:
        return renderer(include_cross_conversation=False)
    except TypeError:
        return renderer()


async def _recognize_chat_intent(message: str, history: Optional[List[Dict[str, str]]]):
    recognizer = _chat_intent_recognizer
    if recognizer is None and _orchestrator is not None:
        recognizer = getattr(_orchestrator, "_intent_recognizer", None)
    if recognizer is None:
        return None
    try:
        return await recognizer.recognize(message, history=history)
    except Exception as ex:
        logger.warning(f"chat 意图识别失败: {ex}")
        return None


async def _decide_workflow(message: str, intent_result: Any, history: Optional[List[Dict[str, str]]]):
    decider = _workflow_intent_decider
    if decider is None:
        return None
    try:
        return await decider.decide(
            message=message,
            intent=getattr(intent_result, "intent", None),
            entities=getattr(intent_result, "entities", {}) if intent_result else {},
            history=history,
        )
    except Exception as ex:
        logger.warning(f"workflow 语义决策失败: {ex}")
        return None


def _entities_with_resolved_order_id(intent_result: Any, decision: Any) -> Dict[str, List[str]]:
    entities = dict(getattr(intent_result, "entities", {}) or {}) if intent_result else {}
    order_ids = entities.get("order_id") or []
    if not isinstance(order_ids, list):
        order_ids = [order_ids]
    resolved_order_id = str(getattr(decision, "order_id", "") or "").strip()
    if resolved_order_id:
        entities["order_id"] = [resolved_order_id]
    else:
        entities["order_id"] = [str(order_id).strip() for order_id in order_ids if str(order_id).strip()]
    return entities


async def _execute_workflow_plan(
    plan: Any,
    req: ChatRequest,
    conv_id: str,
    mem_ctx: Any,
    intent_result: Any,
    trace: Any = None,
):
    from workflow.action_executor import ActionExecutor

    executor = ActionExecutor(_build_workflow_tool_registry())
    runtime = {
        "message": req.message,
        "req": req,
        "conv_id": conv_id,
        "mem_ctx": mem_ctx,
        "intent_result": intent_result,
        "trace": trace,
        "entities": getattr(intent_result, "entities", {}) if intent_result else {},
        "idempotency_keys": _workflow_idempotency_keys(plan, req, conv_id, intent_result),
    }
    if trace is None:
        result = await executor.execute(plan, runtime=runtime)
    else:
        with trace.stage("workflow.execute"):
            result = await executor.execute(plan, runtime=runtime)
        with trace.stage("workflow.verify"):
            trace._stages[-1].meta["evidence_keys"] = list(result.evidence_store.items.keys())
            trace._stages[-1].meta["failed_actions"] = list(result.failed_actions)
            trace._stages[-1].meta["degraded"] = result.degraded
            trace._stages[-1].meta["handoff_required"] = result.handoff_required
            trace._stages[-1].meta["user_input_required"] = result.user_input_required
    return result


def _build_workflow_tool_registry():
    from workflow.action_models import ActionType, EvidenceItem
    from workflow.tool_registry import ToolRegistry

    registry = ToolRegistry()

    async def _retrieve_policy(action, runtime, evidence_store):
        trace = runtime.get("trace")
        ctx = {"trace": trace} if trace is not None else None
        if trace is None:
            text, used = await _build_knowledge_context(runtime["message"], context=ctx)
        else:
            with trace.stage("knowledge_context.build"):
                text, used = await _build_knowledge_context(runtime["message"], context=ctx)
        if not used or not text:
            return None
        return EvidenceItem(
            key=action.output_key or "knowledge.policy",
            source="knowledge_search",
            value={"used": used, "text": text},
            prompt_block=text,
            tool_name="knowledge_search",
        )

    async def _lookup_order(action, runtime, evidence_store):
        payload = _require_successful_tool_payload(
            await _maybe_lookup_order(
                runtime["req"],
                runtime["intent_result"],
                force=True,
                trace=runtime.get("trace"),
            ),
            "order_lookup",
        )
        return EvidenceItem(
            key=action.output_key or "order.snapshot",
            source="order_lookup",
            value=payload,
            prompt_block=_format_order_lookup_context(payload),
            tool_name="order_lookup",
        )

    async def _track_shipment(action, runtime, evidence_store):
        payload = _require_successful_tool_payload(
            await _maybe_track_shipment(
                runtime["req"],
                runtime["intent_result"],
                force=True,
                trace=runtime.get("trace"),
            ),
            "shipment_track",
        )
        return EvidenceItem(
            key=action.output_key or "shipment.snapshot",
            source="shipment_track",
            value=payload,
            prompt_block=_format_shipment_context(payload),
            tool_name="shipment_track",
        )

    async def _create_refund(action, runtime, evidence_store):
        payload = _require_successful_tool_payload(
            await _maybe_create_refund(
                runtime["req"],
                runtime["intent_result"],
                force=True,
                trace=runtime.get("trace"),
                idempotency_key=(runtime.get("idempotency_keys") or {}).get(action.id, ""),
            ),
            "refund_create",
        )
        return EvidenceItem(
            key=action.output_key or "refund.result",
            source="refund_create",
            value=payload,
            prompt_block=_format_refund_create_context(payload),
            tool_name="refund_create",
        )

    async def _create_handoff(action, runtime, evidence_store):
        order_item = evidence_store.get("order.snapshot")
        payload = _require_successful_tool_payload(
            await _maybe_handoff(
                runtime["req"],
                runtime["conv_id"],
                runtime["mem_ctx"],
                runtime["intent_result"],
                order_item.value if order_item else None,
                force=True,
                trace=runtime.get("trace"),
                idempotency_key=(runtime.get("idempotency_keys") or {}).get(action.id, ""),
            ),
            "human_handoff",
        )
        return EvidenceItem(
            key=action.output_key or "handoff.result",
            source="human_handoff",
            value=payload,
            prompt_block=_format_handoff_context(payload),
            tool_name="human_handoff",
        )

    async def _synthesize(action, runtime, evidence_store):
        return EvidenceItem(
            key=action.output_key or "final.answer",
            source="workflow",
            value={"status": "ready", "input_keys": list(action.input_keys)},
        )

    registry.register(ActionType.RETRIEVE_POLICY, _retrieve_policy)
    registry.register(ActionType.RETRIEVE_FAQ, _retrieve_policy)
    registry.register(ActionType.LOOKUP_ORDER, _lookup_order)
    registry.register(ActionType.TRACK_SHIPMENT, _track_shipment)
    registry.register(ActionType.CREATE_REFUND, _create_refund)
    registry.register(ActionType.CREATE_HANDOFF, _create_handoff)
    registry.register(ActionType.SYNTHESIZE_ANSWER, _synthesize)
    registry.register(ActionType.SYNTHESIZE_MULTI_AGENT, _synthesize)
    registry.register(ActionType.CLARIFY_SLOT, _synthesize)
    registry.register(ActionType.DECIDE_HANDOFF, _synthesize)
    return registry


async def _run_planned_orchestrator(
    orch_req: Any,
    plan: Any,
    trace: Any = None,
    skill_snapshot: Any = None,
):
    from agents.agent_orchestrator import OrchestratorResult, Request as OrcReq

    route_plan = getattr(plan, "route_plan", None)
    skills_by_role = _build_agent_skills_context(skill_snapshot, plan, orch_req.intent)
    if trace is not None and trace._stages:
        trace._stages[-1].meta["skill_snapshot_generation"] = getattr(skill_snapshot, "generation", 0)
        trace._stages[-1].meta["skill_ids_by_role"] = {
            role: [str(skill.get("id", "")) for skill in skills]
            for role, skills in skills_by_role.items()
        }
    execution_context: Dict[str, Any] = {}
    if trace is not None:
        execution_context["trace"] = trace
    if skills_by_role:
        execution_context["agent_skills_by_role"] = skills_by_role
    context = execution_context or None
    if _orchestrator is None or route_plan is None:
        return await _orchestrator.run(orch_req, context=context)

    if not getattr(route_plan, "supporting_agents", []):
        primary_agent_type = _agent_type_from_role(getattr(route_plan, "final_writer", None) or route_plan.primary_agent)
        overall_t0 = time.monotonic()
        if trace is None:
            primary_response = await _orchestrator._execute(orch_req, primary_agent_type, context=context)
        else:
            with trace.stage("orchestrator.execute", agent_type=primary_agent_type.value):
                primary_response = await _orchestrator._execute(
                    orch_req,
                    primary_agent_type,
                    context=context,
                )

        escalated = bool(getattr(primary_response, "escalate", False))
        return OrchestratorResult(
            request_id=orch_req.request_id,
            response=primary_response.content,
            agent_type=primary_response.agent_type,
            intent=orch_req.intent,
            escalated=escalated,
            latency_ms=(time.monotonic() - overall_t0) * 1000,
        )

    overall_t0 = time.monotonic()
    supporting_agent_types = [_agent_type_from_role(agent) for agent in route_plan.supporting_agents]
    if getattr(route_plan, "merge_mode", None) and getattr(route_plan.merge_mode, "value", "") == "parallel_sections":
        agent_types = [_agent_type_from_role(route_plan.primary_agent)] + supporting_agent_types
        return await _orchestrator.run_parallel(orch_req, agent_types, context=context)

    primary_agent_type = _agent_type_from_role(getattr(route_plan, "final_writer", None) or route_plan.primary_agent)
    support_blocks = []
    escalated = False
    if trace is None:
        support_results = await asyncio.gather(
            *[
                _orchestrator._execute(_build_supporting_agent_request(orch_req, agent_type), agent_type, context=context)
                for agent_type in supporting_agent_types
            ],
            return_exceptions=True,
        )
    else:
        with trace.stage("orchestrator.supporting_agents"):
            support_results = await asyncio.gather(
                *[
                    _orchestrator._execute(
                        _build_supporting_agent_request(orch_req, agent_type),
                        agent_type,
                        context=context,
                    )
                    for agent_type in supporting_agent_types
                ],
                return_exceptions=True,
            )
    for result in support_results:
        if isinstance(result, Exception) or not getattr(result, "success", False):
            continue
        support_blocks.append(f"[{result.agent_type.value} 结构化意见]\n{result.content}")
        escalated = escalated or bool(getattr(result, "escalate", False))

    primary_context = orch_req.context
    if support_blocks:
        primary_context = "\n\n".join([orch_req.context, "[辅助专家结构化意见]", *support_blocks])

    primary_req = OrcReq(
        message=orch_req.message,
        user_id=orch_req.user_id,
        conv_id=orch_req.conv_id,
        context=primary_context,
        history=orch_req.history,
        intent=orch_req.intent,
        urgency=orch_req.urgency,
        request_id=orch_req.request_id,
    )

    if trace is None:
        primary_response = await _orchestrator._execute(primary_req, primary_agent_type, context=context)
    else:
        with trace.stage("orchestrator.primary_summarize", primary_agent=primary_agent_type.value):
            primary_response = await _orchestrator._execute(primary_req, primary_agent_type, context=context)

    escalated = escalated or bool(getattr(primary_response, "escalate", False))
    return OrchestratorResult(
        request_id=primary_req.request_id,
        response=primary_response.content,
        agent_type=primary_response.agent_type,
        intent=primary_req.intent,
        escalated=escalated,
        latency_ms=(time.monotonic() - overall_t0) * 1000,
    )


def _build_supporting_agent_request(orch_req: Any, agent_type: Any):
    from agents.agent_orchestrator import Request as OrcReq

    role_name = getattr(agent_type, "value", str(agent_type))
    structured_prompt = (
        f"你现在是 {role_name} 辅助专家，不直接面向用户回复，也不要给寒暄。"
        "请仅基于当前问题和背景信息，输出结构化辅助意见，严格使用以下四段标题："
        "[问题判断]、[关键事实]、[风险与限制]、[建议动作]。"
        "只写你本专业负责的判断，不要越权承诺操作。"
        f"\n\n用户原始问题：{orch_req.message}"
    )
    return OrcReq(
        message=structured_prompt,
        user_id=orch_req.user_id,
        conv_id=orch_req.conv_id,
        context=orch_req.context,
        history=orch_req.history,
        intent=orch_req.intent,
        urgency=orch_req.urgency,
        request_id=orch_req.request_id,
    )


async def _build_tool_context(
    req: ChatRequest,
    conv_id: str,
    mem_ctx: Any,
    intent_result: Any,
    plan: Any = None,
    trace: Any = None,
) -> str:
    if _tool_manager is None or intent_result is None:
        return ""

    called_tools: List[str] = []
    skipped_tools: List[str] = []
    sections: List[str] = []

    order_snapshot = None
    if plan is not None and getattr(plan, "need_order_lookup", False):
        order_snapshot = await _maybe_lookup_order(req, intent_result, force=True, trace=trace)
        if order_snapshot is not None:
            called_tools.append("order_lookup")
            sections.append(_format_order_lookup_context(order_snapshot))
        else:
            skipped_tools.append("order_lookup")
    else:
        skipped_tools.append("order_lookup")

    if plan is not None and getattr(plan, "need_handoff", False):
        handoff_result = await _maybe_handoff(
            req,
            conv_id,
            mem_ctx,
            intent_result,
            order_snapshot,
            force=True,
            trace=trace,
            idempotency_key=_make_idempotency_key(
                operation="create_handoff",
                command_id=str(req.operation_id or conv_id),
                user_id=req.user_id,
                resource_id=conv_id,
            ),
        )
        if handoff_result is not None:
            called_tools.append("human_handoff")
            sections.append(_format_handoff_context(handoff_result))
        else:
            skipped_tools.append("human_handoff")
    else:
        skipped_tools.append("human_handoff")

    if not sections:
        return ""

    summary = ["[工具增强上下文]", "", "[工具执行摘要]"]
    summary.append(f"- 已调用工具: {', '.join(called_tools) if called_tools else '无'}")
    summary.append(f"- 未调用工具: {', '.join(skipped_tools) if skipped_tools else '无'}")
    summary.append("- 事实优先级: 实时订单/物流/退款结果优先于通用知识说明；人工转接结果优先于推测性答复")
    return "\n".join(summary + [""] + sections)


def _looks_like_order_query(message: str) -> bool:
    msg = (message or "").lower()
    keywords = [
        "订单", "物流", "发货", "配送", "到哪", "进度", "状态", "快递",
        "order", "shipment", "delivery", "tracking", "status",
    ]
    return any(kw in msg for kw in keywords)


def _extract_order_id(intent_result: Any) -> str:
    entities = getattr(intent_result, "entities", {}) or {}
    order_ids = entities.get("order_id") or []
    if not order_ids:
        return ""
    return str(order_ids[0]).strip()


def _extract_refund_reason(message: str) -> str:
    msg = (message or "").strip()
    markers = ("原因是", "因为", "理由是")
    for marker in markers:
        if marker in msg:
            return msg.split(marker, 1)[1].strip(" ：:，,。")
    return ""


def _make_idempotency_key(
    *,
    operation: str,
    command_id: str,
    user_id: str,
    resource_id: str,
) -> str:
    material = "\n".join([operation, command_id, user_id, resource_id]).encode("utf-8")
    digest = hashlib.sha256(material).hexdigest()
    return f"echomind-{operation}-v1-{digest}"


def _workflow_idempotency_keys(
    plan: Any,
    req: ChatRequest,
    conv_id: str,
    intent_result: Any,
) -> Dict[str, str]:
    command_id = str(req.operation_id or conv_id).strip()
    if not command_id:
        return {}
    keys: Dict[str, str] = {}
    order_id = _extract_order_id(intent_result)
    for action in getattr(plan, "actions", []):
        action_type = getattr(getattr(action, "type", None), "value", "")
        if action_type == "create_refund":
            resource_id = order_id or "refund"
        elif action_type == "create_handoff":
            resource_id = conv_id
        else:
            continue
        keys[action.id] = _make_idempotency_key(
            operation=action_type,
            command_id=command_id,
            user_id=req.user_id,
            resource_id=resource_id,
        )
    return keys


def _looks_like_shipment_query(message: str) -> bool:
    msg = (message or "").lower()
    keywords = ["物流", "快递", "配送", "发货", "运输", "tracking", "shipment", "delivery"]
    return any(kw in msg for kw in keywords)


def _looks_like_refund_create(message: str) -> bool:
    msg = (message or "").lower()
    action_keywords = ["我要退款", "申请退款", "帮我退款", "退款一下", "给我退款", "直接退款"]
    return any(kw in msg for kw in action_keywords)


async def _call_business_tool(
    tool_name: str,
    params: Dict[str, Any],
    trace: Any = None,
) -> Optional[Any]:
    try:
        stage = trace.stage(f"tool.{tool_name}") if trace is not None else None
        if stage is None:
            result = await _tool_manager.call(tool_name, params)
        else:
            with stage:
                result = await _tool_manager.call(tool_name, params, context={"trace": trace})
    except Exception as ex:
        logger.warning(f"{tool_name} 调用失败: {ex}")
        return {"status": "failed", "error": str(ex), **params}

    if not result.success or not isinstance(result.data, dict):
        return {"status": "failed", "error": result.error or "tool_failed", **params}
    return result.data


def _require_successful_tool_payload(payload: Optional[Dict[str, Any]], tool_name: str) -> Dict[str, Any]:
    if payload is None:
        raise RuntimeError(f"{tool_name} 未返回结果")
    if str(payload.get("status", "")).strip().lower() == "failed":
        raise RuntimeError(str(payload.get("error", "tool_failed")))
    return payload


async def _maybe_lookup_order(
    req: ChatRequest,
    intent_result: Any,
    force: bool = False,
    trace: Any = None,
) -> Optional[Dict[str, Any]]:
    order_id = _extract_order_id(intent_result)
    if not order_id:
        return None
    if not force and not _looks_like_order_query(req.message):
        return None
    return await _call_business_tool(
        "order_lookup",
        {"user_id": req.user_id, "order_id": order_id},
        trace=trace,
    )


async def _maybe_track_shipment(
    req: ChatRequest,
    intent_result: Any,
    force: bool = False,
    trace: Any = None,
) -> Optional[Dict[str, Any]]:
    order_id = _extract_order_id(intent_result)
    if not order_id:
        return None
    if not force and not _looks_like_shipment_query(req.message):
        return None
    return await _call_business_tool(
        "shipment_track",
        {"user_id": req.user_id, "order_id": order_id},
        trace=trace,
    )


async def _maybe_create_refund(
    req: ChatRequest,
    intent_result: Any,
    force: bool = False,
    trace: Any = None,
    idempotency_key: str = "",
) -> Optional[Dict[str, Any]]:
    order_id = _extract_order_id(intent_result)
    if not order_id:
        return None
    if not force and not _looks_like_refund_create(req.message):
        return None
    return await _call_business_tool(
        "refund_create",
        {
            "user_id": req.user_id,
            "order_id": order_id,
            "reason": _extract_refund_reason(req.message),
            "idempotency_key": idempotency_key,
        },
        trace=trace,
    )


def _should_handoff(message: str, intent_result: Any) -> bool:
    intent = getattr(intent_result, "intent", None)
    if intent is not None and getattr(intent, "value", "") == "escalation":
        return True
    msg = (message or "").lower()
    handoff_keywords = ["转人工", "人工客服", "客服专员", "找人工", "人工处理", "投诉专员"]
    return any(kw in msg for kw in handoff_keywords)


async def _maybe_handoff(
    req: ChatRequest,
    conv_id: str,
    mem_ctx: Any,
    intent_result: Any,
    order_snapshot: Optional[Dict[str, Any]],
    force: bool = False,
    trace: Any = None,
    idempotency_key: str = "",
) -> Optional[Dict[str, Any]]:
    if not force and not _should_handoff(req.message, intent_result):
        return None

    recent_messages = []
    for item in getattr(mem_ctx, "recent_messages", []) or []:
        role = getattr(getattr(item, "role", None), "value", "")
        content = getattr(item, "content", "")
        if role and content:
            recent_messages.append(f"{role}: {content}")

    params = {
        "user_id": req.user_id,
        "conv_id": conv_id,
        "latest_message": req.message,
        "intent": getattr(getattr(intent_result, "intent", None), "value", ""),
        "urgency": getattr(getattr(intent_result, "urgency", None), "name", "").lower(),
        "reason": "用户明确要求转人工",
        "summary": getattr(mem_ctx, "summary", ""),
        "recent_messages": recent_messages,
        "user_profile": getattr(mem_ctx, "user_profile", {}) or {},
        "order_snapshot": order_snapshot or {},
        "knowledge_context": [],
        "idempotency_key": idempotency_key,
    }

    try:
        stage = trace.stage("tool.human_handoff") if trace is not None else None
        if stage is None:
            result = await _tool_manager.call("human_handoff", params)
        else:
            with stage:
                result = await _tool_manager.call("human_handoff", params, context={"trace": trace})
    except Exception as ex:
        logger.warning(f"human_handoff 调用失败: {ex}")
        return {"status": "failed", "error": str(ex)}

    if not result.success or not isinstance(result.data, dict):
        return {"status": "failed", "error": result.error or "tool_failed"}
    return result.data


def _format_order_lookup_context(payload: Dict[str, Any]) -> str:
    lines = ["[order_lookup]"]
    status = str(payload.get("status", "")).strip() or "unknown"
    lines.append(f"- 状态: {status}")
    order_id = str(payload.get("order_id", "")).strip()
    if order_id:
        lines.append(f"- 订单号: {order_id}")
    payment_status = str(payload.get("payment_status", "")).strip()
    if payment_status:
        lines.append(f"- 支付状态: {payment_status}")
    shipment_status = str(payload.get("shipment_status", "")).strip()
    if shipment_status:
        lines.append(f"- 物流状态: {shipment_status}")
    refund_status = str(payload.get("refund_status", "")).strip()
    if refund_status:
        lines.append(f"- 退款状态: {refund_status}")
    updated_at = str(payload.get("updated_at", "")).strip()
    if updated_at:
        lines.append(f"- 更新时间: {updated_at}")
    error = str(payload.get("error", "")).strip()
    if error:
        lines.append(f"- 错误信息: {error}")
    return "\n".join(lines)


def _format_shipment_context(payload: Dict[str, Any]) -> str:
    lines = ["[shipment_track]"]
    shipment_status = str(payload.get("shipment_status", "")).strip() or "unknown"
    lines.append(f"- 物流状态: {shipment_status}")
    order_id = str(payload.get("order_id", "")).strip()
    if order_id:
        lines.append(f"- 订单号: {order_id}")
    carrier = str(payload.get("carrier", "")).strip()
    if carrier:
        lines.append(f"- 承运商: {carrier}")
    tracking_no = str(payload.get("tracking_no", "")).strip()
    if tracking_no:
        lines.append(f"- 运单号: {tracking_no}")
    events = payload.get("events") or []
    if events:
        latest = events[-1]
        lines.append(f"- 最新轨迹: {latest.get('time', '')} {latest.get('status', '')}".strip())
    error = str(payload.get("error", "")).strip()
    if error:
        lines.append(f"- 错误信息: {error}")
    return "\n".join(lines)


def _format_refund_create_context(payload: Dict[str, Any]) -> str:
    lines = ["[refund_create]"]
    status = str(payload.get("status", "")).strip() or "unknown"
    lines.append(f"- 申请状态: {status}")
    refund_id = str(payload.get("refund_id", "")).strip()
    if refund_id:
        lines.append(f"- 退款单号: {refund_id}")
    order_id = str(payload.get("order_id", "")).strip()
    if order_id:
        lines.append(f"- 订单号: {order_id}")
    amount = str(payload.get("amount", "")).strip()
    if amount:
        lines.append(f"- 退款金额: {amount}")
    reason = str(payload.get("reason", "")).strip()
    if reason:
        lines.append(f"- 退款原因: {reason}")
    submitted_at = str(payload.get("submitted_at", "")).strip()
    if submitted_at:
        lines.append(f"- 提交时间: {submitted_at}")
    error = str(payload.get("error", "")).strip()
    if error:
        lines.append(f"- 错误信息: {error}")
    return "\n".join(lines)


def _format_handoff_context(payload: Dict[str, Any]) -> str:
    lines = ["[human_handoff]"]
    status = str(payload.get("status", "")).strip() or "unknown"
    lines.append(f"- 状态: {status}")
    handoff_id = str(payload.get("handoff_id", "")).strip()
    if handoff_id:
        lines.append(f"- handoff_id: {handoff_id}")
    queue = str(payload.get("queue", "")).strip()
    if queue:
        lines.append(f"- 队列: {queue}")
    eta = payload.get("eta_minutes")
    if eta not in (None, ""):
        lines.append(f"- 预计等待分钟数: {eta}")
    error = str(payload.get("error", "")).strip()
    if error:
        lines.append(f"- 错误信息: {error}")
    return "\n".join(lines)


def _should_use_knowledge(message: str) -> bool:
    """跳过纯寒暄，业务类问题才检索知识库，避免无关 RAG 干扰回复。"""
    msg = (message or "").strip().lower()
    if not msg:
        return False
    greetings = {"你好", "您好", "嗨", "hi", "hello", "hey", "早上好", "晚上好"}
    if msg in greetings:
        return False
    business_keywords = [
        "退款", "订单", "物流", "配送", "发票", "扣款", "支付", "账单", "订阅",
        "登录", "报错", "错误", "崩溃", "会员", "积分", "账户", "密码", "地址",
        "refund", "order", "invoice", "payment", "error", "login",
    ]
    return len(msg) >= 4 or any(kw in msg for kw in business_keywords)


def _get_slot_manager():
    global _slot_manager
    if _slot_manager is None:
        from workflow.slot_manager import SlotManager
        _slot_manager = SlotManager()
    return _slot_manager


def _get_action_planner():
    global _action_planner
    if _action_planner is None:
        from workflow.action_planner import ActionPlanner
        _action_planner = ActionPlanner(slot_manager=_get_slot_manager())
    return _action_planner


def _get_state_machine():
    global _state_machine
    if _state_machine is None:
        from workflow.state_machine import WorkflowStateMachine
        _state_machine = WorkflowStateMachine(max_action_steps=3)
    return _state_machine


def _agent_type_for_intent(intent: Any, action_plan: Any):
    from agents.agent_orchestrator import AgentType
    from core.intent_recognizer import IntentCategory

    route_plan = getattr(action_plan, "route_plan", None)
    if route_plan is not None and getattr(route_plan, "primary_agent", None):
        return _agent_type_from_role(route_plan.primary_agent)
    if getattr(action_plan, "need_handoff", False):
        return AgentType.ESCALATION
    if intent == IntentCategory.TECHNICAL:
        return AgentType.TECHNICAL
    if intent in {IntentCategory.BILLING, IntentCategory.ACCOUNT}:
        return AgentType.BILLING
    return AgentType.GENERAL


def _build_agent_skills_context(snapshot: Any, action_plan: Any, intent: Any) -> Dict[str, List[Dict[str, Any]]]:
    if _skill_runtime is None or snapshot is None:
        return {}
    route_plan = getattr(action_plan, "route_plan", None)
    if route_plan is None:
        return {}

    roles = [getattr(route_plan.primary_agent, "value", str(route_plan.primary_agent))]
    roles.extend(getattr(agent, "value", str(agent)) for agent in getattr(route_plan, "supporting_agents", []))
    final_writer = getattr(route_plan, "final_writer", None)
    if final_writer is not None:
        roles.append(getattr(final_writer, "value", str(final_writer)))
    planned_tools = [
        str(getattr(action, "tool_name", "") or "")
        for action in getattr(action_plan, "actions", [])
        if getattr(action, "tool_name", "")
    ]
    intent_category = getattr(intent, "value", "other")
    goal = str(getattr(action_plan, "primary_goal", "") or "")
    context: Dict[str, List[Dict[str, Any]]] = {}
    for role in dict.fromkeys(roles):
        skills = _skill_runtime.select_from_snapshot(
            snapshot,
            agent_role=role,
            intent_category=intent_category,
            goal=goal,
            planned_tools=planned_tools,
        )
        context[role] = [
            {
                "id": skill.id,
                "version": skill.version,
                "prompt": skill.prompt,
                "allowed_tools": list(skill.allowed_tools),
            }
            for skill in skills
        ]
    return context


def _agent_type_from_role(role: Any):
    from agents.agent_orchestrator import AgentType

    value = getattr(role, "value", str(role))
    mapping = {
        "general": AgentType.GENERAL,
        "technical": AgentType.TECHNICAL,
        "billing": AgentType.BILLING,
        "escalation": AgentType.ESCALATION,
    }
    return mapping.get(value, AgentType.GENERAL)


@app.get("/monitor")
async def monitor_summary():
    """实时监控摘要：Agent 成功率、工具统计、告警、优化建议。"""
    if _monitor is None:
        raise HTTPException(503, "服务未就绪")
    return _monitor.summary()


@app.get("/traces/recent", tags=["Trace"])
async def recent_traces(limit: int = 20):
    store = _ensure_trace_store()
    return {"items": store.recent(limit=limit)}


@app.get("/traces/{trace_id}", tags=["Trace"])
async def get_trace(trace_id: str):
    store = _ensure_trace_store()
    payload = store.get(trace_id)
    if payload is None:
        raise HTTPException(404, "trace 不存在")
    return payload


def _get_owned_mock_record(store: Dict[str, Dict[str, Any]], resource_id: str, user_id: str) -> Dict[str, Any]:
    record = store.get(resource_id)
    if record is None or str(record.get("user_id", "")).strip() != str(user_id).strip():
        raise HTTPException(404, "资源不存在")
    return record


def _mock_idempotency_replay(
    operation: str,
    idempotency_key: Optional[str],
    request_payload: Dict[str, Any],
) -> Optional[Dict[str, Any]]:
    key = str(idempotency_key or "").strip()
    if not key:
        return None
    existing = _mock_idempotency_records.get(f"{operation}:{key}")
    if existing is None:
        return None
    expected_payload, result = existing
    if expected_payload != request_payload:
        raise HTTPException(409, "幂等键已用于不同请求")
    return dict(result)


def _remember_mock_idempotency(
    operation: str,
    idempotency_key: Optional[str],
    request_payload: Dict[str, Any],
    result: Dict[str, Any],
) -> None:
    key = str(idempotency_key or "").strip()
    if key:
        _mock_idempotency_records[f"{operation}:{key}"] = (dict(request_payload), dict(result))


@app.get("/mock/external/orders/{order_id}", tags=["Mock External"])
async def mock_external_order_lookup(order_id: str, user_id: str):
    record = _get_owned_mock_record(_MOCK_ORDER_DATA, order_id, user_id)
    return dict(record)


@app.get("/mock/external/shipments/{order_id}", tags=["Mock External"])
async def mock_external_shipment_track(order_id: str, user_id: str):
    record = _get_owned_mock_record(_MOCK_SHIPMENT_DATA, order_id, user_id)
    return dict(record)


class MockRefundCreateInput(BaseModel):
    user_id: str
    order_id: str
    reason: Optional[str] = ""


@app.post("/mock/external/refunds", tags=["Mock External"])
async def mock_external_refund_create(
    body: MockRefundCreateInput,
    idempotency_key: Optional[str] = Header(default=None, alias="Idempotency-Key"),
):
    order_record = _get_owned_mock_record(_MOCK_ORDER_DATA, body.order_id, body.user_id)
    request_payload = body.model_dump()
    replay = _mock_idempotency_replay("refund", idempotency_key, request_payload)
    if replay is not None:
        return replay
    existing = _mock_refund_records.get(body.order_id)
    if existing is not None and existing.get("user_id") == body.user_id:
        result = dict(existing)
        _remember_mock_idempotency("refund", idempotency_key, request_payload, result)
        return result

    submitted_at = datetime.now(timezone.utc).astimezone().isoformat()
    record = {
        "refund_id": f"RF{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S%f')}",
        "order_id": body.order_id,
        "user_id": body.user_id,
        "status": "submitted",
        "amount": order_record.get("amount", ""),
        "reason": body.reason or "",
        "submitted_at": submitted_at,
        "source": "mock_refund",
    }
    _mock_refund_records[body.order_id] = record

    order_record["refund_status"] = "退款申请已提交"
    order_record["updated_at"] = submitted_at
    result = dict(record)
    _remember_mock_idempotency("refund", idempotency_key, request_payload, result)
    return result


class MockHandoffInput(BaseModel):
    user_id: str
    conv_id: str
    latest_message: str
    intent: Optional[str] = ""
    urgency: Optional[str] = ""
    reason: Optional[str] = ""
    summary: Optional[str] = ""
    recent_messages: List[str] = []
    user_profile: Dict[str, Any] = {}
    order_snapshot: Dict[str, Any] = {}
    knowledge_context: List[str] = []


@app.post("/mock/external/handoffs", tags=["Mock External"])
async def mock_external_handoff(
    body: MockHandoffInput,
    idempotency_key: Optional[str] = Header(default=None, alias="Idempotency-Key"),
):
    request_payload = body.model_dump()
    replay = _mock_idempotency_replay("handoff", idempotency_key, request_payload)
    if replay is not None:
        return replay
    record = body.model_dump()
    record["handoff_id"] = f"HD{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S%f')}"
    record["queue"] = "general_support"
    record["status"] = "created"
    record["eta_minutes"] = 8
    _mock_handoff_records.append(record)
    result = {
        "handoff_id": record["handoff_id"],
        "queue": record["queue"],
        "status": record["status"],
        "eta_minutes": record["eta_minutes"],
    }
    _remember_mock_idempotency("handoff", idempotency_key, request_payload, result)
    return result


@app.get("/metrics")
async def prometheus_metrics():
    """Prometheus 指标入口。"""
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.post("/search")
async def search(
    query: str,
    top_k: int = 5,
    recall_k: Optional[int] = None,
    include_debug: bool = False,
):
    """
    演示检索优化链路：查询改写 → 并行召回 → 重排 → Top-K。
    展示 MCP 工具调用的核心亮点。
    """
    if _tool_manager is None:
        raise HTTPException(503, "服务未就绪")
    result = await _tool_manager.search_with_rewrite(
        "knowledge_search",
        query,
        top_k=top_k,
        recall_k=recall_k,
    )
    results = result.data if isinstance(result.data, list) else []
    rerank_providers = sorted({
        str(item.get("rerank_provider"))
        for item in results
        if isinstance(item, dict) and item.get("rerank_provider")
    })
    rerank_applied = bool(results) and all(
        isinstance(item, dict) and item.get("rerank_applied") is True
        for item in results
    )
    payload = {
        "query": query,
        "results": results,
        "reranked": result.reranked,
        "rerank_applied": rerank_applied,
        "rerank_providers": rerank_providers,
        "top_k": top_k,
        "recall_k": recall_k,
    }
    if include_debug:
        payload["retrieval_debug"] = result.retrieval_debug
    return payload


class DocInput(BaseModel):
    """单篇文档输入。"""
    title: str
    content: str


class BatchDocInput(BaseModel):
    """批量文档导入请求体。"""
    documents: List[DocInput]


class EvalIntentInput(BaseModel):
    """意图识别评测用例。"""
    message: str
    expected_intent: str
    context: Optional[Dict[str, Any]] = None


class EvalDialogInput(BaseModel):
    """对话质量评测用例。question 单轮，turns 多轮。"""
    question: Optional[str] = None
    turns: Optional[List[str]] = None
    user_id: Optional[str] = None
    conv_id: Optional[str] = None


class EvalRunInput(BaseModel):
    """评测请求。为空时使用内置默认用例。"""
    intent_cases: Optional[List[EvalIntentInput]] = None
    dialog_cases: Optional[List[EvalDialogInput]] = None


@app.post("/knowledge/add", tags=["知识库"])
async def add_knowledge(body: BatchDocInput):
    """
    批量导入文档到知识库。

    文档会先切为 300 字父块，再切为 100 字子块并存入 ChromaDB。

    示例请求体：
    ```json
    {
      "documents": [
        {"title": "退款政策", "content": "用户在购买后 7 天内可以申请无理由退款..."},
        {"title": "配送说明", "content": "标准配送 3-5 个工作日..."}
      ]
    }
    ```
    """
    tool = _tool_manager._tools.get("knowledge_search") if _tool_manager else None
    if tool is None:
        raise HTTPException(503, "知识库未初始化")
    kb = tool.handler.__self__
    count = kb.add_documents([
        {
            "title": d.title,
            "content": d.content,
        }
        for d in body.documents
    ])
    return {"message": f"成功导入 {count} 个文档片段", "added_chunks": count, "total_chunks": kb.doc_count}


@app.post("/knowledge/replace", tags=["知识库"])
async def replace_knowledge(body: BatchDocInput):
    """Replace all active knowledge documents and rebuild parent, child, and BM25 indexes."""
    tool = _tool_manager._tools.get("knowledge_search") if _tool_manager else None
    if tool is None:
        raise HTTPException(503, "知识库未初始化")
    if not body.documents:
        raise HTTPException(400, "至少需要提供一篇文档")
    kb = tool.handler.__self__
    count = kb.replace_documents([
        {
            "title": document.title,
            "content": document.content,
        }
        for document in body.documents
    ])
    return {
        "message": f"已替换知识库并重建 {count} 个文档片段",
        "added_chunks": count,
        "total_chunks": kb.doc_count,
    }


@app.post("/knowledge/upload", tags=["知识库"])
async def upload_knowledge(file: UploadFile = File(...)):
    """
    上传文件导入知识库。

    支持格式：
    - `.txt` / `.md`：整个文件作为一篇文档，文件名作为标题
    - `.json`：JSON 数组格式 `[{"title": "...", "content": "..."}, ...]`

    文件大小限制：10MB
    """
    tool = _tool_manager._tools.get("knowledge_search") if _tool_manager else None
    if tool is None:
        raise HTTPException(503, "知识库未初始化")
    kb = tool.handler.__self__

    content = await file.read()
    if len(content) > 10 * 1024 * 1024:
        raise HTTPException(413, "文件大小超过 10MB 限制")

    text = content.decode("utf-8", errors="ignore")
    filename = file.filename or "unknown"

    if filename.endswith(".json"):
        import json as _json
        try:
            docs = _json.loads(text)
            if not isinstance(docs, list):
                raise HTTPException(400, "JSON 文件应为数组格式: [{title, content}, ...]")
        except _json.JSONDecodeError as e:
            raise HTTPException(400, f"JSON 解析失败: {e}")
    else:
        # txt / md：整个文件作为一篇文档
        title = filename.rsplit(".", 1)[0] if "." in filename else filename
        docs = [{"title": title, "content": text}]

    count = kb.add_documents(docs)
    return {
        "message": f"文件 {filename} 导入成功",
        "added_chunks": count,
        "total_chunks": kb.doc_count,
    }


@app.get("/knowledge/stats", tags=["知识库"])
async def knowledge_stats():
    """查看知识库统计信息（文档片段总数）。"""
    tool = _tool_manager._tools.get("knowledge_search") if _tool_manager else None
    if tool is None:
        raise HTTPException(503, "知识库未初始化")
    kb = tool.handler.__self__
    return {"total_chunks": kb.doc_count}


@app.get("/knowledge/chunks", tags=["知识库"])
async def knowledge_chunks(limit: int = 1000, offset: int = 0):
    """导出切分后的 child chunk 清单，供 RAG 标注与检索评测使用。"""
    tool = _tool_manager._tools.get("knowledge_search") if _tool_manager else None
    if tool is None:
        raise HTTPException(503, "知识库未初始化")
    kb = tool.handler.__self__

    safe_limit = max(1, min(limit, 5000))
    safe_offset = max(0, offset)
    records = getattr(kb, "_child_records", []) or []

    items = []
    for record in records[safe_offset:safe_offset + safe_limit]:
        if not isinstance(record, dict):
            continue
        parent_id = str(record.get("parent_id", "") or "")
        child_chunk_index = int(record.get("child_chunk_index", record.get("chunk_index", 0)) or 0)
        items.append({
            "chunk_key": f"{parent_id}:{child_chunk_index}" if parent_id else "",
            "title": str(record.get("title", "") or ""),
            "doc_id": str(record.get("doc_id", "") or ""),
            "parent_id": parent_id,
            "chunk_index": int(record.get("chunk_index", 0) or 0),
            "child_chunk_index": child_chunk_index,
            "heading_path": str(record.get("heading_path", "") or ""),
            "section_title": str(record.get("section_title", "") or ""),
            "content": str(record.get("content", "") or ""),
        })

    return {
        "scope": "active",
        "total": len(records),
        "offset": safe_offset,
        "limit": safe_limit,
        "items": items,
    }


@app.post("/eval/run")
async def run_eval(body: Optional[EvalRunInput] = None):
    """运行内置评测用例，返回评测报告。"""
    if _evaluator is None:
        raise HTTPException(503, "服务未就绪")
    from evaluation.evaluator import DEFAULT_DIALOG_CASES, DEFAULT_INTENT_CASES, IntentTestCase

    if body and body.intent_cases is not None:
        intent_cases = [
            IntentTestCase(
                message=c.message,
                expected_intent=c.expected_intent,
                context=c.context,
            )
            for c in body.intent_cases
        ]
    else:
        intent_cases = DEFAULT_INTENT_CASES

    if body and body.dialog_cases is not None:
        dialog_cases = [
            c.model_dump(exclude_none=True)
            for c in body.dialog_cases
        ]
    else:
        dialog_cases = DEFAULT_DIALOG_CASES

    report = await _evaluator.run(
        intent_cases=intent_cases,
        dialog_cases=dialog_cases,
    )
    return {
        "pass_rate":       report.pass_rate,
        "total":           report.total,
        "passed":          report.passed,
        "avg_scores":      report.avg_scores,
        "regressions":     report.regressions,
        "recommendations": report.recommendations,
        "results": [
            {
                "test_id": r.test_id,
                "passed": r.passed,
                "scores": r.scores,
                "detail": r.detail,
                "metadata": r.metadata,
            }
            for r in report.results
        ],
    }


# ── 交互式 CLI ────────────────────────────────────────────────────────────────
async def _cli():
    print(BANNER)
    print("EchoMind CLI — 输入 quit 退出\n")

    from agents.agent_orchestrator import AgentOrchestrator, Request
    from memory.conversation_memory import MemoryManager, MsgRole

    cfg = _anthropic_cfg()
    orch = AgentOrchestrator(api_key=cfg["api_key"], base_url=cfg.get("base_url"), model=cfg["model"])
    mem  = MemoryManager(
        redis_url=os.getenv("REDIS_URL", "redis://localhost:6379/0"),
        chroma_host=os.getenv("CHROMA_HOST", "localhost"),
        chroma_port=int(os.getenv("CHROMA_PORT", "8000")),
        chroma_path=os.getenv("CHROMA_PERSIST_DIRECTORY", "/tmp/chroma"),
        api_key=cfg["api_key"],
        base_url=cfg.get("base_url"),
        model=cfg["model"],
    )

    user_id, conv_id = "cli_user", str(uuid.uuid4())

    while True:
        try:
            msg = input("你: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n再见 ʕ•ᴥ•ʔ")
            break
        if not msg or msg.lower() in ("quit", "exit", "退出"):
            print("再见 ʕ•ᴥ•ʔ")
            break

        ctx = await mem.get_context(user_id, conv_id, query=msg)
        history = [
            {"role": m.role.value, "content": m.content}
            for m in ctx.recent_messages[-5:]
        ] if ctx.recent_messages else None
        req = Request(message=msg, user_id=user_id, conv_id=conv_id, context=ctx.to_prompt_text(), history=history)
        result = await orch.run(req)

        await mem.add_message(user_id, conv_id, MsgRole.USER, msg)
        await mem.add_message(user_id, conv_id, MsgRole.ASSISTANT, result.response)

        print(f"\nEchoMind [{result.agent_type.value}]: {result.response}\n")


if __name__ == "__main__":
    if "--cli" in sys.argv:
        asyncio.run(_cli())
    else:
        uvicorn.run(
            "api.main:app",
            host=os.getenv("API_HOST", "0.0.0.0"),
            port=int(os.getenv("API_PORT", "8000")),
            reload=os.getenv("APP_ENV") == "development",
        )
