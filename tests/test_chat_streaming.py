import json
import sys
import types
import unittest
from dataclasses import dataclass
from enum import Enum

from fastapi.testclient import TestClient

from api import main as api_main
from agents.agent_orchestrator import AgentResponse, AgentType, OrchestratorResult
from core.intent_recognizer import IntentCategory, UrgencyLevel
from workflow.intent_decider import DecisionMode, WorkflowDecision


@dataclass
class _FakeIntentResult:
    intent: IntentCategory
    confidence: float
    urgency: UrgencyLevel
    entities: dict
    reasoning: str
    latency_ms: float


class _FakeRecognizer:
    def __init__(self, result: _FakeIntentResult):
        self._result = result

    async def recognize(self, message: str, history=None):
        return self._result


class _FakeWorkflowDecider:
    def __init__(self, decision):
        self._decision = decision

    async def decide(self, message, intent, entities, history):
        return self._decision


class _FakeMemoryContext:
    def __init__(self):
        self.summary = "用户最近咨询退款。"
        self.relevant_history = []
        self.user_profile = {"tier": "vip"}
        self.recent_messages = []

    def to_prompt_text(self) -> str:
        return "[会话摘要]\n用户最近咨询退款。"


class _FakeMemory:
    def __init__(self):
        self.added = []
        self.updated = []

    async def get_context(self, user_id: str, conv_id: str, query: str):
        return _FakeMemoryContext()

    async def add_message(self, user_id, conv_id, role, content):
        self.added.append((user_id, conv_id, role.value, content))

    async def update_profile(self, user_id, conv_id):
        self.updated.append((user_id, conv_id))


class _FakeSearchResult:
    def __init__(self, data):
        self.success = True
        self.data = data
        self.reranked = True


class _FakeToolManager:
    def __init__(self):
        self.search_calls = []
        self._tools = {}

    async def search_with_rewrite(self, tool_name, message, top_k=3, recall_k=None, context=None):
        self.search_calls.append({
            "tool_name": tool_name,
            "message": message,
            "top_k": top_k,
        })
        return _FakeSearchResult([
            {
                "title": "退款政策",
                "heading_path": "制度中心 > 财务 > 退款政策",
                "score": 0.93,
                "matched_child_content": "审核通过后，款项将在 5-7 个工作日内退回原支付账户。",
                "parent_content": "退款政策说明。审核通过后，款项将在 5-7 个工作日内退回原支付账户。",
            }
        ])

    async def call(self, name, params, context=None, use_cache=True, rerank_top_k=0):
        raise AssertionError(f"unexpected tool call: {name}")


class _FakeStreamingOrchestrator:
    def __init__(self, parts=None, fail_after_parts=None):
        self.parts = parts or ["退款审核通过后", " 5-7 个工作日到账。"]
        self.fail_after_parts = fail_after_parts
        self.requests = []

    async def _emit(self, on_delta):
        for index, part in enumerate(self.parts):
            await on_delta(part)
            if self.fail_after_parts is not None and index + 1 >= self.fail_after_parts:
                raise RuntimeError("stream interrupted")

    async def _execute_streaming(self, req, agent_type, on_delta, context=None):
        self.requests.append(req)
        await self._emit(on_delta)
        return AgentResponse(
            agent_type=agent_type,
            content="".join(self.parts),
            success=True,
            latency_ms=12.3,
        )

    async def stream(self, req, on_delta, context=None):
        self.requests.append(req)
        await self._emit(on_delta)
        return OrchestratorResult(
            request_id=req.request_id,
            response="".join(self.parts),
            agent_type=AgentType.BILLING,
            intent=req.intent,
            escalated=False,
            latency_ms=12.3,
        )

    async def _execute(self, req, agent_type, context=None):
        self.requests.append(req)
        return AgentResponse(
            agent_type=agent_type,
            content="".join(self.parts),
            success=True,
            latency_ms=12.3,
        )


class _FakeMsgRole(Enum):
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"


def _parse_sse_events(text: str):
    events = []
    for raw_block in text.strip().split("\n\n"):
        if not raw_block.strip():
            continue
        event_name = "message"
        data_lines = []
        for line in raw_block.splitlines():
            if line.startswith("event:"):
                event_name = line.split(":", 1)[1].strip()
            elif line.startswith("data:"):
                data_lines.append(line.split(":", 1)[1].lstrip())
        payload = json.loads("\n".join(data_lines)) if data_lines else None
        events.append({"event": event_name, "data": payload})
    return events


class ChatStreamingTests(unittest.TestCase):
    def setUp(self):
        self._original_memory = api_main._memory
        self._original_tool_manager = api_main._tool_manager
        self._original_orchestrator = api_main._orchestrator
        self._original_recognizer = getattr(api_main, "_chat_intent_recognizer", None)
        self._original_workflow_decider = getattr(api_main, "_workflow_intent_decider", None)
        self._original_memory_module = sys.modules.get("memory.conversation_memory")

        self.fake_memory = _FakeMemory()
        self.fake_tool_manager = _FakeToolManager()
        self.fake_orchestrator = _FakeStreamingOrchestrator()

        api_main._memory = self.fake_memory
        api_main._tool_manager = self.fake_tool_manager
        api_main._orchestrator = self.fake_orchestrator

        fake_memory_module = types.ModuleType("memory.conversation_memory")
        fake_memory_module.MsgRole = _FakeMsgRole
        sys.modules["memory.conversation_memory"] = fake_memory_module

    def tearDown(self):
        api_main._memory = self._original_memory
        api_main._tool_manager = self._original_tool_manager
        api_main._orchestrator = self._original_orchestrator
        api_main._chat_intent_recognizer = self._original_recognizer
        api_main._workflow_intent_decider = self._original_workflow_decider
        if self._original_memory_module is None:
            sys.modules.pop("memory.conversation_memory", None)
        else:
            sys.modules["memory.conversation_memory"] = self._original_memory_module

    def test_sse_event_formats_json_payload(self):
        payload = api_main._sse_event("stage", {"type": "stage.started", "stage": "planning"})

        self.assertEqual(
            payload,
            'event: stage\ndata: {"type":"stage.started","stage":"planning"}\n\n',
        )

    def test_chat_stream_emits_stage_and_answer_events_for_knowledge_question(self):
        api_main._chat_intent_recognizer = _FakeRecognizer(_FakeIntentResult(
            intent=IntentCategory.BILLING,
            confidence=0.96,
            urgency=UrgencyLevel.LOW,
            entities={"order_id": [], "product": [], "date": [], "amount": [], "error_code": []},
            reasoning="用户在咨询退款到账时效",
            latency_ms=2.0,
        ))
        api_main._workflow_intent_decider = _FakeWorkflowDecider(WorkflowDecision(
            mode=DecisionMode.KNOWLEDGE,
            tools=["knowledge_search"],
            specific_order=False,
            confidence=0.95,
            reason="该问题可由知识库回答",
        ))

        response = TestClient(api_main.app).post(
            "/chat/stream",
            json={"message": "退款多久到账", "user_id": "u123", "conv_id": "conv-stream-1"},
        )
        body = response.text

        self.assertEqual(response.status_code, 200)
        self.assertIn("text/event-stream", response.headers["content-type"])

        events = _parse_sse_events(body)
        self.assertEqual(events[0]["data"]["type"], "run.started")
        self.assertEqual(
            [item["data"]["stage"] for item in events if item["data"]["type"] == "stage.started"],
            ["understanding", "planning", "retrieving", "answering"],
        )
        self.assertEqual(
            "".join(item["data"]["delta"] for item in events if item["data"]["type"] == "answer.delta"),
            "退款审核通过后 5-7 个工作日到账。",
        )
        completed = next(item["data"] for item in events if item["data"]["type"] == "answer.completed")
        self.assertEqual(completed["response"], "退款审核通过后 5-7 个工作日到账。")
        self.assertTrue(completed["knowledge_used"])
        self.assertEqual(
            self.fake_memory.added,
            [
                ("u123", "conv-stream-1", "user", "退款多久到账"),
                ("u123", "conv-stream-1", "assistant", "退款审核通过后 5-7 个工作日到账。"),
            ],
        )

    def test_chat_stream_clarify_flow_skips_retrieval_and_answer_deltas(self):
        api_main._chat_intent_recognizer = _FakeRecognizer(_FakeIntentResult(
            intent=IntentCategory.BILLING,
            confidence=0.94,
            urgency=UrgencyLevel.MEDIUM,
            entities={"order_id": [], "product": [], "date": [], "amount": [], "error_code": []},
            reasoning="用户想退款但没提供订单号",
            latency_ms=2.0,
        ))

        response = TestClient(api_main.app).post(
            "/chat/stream",
            json={"message": "我要退款", "user_id": "u123", "conv_id": "conv-stream-clarify"},
        )
        body = response.text

        events = _parse_sse_events(body)
        started_stages = [item["data"]["stage"] for item in events if item["data"]["type"] == "stage.started"]
        self.assertEqual(started_stages, ["understanding", "planning", "answering"])
        self.assertEqual([item for item in events if item["data"]["type"] == "answer.delta"], [])
        completed = next(item["data"] for item in events if item["data"]["type"] == "answer.completed")
        self.assertIn("订单号", completed["response"])
        self.assertEqual(self.fake_orchestrator.requests, [])

    def test_chat_stream_failure_does_not_persist_partial_assistant_message(self):
        api_main._chat_intent_recognizer = _FakeRecognizer(_FakeIntentResult(
            intent=IntentCategory.BILLING,
            confidence=0.96,
            urgency=UrgencyLevel.LOW,
            entities={"order_id": [], "product": [], "date": [], "amount": [], "error_code": []},
            reasoning="用户在咨询退款到账时效",
            latency_ms=2.0,
        ))
        api_main._workflow_intent_decider = _FakeWorkflowDecider(WorkflowDecision(
            mode=DecisionMode.KNOWLEDGE,
            tools=["knowledge_search"],
            specific_order=False,
            confidence=0.95,
            reason="该问题可由知识库回答",
        ))
        api_main._orchestrator = _FakeStreamingOrchestrator(fail_after_parts=1)

        response = TestClient(api_main.app).post(
            "/chat/stream",
            json={"message": "退款多久到账", "user_id": "u123", "conv_id": "conv-stream-fail"},
        )
        body = response.text

        events = _parse_sse_events(body)
        self.assertEqual(next(item["data"]["type"] for item in events[-1:]), "error")
        self.assertEqual(self.fake_memory.added, [])


if __name__ == "__main__":
    unittest.main()
