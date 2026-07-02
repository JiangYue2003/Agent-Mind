import asyncio
import sys
import types
import unittest
from dataclasses import dataclass
from enum import Enum

from api import main as api_main
from agents.agent_orchestrator import AgentType, OrchestratorResult
from core.intent_recognizer import IntentCategory, UrgencyLevel


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


class _FakeMemoryContext:
    def __init__(self):
        self.summary = "用户最近咨询退款。"
        self.relevant_history = []
        self.user_profile = {"tier": "vip"}
        self.recent_messages = []

    def to_prompt_text(self) -> str:
        return "[会话摘要]\n用户最近咨询退款。"


class _FakeMemory:
    async def get_context(self, user_id: str, conv_id: str, query: str):
        return _FakeMemoryContext()

    async def add_message(self, user_id, conv_id, role, content):
        return None

    async def update_profile(self, user_id, conv_id):
        return None


class _FakeSearchResult:
    def __init__(self, data):
        self.success = True
        self.data = data
        self.reranked = True


class _FakeToolManager:
    def __init__(self):
        self.last_context = None

    async def search_with_rewrite(self, tool_name, message, top_k=3, recall_k=None, context=None):
        self.last_context = context
        trace = (context or {}).get("trace")
        if trace is not None:
            with trace.stage("knowledge.query_rewrite"):
                pass
            with trace.stage("knowledge.hybrid_recall"):
                pass
            with trace.stage("knowledge.rerank"):
                pass
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
        raise AssertionError("tool call should not happen in this trace test")


class _FakeOrchestrator:
    def __init__(self):
        self.requests = []

    async def run(self, req, context=None):
        self.requests.append(req)
        return OrchestratorResult(
            request_id=req.request_id,
            response="退款审核通过后 5-7 个工作日到账。",
            agent_type=AgentType.BILLING,
            intent=req.intent,
            escalated=False,
            latency_ms=25.0,
        )


class _FakeMsgRole(Enum):
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"


class TracePipelineTests(unittest.TestCase):
    def setUp(self):
        self._original_memory = api_main._memory
        self._original_tool_manager = api_main._tool_manager
        self._original_orchestrator = api_main._orchestrator
        self._original_create_task = api_main.asyncio.create_task
        self._original_recognizer = getattr(api_main, "_chat_intent_recognizer", None)
        self._original_trace_store = getattr(api_main, "_trace_store", None)
        self._original_memory_module = sys.modules.get("memory.conversation_memory")

        api_main._memory = _FakeMemory()
        api_main._tool_manager = _FakeToolManager()
        api_main._orchestrator = _FakeOrchestrator()
        api_main.asyncio.create_task = lambda coro: coro.close()
        api_main._chat_intent_recognizer = _FakeRecognizer(_FakeIntentResult(
            intent=IntentCategory.BILLING,
            confidence=0.95,
            urgency=UrgencyLevel.LOW,
            entities={"order_id": [], "product": [], "date": [], "amount": [], "error_code": []},
            reasoning="用户在问退款到账时效",
            latency_ms=4.2,
        ))
        api_main._trace_store = None

        fake_memory_module = types.ModuleType("memory.conversation_memory")
        fake_memory_module.MsgRole = _FakeMsgRole
        sys.modules["memory.conversation_memory"] = fake_memory_module

    def tearDown(self):
        api_main._memory = self._original_memory
        api_main._tool_manager = self._original_tool_manager
        api_main._orchestrator = self._original_orchestrator
        api_main.asyncio.create_task = self._original_create_task
        api_main._chat_intent_recognizer = self._original_recognizer
        api_main._trace_store = self._original_trace_store
        if self._original_memory_module is None:
            sys.modules.pop("memory.conversation_memory", None)
        else:
            sys.modules["memory.conversation_memory"] = self._original_memory_module

    def test_chat_response_contains_trace_id_and_trace_can_be_queried(self):
        response = asyncio.run(api_main.chat(api_main.ChatRequest(
            message="退款多久到账",
            user_id="u123",
            conv_id="conv-trace-1",
        )))

        self.assertTrue(getattr(response, "trace_id", ""))

        trace_payload = asyncio.run(api_main.get_trace(response.trace_id))
        self.assertEqual(trace_payload["trace_id"], response.trace_id)
        self.assertEqual(trace_payload["user_id"], "u123")
        stage_names = [item["name"] for item in trace_payload["stages"]]
        self.assertIn("chat.total", stage_names)
        self.assertIn("memory.read", stage_names)
        self.assertIn("intent.recognize", stage_names)
        self.assertIn("knowledge_context.build", stage_names)
        self.assertIn("orchestrator.run", stage_names)
        self.assertIn("memory.write", stage_names)

    def test_trace_records_knowledge_substages(self):
        response = asyncio.run(api_main.chat(api_main.ChatRequest(
            message="退款多久到账",
            user_id="u123",
            conv_id="conv-trace-2",
        )))

        trace_payload = asyncio.run(api_main.get_trace(response.trace_id))
        stage_names = [item["name"] for item in trace_payload["stages"]]
        self.assertIn("knowledge.query_rewrite", stage_names)
        self.assertIn("knowledge.hybrid_recall", stage_names)
        self.assertIn("knowledge.rerank", stage_names)


if __name__ == "__main__":
    unittest.main()
