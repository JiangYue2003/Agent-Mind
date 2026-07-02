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
        self.calls = []

    async def recognize(self, message: str, history=None):
        self.calls.append({"message": message, "history": history})
        return self._result


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


class _FakeToolCallResult:
    def __init__(self, success, data):
        self.success = success
        self.data = data


class _FakeToolManager:
    def __init__(self):
        self.search_calls = []
        self.call_calls = []

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
                "score": 0.91,
                "matched_child_content": "审核通过后，款项将在 5-7 个工作日内退回原支付账户。",
                "parent_content": "退款政策说明。审核通过后，款项将在 5-7 个工作日内退回原支付账户。",
            }
        ])

    async def call(self, name, params, context=None, use_cache=True, rerank_top_k=0):
        self.call_calls.append({
            "name": name,
            "params": dict(params),
        })
        if name == "order_lookup":
            return _FakeToolCallResult(True, {
                "order_id": params["order_id"],
                "status": "运输中",
                "payment_status": "已支付",
                "shipment_status": "运输中",
                "refund_status": "",
                "updated_at": "2026-07-02T10:20:00+08:00",
                "source": "mock_oms",
            })
        if name == "human_handoff":
            return _FakeToolCallResult(True, {
                "handoff_id": "HD202607020001",
                "queue": "general_support",
                "status": "created",
                "eta_minutes": 8,
            })
        raise AssertionError(f"unexpected tool call: {name}")


class _FakeOrchestrator:
    def __init__(self):
        self.requests = []

    async def run(self, req, context=None):
        self.requests.append(req)
        return OrchestratorResult(
            request_id=req.request_id,
            response="好的，已为您处理。",
            agent_type=AgentType.GENERAL,
            intent=req.intent,
            escalated=req.intent == IntentCategory.ESCALATION,
            latency_ms=12.3,
        )


class _FakeMemoryContext:
    def __init__(self):
        self.summary = "用户最近咨询订单进度。"
        self.relevant_history = []
        self.user_profile = {"tier": "vip"}
        self.recent_messages = []

    def to_prompt_text(self) -> str:
        return "[会话摘要]\n用户最近咨询订单进度。\n\n[用户画像]\n{\"tier\": \"vip\"}"


class _FakeMsgRole(Enum):
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"


class ChatToolAugmentationTests(unittest.TestCase):
    def setUp(self):
        self._original_memory = api_main._memory
        self._original_tool_manager = api_main._tool_manager
        self._original_orchestrator = api_main._orchestrator
        self._original_create_task = api_main.asyncio.create_task
        self._original_recognizer = getattr(api_main, "_chat_intent_recognizer", None)
        self._original_memory_module = sys.modules.get("memory.conversation_memory")

        self.fake_memory = _FakeMemory()
        self.fake_tool_manager = _FakeToolManager()
        self.fake_orchestrator = _FakeOrchestrator()

        api_main._memory = self.fake_memory
        api_main._tool_manager = self.fake_tool_manager
        api_main._orchestrator = self.fake_orchestrator
        api_main.asyncio.create_task = lambda coro: coro.close()

        fake_memory_module = types.ModuleType("memory.conversation_memory")
        fake_memory_module.MsgRole = _FakeMsgRole
        sys.modules["memory.conversation_memory"] = fake_memory_module

    def tearDown(self):
        api_main._memory = self._original_memory
        api_main._tool_manager = self._original_tool_manager
        api_main._orchestrator = self._original_orchestrator
        api_main.asyncio.create_task = self._original_create_task
        api_main._chat_intent_recognizer = self._original_recognizer
        if self._original_memory_module is None:
            sys.modules.pop("memory.conversation_memory", None)
        else:
            sys.modules["memory.conversation_memory"] = self._original_memory_module

    def test_chat_injects_order_lookup_context_for_order_status_question(self):
        api_main._chat_intent_recognizer = _FakeRecognizer(_FakeIntentResult(
            intent=IntentCategory.QUERY,
            confidence=0.96,
            urgency=UrgencyLevel.MEDIUM,
            entities={"order_id": ["ORD20250701001"], "product": [], "date": [], "amount": [], "error_code": []},
            reasoning="用户在查询订单状态",
            latency_ms=3.2,
        ))

        response = asyncio.run(api_main.chat(api_main.ChatRequest(
            message="订单 ORD20250701001 现在到哪了？",
            user_id="u123",
            conv_id="conv-1",
        )))

        self.assertEqual(len(self.fake_tool_manager.call_calls), 1)
        self.assertEqual(self.fake_tool_manager.call_calls[0]["name"], "order_lookup")

        orch_req = self.fake_orchestrator.requests[0]
        self.assertEqual(orch_req.intent, IntentCategory.QUERY)
        self.assertEqual(orch_req.urgency, UrgencyLevel.MEDIUM)
        self.assertIn("[工具增强上下文]", orch_req.context)
        self.assertIn("[order_lookup]", orch_req.context)
        self.assertIn("订单号: ORD20250701001", orch_req.context)
        self.assertNotIn("[human_handoff]", orch_req.context)

    def test_chat_injects_handoff_context_for_explicit_escalation(self):
        api_main._chat_intent_recognizer = _FakeRecognizer(_FakeIntentResult(
            intent=IntentCategory.ESCALATION,
            confidence=0.99,
            urgency=UrgencyLevel.HIGH,
            entities={"order_id": [], "product": [], "date": [], "amount": [], "error_code": []},
            reasoning="用户明确要求转人工",
            latency_ms=2.1,
        ))

        response = asyncio.run(api_main.chat(api_main.ChatRequest(
            message="帮我转人工客服",
            user_id="u123",
            conv_id="conv-2",
        )))

        self.assertEqual(len(self.fake_tool_manager.call_calls), 1)
        self.assertEqual(self.fake_tool_manager.call_calls[0]["name"], "human_handoff")

        orch_req = self.fake_orchestrator.requests[0]
        self.assertEqual(orch_req.intent, IntentCategory.ESCALATION)
        self.assertIn("[工具增强上下文]", orch_req.context)
        self.assertIn("[human_handoff]", orch_req.context)
        self.assertIn("状态: created", orch_req.context)
        self.assertNotIn("[order_lookup]", orch_req.context)

    def test_chat_keeps_knowledge_only_for_regular_policy_question(self):
        api_main._chat_intent_recognizer = _FakeRecognizer(_FakeIntentResult(
            intent=IntentCategory.BILLING,
            confidence=0.88,
            urgency=UrgencyLevel.LOW,
            entities={"order_id": [], "product": [], "date": [], "amount": [], "error_code": []},
            reasoning="用户在问退款规则",
            latency_ms=2.4,
        ))

        response = asyncio.run(api_main.chat(api_main.ChatRequest(
            message="退款多久到账",
            user_id="u123",
            conv_id="conv-3",
        )))

        self.assertEqual(self.fake_tool_manager.search_calls[0]["tool_name"], "knowledge_search")
        self.assertEqual(self.fake_tool_manager.call_calls, [])

        orch_req = self.fake_orchestrator.requests[0]
        self.assertIn("[知识库检索结果]", orch_req.context)
        self.assertNotIn("[工具增强上下文]", orch_req.context)


if __name__ == "__main__":
    unittest.main()
