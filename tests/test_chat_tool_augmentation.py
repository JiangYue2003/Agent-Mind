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
        if name == "shipment_track":
            return _FakeToolCallResult(True, {
                "order_id": params["order_id"],
                "carrier": "顺丰速运",
                "tracking_no": "SF1234567890",
                "shipment_status": "运输中",
                "events": [
                    {"time": "2026-07-02 12:00:00", "status": "已发出"},
                    {"time": "2026-07-03 09:30:00", "status": "运输中"},
                ],
                "source": "mock_tms",
            })
        if name == "refund_create":
            return _FakeToolCallResult(True, {
                "refund_id": "RF202607020001",
                "order_id": params["order_id"],
                "status": "submitted",
                "amount": "99.00",
                "reason": params.get("reason", ""),
                "source": "mock_refund",
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

    def test_chat_clarifies_when_refund_request_is_missing_order_id(self):
        api_main._chat_intent_recognizer = _FakeRecognizer(_FakeIntentResult(
            intent=IntentCategory.BILLING,
            confidence=0.94,
            urgency=UrgencyLevel.MEDIUM,
            entities={"order_id": [], "product": [], "date": [], "amount": [], "error_code": []},
            reasoning="用户想申请退款，但没有提供订单号",
            latency_ms=2.5,
        ))

        response = asyncio.run(api_main.chat(api_main.ChatRequest(
            message="我要退款",
            user_id="u123",
            conv_id="conv-clarify-1",
        )))

        self.assertIn("订单号", response.response)
        self.assertEqual(self.fake_tool_manager.search_calls, [])
        self.assertEqual(self.fake_tool_manager.call_calls, [])
        self.assertEqual(self.fake_orchestrator.requests, [])

    def test_chat_prefers_order_lookup_for_refund_status_with_order_id(self):
        api_main._chat_intent_recognizer = _FakeRecognizer(_FakeIntentResult(
            intent=IntentCategory.BILLING,
            confidence=0.97,
            urgency=UrgencyLevel.MEDIUM,
            entities={"order_id": ["ORD20250701002"], "product": [], "date": [], "amount": [], "error_code": []},
            reasoning="用户在查询自己订单的退款进度",
            latency_ms=2.8,
        ))

        response = asyncio.run(api_main.chat(api_main.ChatRequest(
            message="帮我查一下订单 ORD20250701002 现在退款到哪了",
            user_id="u456",
            conv_id="conv-refund-lookup-1",
        )))

        self.assertEqual(len(self.fake_tool_manager.call_calls), 1)
        self.assertEqual(self.fake_tool_manager.call_calls[0]["name"], "order_lookup")
        self.assertEqual(self.fake_tool_manager.search_calls, [])

        orch_req = self.fake_orchestrator.requests[0]
        self.assertIn("[order_lookup]", orch_req.context)
        self.assertNotIn("[知识库检索结果]", orch_req.context)

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
        self.assertEqual(self.fake_tool_manager.search_calls, [])

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

    def test_chat_combines_knowledge_and_order_lookup_for_mixed_billing_question(self):
        api_main._chat_intent_recognizer = _FakeRecognizer(_FakeIntentResult(
            intent=IntentCategory.BILLING,
            confidence=0.96,
            urgency=UrgencyLevel.MEDIUM,
            entities={"order_id": ["ORD20250701003"], "product": [], "date": [], "amount": [], "error_code": []},
            reasoning="用户既在问退款规则，也在查询具体订单退款进度",
            latency_ms=2.6,
        ))

        response = asyncio.run(api_main.chat(api_main.ChatRequest(
            message="请告诉我退款规则，并帮我看订单 ORD20250701003 现在退款到哪了",
            user_id="u123",
            conv_id="conv-4",
        )))

        self.assertEqual(self.fake_tool_manager.search_calls[0]["tool_name"], "knowledge_search")
        self.assertEqual(len(self.fake_tool_manager.call_calls), 1)
        self.assertEqual(self.fake_tool_manager.call_calls[0]["name"], "order_lookup")

        orch_req = self.fake_orchestrator.requests[0]
        self.assertIn("[知识库检索结果]", orch_req.context)
        self.assertIn("[工具增强上下文]", orch_req.context)
        self.assertIn("[order_lookup]", orch_req.context)

    def test_chat_uses_shipment_track_for_logistics_query(self):
        api_main._chat_intent_recognizer = _FakeRecognizer(_FakeIntentResult(
            intent=IntentCategory.QUERY,
            confidence=0.95,
            urgency=UrgencyLevel.MEDIUM,
            entities={"order_id": ["ORD20250701001"], "product": [], "date": [], "amount": [], "error_code": []},
            reasoning="用户在查询物流进度",
            latency_ms=2.1,
        ))

        response = asyncio.run(api_main.chat(api_main.ChatRequest(
            message="帮我查一下订单 ORD20250701001 的物流到哪了",
            user_id="u123",
            conv_id="conv-shipment-1",
        )))

        self.assertEqual(len(self.fake_tool_manager.call_calls), 1)
        self.assertEqual(self.fake_tool_manager.call_calls[0]["name"], "shipment_track")
        orch_req = self.fake_orchestrator.requests[0]
        self.assertIn("[shipment_track]", orch_req.context)
        self.assertIn("顺丰速运", orch_req.context)

    def test_chat_uses_refund_create_for_explicit_refund_request_with_order_id(self):
        api_main._chat_intent_recognizer = _FakeRecognizer(_FakeIntentResult(
            intent=IntentCategory.BILLING,
            confidence=0.97,
            urgency=UrgencyLevel.MEDIUM,
            entities={"order_id": ["ORD20250701001"], "product": [], "date": [], "amount": [], "error_code": []},
            reasoning="用户希望直接申请退款",
            latency_ms=2.0,
        ))

        response = asyncio.run(api_main.chat(api_main.ChatRequest(
            message="请直接帮我给订单 ORD20250701001 申请退款，原因是买错了",
            user_id="u123",
            conv_id="conv-refund-create-1",
        )))

        self.assertEqual(len(self.fake_tool_manager.call_calls), 1)
        self.assertEqual(self.fake_tool_manager.call_calls[0]["name"], "refund_create")
        orch_req = self.fake_orchestrator.requests[0]
        self.assertIn("[refund_create]", orch_req.context)
        self.assertIn("申请状态: submitted", orch_req.context)


if __name__ == "__main__":
    unittest.main()
