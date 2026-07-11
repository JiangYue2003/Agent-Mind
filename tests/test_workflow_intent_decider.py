import asyncio
import json
import unittest
from types import SimpleNamespace

from core.intent_recognizer import IntentCategory
from workflow.intent_decider import DecisionMode, WorkflowIntentDecider


class _FakeMessages:
    def __init__(self, payload):
        self._payload = payload
        self.calls = []

    async def create(self, **kwargs):
        self.calls.append(kwargs)
        return SimpleNamespace(content=[SimpleNamespace(text=json.dumps(self._payload))])


class _FakeClient:
    def __init__(self, payload):
        self.messages = _FakeMessages(payload)


class WorkflowIntentDeciderTests(unittest.TestCase):
    def test_retries_a_transient_llm_failure_before_using_workflow_fallback(self):
        payload = {
            "goal": "action",
            "tools": ["refund_create"],
            "confidence": 0.96,
            "reason": "用户明确要求提交退款",
        }

        class FlakyMessages:
            def __init__(self):
                self.calls = []

            async def create(self, **kwargs):
                self.calls.append(kwargs)
                if len(self.calls) == 1:
                    raise OSError("temporary model network failure")
                return SimpleNamespace(content=[SimpleNamespace(text=json.dumps(payload))])

        client = SimpleNamespace(messages=FlakyMessages())
        decider = WorkflowIntentDecider(api_key="test-key", client=client)

        decision = asyncio.run(decider.decide(
            message="请帮我申请退款",
            intent=IntentCategory.BILLING,
            entities={"order_id": ["ORD20250701001"]},
            history=[],
        ))

        self.assertEqual(decision.mode, DecisionMode.ACTION)
        self.assertEqual(decision.tools, ["refund_create"])
        self.assertEqual(len(client.messages.calls), 2)

    def test_routes_general_delivery_question_to_knowledge_without_order_id(self):
        client = _FakeClient({
            "mode": "knowledge",
            "tools": ["knowledge_search"],
            "specific_order": False,
            "confidence": 0.94,
            "reason": "用户询问通用发货时效",
        })
        decider = WorkflowIntentDecider(api_key="test-key", client=client)

        decision = asyncio.run(decider.decide(
            message="有单号但没有物流记录正常吗？",
            intent=IntentCategory.QUERY,
            entities={"order_id": []},
            history=[],
        ))

        self.assertEqual(decision.mode, DecisionMode.KNOWLEDGE)
        self.assertEqual(decision.tools, ["knowledge_search"])
        self.assertFalse(decision.requires_order_id)
        self.assertFalse(decision.should_clarify)
        self.assertEqual(len(client.messages.calls), 1)

    def test_routes_specific_order_without_id_to_clarification(self):
        client = _FakeClient({
            "mode": "live_record",
            "tools": ["shipment_track"],
            "specific_order": True,
            "confidence": 0.96,
            "reason": "用户要查询自己订单的实时物流",
        })
        decider = WorkflowIntentDecider(api_key="test-key", client=client)

        decision = asyncio.run(decider.decide(
            message="我的订单怎么还没发货？",
            intent=IntentCategory.QUERY,
            entities={"order_id": []},
            history=[],
        ))

        self.assertEqual(decision.mode, DecisionMode.LIVE_RECORD)
        self.assertTrue(decision.requires_order_id)
        self.assertTrue(decision.should_clarify)

    def test_goal_action_without_order_id_keeps_tool_and_uses_schema_slots(self):
        client = _FakeClient({
            "goal": "action",
            "tools": ["refund_create"],
            "confidence": 0.96,
            "reason": "用户希望发起退款，但尚未提供订单号",
        })
        decider = WorkflowIntentDecider(api_key="test-key", client=client)

        decision = asyncio.run(decider.decide(
            message="我想退款",
            intent=IntentCategory.REQUEST,
            entities={"order_id": []},
            history=[],
        ))

        self.assertEqual(decision.mode, DecisionMode.ACTION)
        self.assertEqual(decision.tools, ["refund_create"])
        self.assertEqual(decision.required_slots, ["order_id"])
        self.assertTrue(decision.should_clarify)
        self.assertIn("我想退款", client.messages.calls[0]["messages"][0]["content"])

    def test_reuses_only_one_order_id_from_recent_history(self):
        client = _FakeClient({
            "mode": "live_record",
            "tools": ["shipment_track"],
            "specific_order": True,
            "confidence": 0.95,
            "reason": "用户继续查询上一笔订单的物流",
        })
        decider = WorkflowIntentDecider(api_key="test-key", client=client)

        decision = asyncio.run(decider.decide(
            message="现在到哪了？",
            intent=IntentCategory.QUERY,
            entities={"order_id": []},
            history=[{"role": "user", "content": "订单号是 ORD20250701001"}],
        ))

        self.assertEqual(decision.order_id, "ORD20250701001")
        self.assertFalse(decision.should_clarify)

    def test_does_not_reuse_an_ambiguous_order_id_from_recent_history(self):
        client = _FakeClient({
            "mode": "live_record",
            "tools": ["order_lookup"],
            "specific_order": True,
            "confidence": 0.95,
            "reason": "用户要查询具体订单",
        })
        decider = WorkflowIntentDecider(api_key="test-key", client=client)

        decision = asyncio.run(decider.decide(
            message="现在什么状态？",
            intent=IntentCategory.QUERY,
            entities={"order_id": []},
            history=[
                {"role": "user", "content": "订单号是 ORD20250701001"},
                {"role": "user", "content": "还有 ORD20250701002"},
            ],
        ))

        self.assertEqual(decision.order_id, "")
        self.assertTrue(decision.should_clarify)

    def test_ignores_invalid_order_id_entities_before_selecting_live_tool(self):
        client = _FakeClient({
            "mode": "live_record",
            "tools": ["order_lookup"],
            "specific_order": True,
            "confidence": 0.95,
            "reason": "用户要查询具体订单",
        })
        decider = WorkflowIntentDecider(api_key="test-key", client=client)

        decision = asyncio.run(decider.decide(
            message="帮我查订单状态",
            intent=IntentCategory.QUERY,
            entities={"order_id": ["订单号不知道"]},
            history=[],
        ))

        self.assertEqual(decision.order_id, "")
        self.assertTrue(decision.should_clarify)

    def test_invalid_or_low_confidence_decision_falls_back_to_knowledge(self):
        client = _FakeClient({
            "mode": "live_record",
            "tools": ["shipment_track"],
            "specific_order": True,
            "confidence": 0.2,
            "reason": "不确定",
        })
        decider = WorkflowIntentDecider(api_key="test-key", client=client)

        decision = asyncio.run(decider.decide(
            message="物流一直不动什么时候找客服？",
            intent=IntentCategory.QUERY,
            entities={"order_id": []},
            history=[],
        ))

        self.assertEqual(decision.mode, DecisionMode.KNOWLEDGE)
        self.assertEqual(decision.tools, ["knowledge_search"])
        self.assertFalse(decision.should_clarify)

    def test_incompatible_mode_and_tool_falls_back_without_business_action(self):
        client = _FakeClient({
            "mode": "knowledge",
            "tools": ["refund_create"],
            "specific_order": True,
            "confidence": 0.98,
            "reason": "错误地选择了退款工具",
        })
        decider = WorkflowIntentDecider(api_key="test-key", client=client)

        decision = asyncio.run(decider.decide(
            message="退款规则是什么？",
            intent=IntentCategory.BILLING,
            entities={"order_id": ["ORD20250701001"]},
            history=[],
        ))

        self.assertEqual(decision.mode, DecisionMode.KNOWLEDGE)
        self.assertEqual(decision.tools, ["knowledge_search"])
        self.assertTrue(decision.used_fallback)


if __name__ == "__main__":
    unittest.main()
