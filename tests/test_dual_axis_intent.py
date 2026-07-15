import asyncio
import json
import unittest
from types import SimpleNamespace

from core.intent_recognizer import (
    IntentCategory,
    IntentDomain,
    IntentRecognizer,
    SpeechAct,
)
from workflow.intent_decider import WorkflowIntentDecider


class _FakeMessages:
    def __init__(self, payload):
        self.payload = payload
        self.calls = []

    async def create(self, **kwargs):
        self.calls.append(kwargs)
        return SimpleNamespace(content=[SimpleNamespace(text=json.dumps(self.payload))])


class DualAxisIntentTests(unittest.TestCase):
    def test_llm_recognizer_parses_speech_act_and_domain(self):
        recognizer = IntentRecognizer.__new__(IntentRecognizer)
        recognizer.client = SimpleNamespace(messages=_FakeMessages({
            "intent": "request",
            "speech_act": "operation",
            "domain": "billing",
            "confidence": 0.93,
            "reasoning": "用户要求退款",
        }))
        recognizer.model = "test-model"

        payload = asyncio.run(recognizer._llm_recognize("我要申请退款", history=[]))

        self.assertEqual(payload["intent"], IntentCategory.REQUEST)
        self.assertEqual(payload["speech_act"], SpeechAct.OPERATION)
        self.assertEqual(payload["domain"], IntentDomain.BILLING)

    def test_llm_prompt_defines_axes_and_cross_axis_examples(self):
        messages = _FakeMessages({
            "intent": "request",
            "speech_act": "operation",
            "domain": "billing",
            "confidence": 0.93,
            "reasoning": "用户要求退款",
        })
        recognizer = IntentRecognizer.__new__(IntentRecognizer)
        recognizer.client = SimpleNamespace(messages=messages)
        recognizer.model = "test-model"

        asyncio.run(recognizer._llm_recognize("我要申请退款", history=[]))

        prompt = messages.calls[0]["messages"][0]["content"]
        self.assertIn("可选行为轴: information, operation, complaint, escalation, social, ood", prompt)
        self.assertIn("可选领域轴: billing, account, technical, order, general, unknown", prompt)
        self.assertIn("“我要申请退款” -> operation + billing", prompt)
        self.assertIn("“帮我订机票” -> ood + unknown", prompt)
        self.assertIn("“怎么做/怎么办/为何发生/状态如何”一律为 information", prompt)
        self.assertIn("仅描述技术故障不构成 complaint", prompt)

    def test_legacy_category_has_stable_dual_axis_fallback(self):
        recognizer = IntentRecognizer.__new__(IntentRecognizer)

        speech_act, domain = recognizer._axes_from_legacy(IntentCategory.BILLING)

        self.assertEqual(speech_act, SpeechAct.INFORMATION)
        self.assertEqual(domain, IntentDomain.BILLING)

    def test_workflow_prompt_includes_dual_axis_evidence(self):
        messages = _FakeMessages({
            "goal": "action",
            "tools": ["refund_create"],
            "confidence": 0.95,
            "reason": "退款操作",
        })
        decider = WorkflowIntentDecider(
            api_key="test-key",
            client=SimpleNamespace(messages=messages),
        )

        asyncio.run(decider.decide(
            message="我要申请退款",
            intent=IntentCategory.REQUEST,
            entities={"order_id": []},
            history=[],
            speech_act=SpeechAct.OPERATION,
            domain=IntentDomain.BILLING,
        ))

        prompt = messages.calls[0]["messages"][0]["content"]
        self.assertIn("行为轴: operation", prompt)
        self.assertIn("领域轴: billing", prompt)
