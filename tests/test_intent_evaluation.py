import asyncio
import json
import tempfile
import unittest
from pathlib import Path

from core.intent_recognizer import IntentCategory, UrgencyLevel
from evaluation.intent_metrics import compute_dual_axis_metrics, compute_intent_metrics, compute_workflow_metrics
from evaluation.intent_runner import IntentEvalCase, evaluate_cases, load_intent_cases
from workflow.intent_decider import DecisionMode, WorkflowDecision


class _FakeIntentResult:
    def __init__(self, intent, confidence=0.9):
        self.intent = intent
        self.confidence = confidence
        self.urgency = UrgencyLevel.LOW
        self.entities = {"order_id": []}
        self.reasoning = "test"
        self.latency_ms = 12.5


class _FakeRecognizer:
    def __init__(self):
        self._cache = {"stale": object()}
        self.calls = []

    async def recognize(self, message, history=None):
        self.calls.append((message, history))
        return _FakeIntentResult(
            IntentCategory.BILLING if "退款" in message else IntentCategory.QUERY
        )


class _FakeDecider:
    async def decide(self, message, intent, entities, history):
        if "退款" in message:
            return WorkflowDecision(
                mode=DecisionMode.ACTION,
                tools=["refund_create"],
                confidence=0.9,
                reason="退款操作",
            )
        return WorkflowDecision(
            mode=DecisionMode.KNOWLEDGE,
            tools=["knowledge_search"],
            confidence=0.9,
            reason="规则查询",
        )


class IntentMetricsTests(unittest.TestCase):
    def test_intent_metrics_include_fixed_labels_and_confusion_matrix(self):
        report = compute_intent_metrics([
            {"expected_intent": "billing", "predicted_intent": "billing"},
            {"expected_intent": "billing", "predicted_intent": "query"},
            {"expected_intent": "query", "predicted_intent": "query"},
        ])

        self.assertEqual(report["total"], 3)
        self.assertEqual(report["accuracy"], 0.6667)
        self.assertIn("other", report["per_class"])
        self.assertEqual(report["confusion_matrix"]["billing"]["query"], 1)

    def test_workflow_metrics_keep_each_decision_dimension_separate(self):
        report = compute_workflow_metrics([
            {
                "expected_workflow_mode": "action",
                "expected_tools": ["refund_create"],
                "expected_should_clarify": True,
                "workflow": {
                    "mode": "action",
                    "tools": ["refund_create"],
                    "should_clarify": False,
                },
            },
            {
                "expected_workflow_mode": "knowledge",
                "expected_tools": ["knowledge_search"],
                "expected_should_clarify": False,
                "workflow": {
                    "mode": "knowledge",
                    "tools": ["knowledge_search"],
                    "should_clarify": False,
                },
            },
        ])

        self.assertEqual(report["total"], 2)
        self.assertEqual(report["mode_accuracy"], 1.0)
        self.assertEqual(report["tool_set_accuracy"], 1.0)
        self.assertEqual(report["clarify_accuracy"], 0.5)
        self.assertEqual(report["exact_match"], 0.5)

    def test_workflow_metrics_skip_cases_without_complete_gold(self):
        report = compute_workflow_metrics([
            {
                "expected_workflow_mode": None,
                "expected_tools": None,
                "expected_should_clarify": None,
                "workflow": None,
            },
        ])

        self.assertEqual(report["total"], 0)

    def test_dual_axis_metrics_score_axes_and_joint_match_separately(self):
        report = compute_dual_axis_metrics([
            {
                "expected_speech_act": "operation",
                "expected_domain": "billing",
                "predicted_speech_act": "operation",
                "predicted_domain": "billing",
            },
            {
                "expected_speech_act": "information",
                "expected_domain": "technical",
                "predicted_speech_act": "information",
                "predicted_domain": "general",
            },
        ])

        self.assertEqual(report["total"], 2)
        self.assertEqual(report["speech_act"]["accuracy"], 1.0)
        self.assertEqual(report["domain"]["accuracy"], 0.5)
        self.assertEqual(report["joint_exact_match"], 0.5)


class IntentRunnerTests(unittest.TestCase):
    def test_versioned_datasets_cover_every_current_intent_label(self):
        root = Path(__file__).resolve().parents[1]
        cases = load_intent_cases(root / "evaluation" / "datasets" / "customer_service_intent_v1.jsonl")
        smoke_cases = load_intent_cases(root / "evaluation" / "datasets" / "customer_service_intent_smoke_v1.jsonl")

        self.assertEqual({case.expected_intent for case in cases}, {item.value for item in IntentCategory})
        self.assertGreaterEqual(len(cases), 80)
        self.assertGreaterEqual(len(smoke_cases), 20)
        self.assertTrue(any(case.history for case in cases))

    def test_boundary_case_labels_follow_the_declared_action_priority(self):
        root = Path(__file__).resolve().parents[1]
        cases = load_intent_cases(root / "evaluation" / "datasets" / "customer_service_intent_v1.jsonl")
        labels = {case.case_id: case.expected_intent for case in cases}

        self.assertEqual(labels["intent-query-004"], "billing")
        self.assertEqual(labels["intent-billing-002"], "request")
        self.assertEqual(labels["intent-billing-004"], "request")
        self.assertEqual(labels["intent-multiturn-002"], "request")
        self.assertEqual(labels["intent-account-006"], "request")

    def test_dual_axis_datasets_cover_all_axis_labels(self):
        root = Path(__file__).resolve().parents[1]
        cases = load_intent_cases(root / "evaluation" / "datasets" / "customer_service_dual_axis_v1.jsonl")
        smoke_cases = load_intent_cases(root / "evaluation" / "datasets" / "customer_service_dual_axis_smoke_v1.jsonl")

        self.assertEqual({case.expected_speech_act for case in cases}, {
            "information", "operation", "complaint", "escalation", "social", "ood",
        })
        self.assertEqual({case.expected_domain for case in cases}, {
            "billing", "account", "technical", "order", "general", "unknown",
        })
        self.assertGreaterEqual(len(cases), 40)
        self.assertGreaterEqual(len(smoke_cases), 20)

    def test_load_intent_cases_rejects_unknown_intent(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            dataset = Path(temp_dir) / "invalid.jsonl"
            dataset.write_text(json.dumps({
                "case_id": "bad-001",
                "user_input": "test",
                "expected_intent": "not-a-category",
            }), encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "expected_intent"):
                load_intent_cases(dataset)

    def test_evaluate_cases_uses_history_clears_cache_and_records_workflow(self):
        recognizer = _FakeRecognizer()
        rows = asyncio.run(evaluate_cases(
            [IntentEvalCase(
                case_id="billing-001",
                user_input="我要退款",
                history=[{"role": "assistant", "content": "请提供订单号"}],
                expected_intent="billing",
                expected_workflow_mode="action",
                expected_tools=["refund_create"],
                expected_should_clarify=True,
            )],
            recognizer=recognizer,
            decider=_FakeDecider(),
            scope="all",
        ))

        self.assertEqual(recognizer._cache, {})
        self.assertEqual(recognizer.calls[0][1][0]["content"], "请提供订单号")
        self.assertEqual(rows[0]["predicted_intent"], "billing")
        self.assertEqual(rows[0]["workflow"]["mode"], "action")
        self.assertTrue(rows[0]["workflow"]["should_clarify"])
