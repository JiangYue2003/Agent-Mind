import collections
from pathlib import Path
import unittest

from evaluation.ragas_runner import load_eval_cases


class CustomerServiceRagasDatasetTests(unittest.TestCase):
    def test_dataset_has_expected_coverage_and_source_grounding(self):
        root = Path(__file__).resolve().parents[1]
        cases = load_eval_cases(root / "evaluation" / "datasets" / "customer_service_v1.jsonl")

        self.assertEqual(len(cases), 80)
        self.assertEqual(
            collections.Counter(case["category"] for case in cases),
            {"direct": 50, "paraphrase": 15, "confusion": 10, "unanswerable": 5},
        )

        for case in cases:
            if case["unanswerable"]:
                self.assertEqual(case["reference_contexts"], [])
                continue
            source = root / "docs" / "customer-service-kb" / case["source_file"]
            self.assertTrue(source.is_file(), case["case_id"])
            source_text = source.read_text(encoding="utf-8")
            for evidence in case["reference_contexts"]:
                self.assertIn(evidence, source_text, case["case_id"])

    def test_workflow_dataset_uses_strict_multi_turn_slot_continuation(self):
        root = Path(__file__).resolve().parents[1]
        cases = load_eval_cases(
            root / "evaluation" / "datasets" / "customer_service_workflow_v1.jsonl"
        )

        self.assertTrue(cases)
        for case in cases:
            self.assertEqual(case["evaluation_mode"], "workflow")
            self.assertEqual(case["turns"][0]["expected_behavior"], "clarify_order_id")
            self.assertEqual(case["turns"][1]["expected_behavior"], "continue_workflow")
            self.assertIn("订单号", case["turns"][1]["user_input"])

    def test_workflow_smoke_dataset_keeps_refund_and_shipment_regressions(self):
        root = Path(__file__).resolve().parents[1]
        cases = load_eval_cases(
            root / "evaluation" / "datasets" / "customer_service_workflow_smoke_v1.jsonl"
        )

        self.assertEqual(
            [case["case_id"] for case in cases],
            ["workflow-refund-001", "workflow-shipment-001"],
        )
        self.assertTrue(all(case["evaluation_mode"] == "workflow" for case in cases))
        self.assertTrue(all(len(case["turns"]) == 2 for case in cases))


if __name__ == "__main__":
    unittest.main()
