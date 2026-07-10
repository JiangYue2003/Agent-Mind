import json
import pathlib
import tempfile
import unittest

from tools.summarize_ragas_report import build_summary, main


class SummarizeRagasReportTests(unittest.TestCase):
    def setUp(self):
        self.report = {
            "run_id": "ragas-test",
            "status": "completed",
            "summary": {"total_cases": 4},
            "evaluation_config": {"top_k": 5, "recall_k": 15},
            "records": [
                {
                    "user_input": "退款多久到账？",
                    "chat": {"knowledge_used": True, "retrieved_context_source": "chat_trace"},
                    "search": {"rerank_applied": True},
                    "metadata": {"case_id": "refund-001", "turn_index": 0, "category": "direct", "difficulty": "easy"},
                },
                {
                    "user_input": "订单状态是什么？",
                    "chat": {"knowledge_used": True, "retrieved_context_source": "chat_trace"},
                    "search": {"rerank_applied": True},
                    "metadata": {"case_id": "order-001", "turn_index": 0, "category": "direct", "difficulty": "medium"},
                },
                {
                    "user_input": "我想退款",
                    "chat": {"knowledge_used": False, "retrieved_context_source": "none"},
                    "search": {"rerank_applied": False},
                    "metadata": {"case_id": "workflow-001", "turn_index": 0, "category": "workflow", "difficulty": "easy"},
                },
            ],
            "ragas_rows": [
                {"case_id": "refund-001", "turn_index": 0, "faithfulness": 0.8, "factual_correctness": 0.6, "answer_relevancy": 0.9},
                {"case_id": "order-001", "turn_index": 0, "faithfulness": None, "factual_correctness": 0.2, "answer_relevancy": 0.7},
            ],
            "workflow_rows": [
                {"case_id": "workflow-001", "turn_index": 0, "expected_behavior": "clarify_order_id", "passed": False},
                {"case_id": "workflow-001", "turn_index": 1, "expected_behavior": "continue_workflow", "passed": True},
            ],
        }

    def test_build_summary_calculates_metrics_coverage_and_workflow_rate(self):
        summary = build_summary(self.report)

        self.assertEqual(summary["run_id"], "ragas-test")
        self.assertEqual(summary["counts"], {
            "records": 3,
            "ragas_rows": 2,
            "workflow_rows": 2,
        })
        self.assertEqual(summary["ragas_metrics"]["factual_correctness"]["valid_count"], 2)
        self.assertEqual(summary["ragas_metrics"]["factual_correctness"]["missing_count"], 0)
        self.assertEqual(summary["ragas_metrics"]["factual_correctness"]["mean"], 0.4)
        self.assertEqual(summary["ragas_metrics"]["faithfulness"]["valid_count"], 1)
        self.assertEqual(summary["ragas_metrics"]["faithfulness"]["missing_count"], 1)
        self.assertEqual(summary["workflow"]["pass_rate"], 0.5)
        self.assertEqual(summary["coverage"]["knowledge_used_rate"], 0.6667)
        self.assertEqual(summary["coverage"]["rerank_applied_rate"], 0.6667)
        self.assertEqual(summary["by_category"]["direct"]["factual_correctness"]["mean"], 0.4)
        self.assertEqual(summary["worst_cases"]["factual_correctness"][0]["case_id"], "order-001")
        self.assertEqual(summary["worst_cases"]["factual_correctness"][0]["user_input"], "订单状态是什么？")

    def test_main_writes_json_summary(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = pathlib.Path(temp_dir)
            report_path = root / "report.json"
            output_path = root / "summary.json"
            report_path.write_text(json.dumps(self.report, ensure_ascii=False), encoding="utf-8")

            exit_code = main([str(report_path), "--output", str(output_path)])

            saved = json.loads(output_path.read_text(encoding="utf-8"))

        self.assertEqual(exit_code, 0)
        self.assertEqual(saved["run_id"], "ragas-test")
        self.assertEqual(saved["workflow"]["passed"], 1)


if __name__ == "__main__":
    unittest.main()
