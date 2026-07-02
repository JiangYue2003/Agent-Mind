import json
import pathlib
import tempfile
import unittest

from tools.evaluate_retrieval import evaluate_cases, load_cases


class RetrievalEvaluationTests(unittest.TestCase):
    def test_load_cases_reads_case_array(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = pathlib.Path(temp_dir) / "cases.json"
            payload = [{
                "case_id": "case-1",
                "query": "退款多久到账",
                "query_type": "current_policy",
                "gold_answer": "5-7 个工作日",
                "gold_policy_ids": ["policy_refund"],
                "gold_version_no": "V2",
                "gold_effective_at": "2025-01-01",
                "gold_parent_ids": ["p1"],
                "gold_chunk_keys": ["p1:0"],
                "must_include_facts": [],
                "must_not_include_facts": [],
                "unanswerable": False,
                "difficulty": "easy",
                "evidence_summary": "evidence",
            }]
            path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")

            loaded = load_cases(path)

            self.assertEqual(len(loaded), 1)
            self.assertEqual(loaded[0]["case_id"], "case-1")

    def test_evaluate_cases_computes_metrics_and_details(self):
        cases = [
            {
                "case_id": "answerable-1",
                "query": "退款多久到账",
                "query_type": "current_policy",
                "gold_answer": "5-7 个工作日",
                "gold_policy_ids": ["policy_refund"],
                "gold_version_no": "V2",
                "gold_effective_at": "2025-01-01",
                "gold_parent_ids": ["refund-parent"],
                "gold_chunk_keys": ["refund-parent:0"],
                "must_include_facts": [],
                "must_not_include_facts": [],
                "unanswerable": False,
                "difficulty": "easy",
                "evidence_summary": "evidence",
            },
            {
                "case_id": "unanswerable-1",
                "query": "签证费能不能报",
                "query_type": "unanswerable",
                "gold_answer": "当前知识库中没有足够依据回答该问题",
                "gold_policy_ids": ["policy_travel"],
                "gold_version_no": "",
                "gold_effective_at": "",
                "gold_parent_ids": [],
                "gold_chunk_keys": [],
                "must_include_facts": [],
                "must_not_include_facts": ["住宿标准 650 元"],
                "unanswerable": True,
                "difficulty": "medium",
                "evidence_summary": "no evidence",
            },
        ]

        responses = {
            "退款多久到账": {
                "results": [
                    {
                        "parent_id": "refund-parent",
                        "matched_child_chunk": 0,
                        "policy_id": "policy_refund",
                        "version_no": "V2",
                        "effective_at": "2025-01-01",
                    },
                    {
                        "parent_id": "other-parent",
                        "matched_child_chunk": 0,
                        "policy_id": "policy_other",
                        "version_no": "V1",
                        "effective_at": "2024-01-01",
                    },
                ]
            },
            "签证费能不能报": {
                "results": [
                    {
                        "parent_id": "travel-parent",
                        "matched_child_chunk": 0,
                        "policy_id": "policy_travel",
                        "version_no": "V2",
                        "effective_at": "2025-03-01",
                    }
                ]
            },
        }

        def fake_search(query, top_k, recall_k=None):
            payload = responses[query]
            return {"query": query, "results": payload["results"][:top_k], "reranked": True}

        report = evaluate_cases(cases, search_fn=fake_search, top_k=3)

        summary = report["summary"]
        self.assertEqual(summary["total_cases"], 2)
        self.assertAlmostEqual(summary["hit_rate_at_k_parent"], 1.0)
        self.assertAlmostEqual(summary["hit_rate_at_k_chunk"], 1.0)
        self.assertAlmostEqual(summary["top1_exact_parent_accuracy"], 1.0)
        self.assertAlmostEqual(summary["policy_accuracy_at_k"], 1.0)
        self.assertAlmostEqual(summary["version_accuracy_at_k"], 1.0)
        self.assertAlmostEqual(summary["effective_at_accuracy_at_k"], 1.0)
        self.assertAlmostEqual(summary["mrr_parent"], 1.0)
        self.assertAlmostEqual(summary["mrr_chunk"], 1.0)
        self.assertAlmostEqual(summary["ndcg_at_k_parent"], 1.0)
        self.assertAlmostEqual(summary["noise_rate_at_k"], 0.5)
        self.assertAlmostEqual(summary["unanswerable_empty_recall_rate"], 0.0)
        self.assertAlmostEqual(summary["unanswerable_noise_rate"], 1.0)
        self.assertAlmostEqual(summary["avg_returned_results"], 1.5)

        details = report["details"]
        self.assertEqual(len(details), 2)
        self.assertTrue(details[0]["metrics"]["hit_parent"])
        self.assertTrue(details[0]["metrics"]["hit_chunk"])
        self.assertFalse(details[1]["metrics"]["expected_answerable"])
        self.assertEqual(details[1]["metrics"]["returned_results"], 1)

    def test_http_search_fn_passes_recall_k_when_provided(self):
        from tools.evaluate_retrieval import _build_http_search_fn

        calls = []

        class FakeResponse:
            def raise_for_status(self):
                return None

            def json(self):
                return {"query": "退款多久到账", "results": [], "reranked": True}

        class FakeClient:
            def post(self, path, params=None):
                calls.append({"path": path, "params": dict(params or {})})
                return FakeResponse()

        search_fn = _build_http_search_fn(
            base_url="http://localhost:8000",
            timeout_s=20.0,
            client=FakeClient(),
        )

        search_fn("退款多久到账", top_k=5, recall_k=12)

        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0]["path"], "/search")
        self.assertEqual(calls[0]["params"]["top_k"], 5)
        self.assertEqual(calls[0]["params"]["recall_k"], 12)

    def test_evaluate_cases_records_recall_k_in_summary(self):
        cases = [{
            "case_id": "answerable-1",
            "query": "退款多久到账",
            "query_type": "current_policy",
            "gold_answer": "5-7 个工作日",
            "gold_policy_ids": ["policy_refund"],
            "gold_version_no": "V2",
            "gold_effective_at": "2025-01-01",
            "gold_parent_ids": ["refund-parent"],
            "gold_chunk_keys": ["refund-parent:0"],
            "must_include_facts": [],
            "must_not_include_facts": [],
            "unanswerable": False,
            "difficulty": "easy",
            "evidence_summary": "evidence",
        }]

        seen = {}

        def fake_search(query, top_k, recall_k):
            seen["top_k"] = top_k
            seen["recall_k"] = recall_k
            return {
                "query": query,
                "results": [{
                    "parent_id": "refund-parent",
                    "matched_child_chunk": 0,
                    "policy_id": "policy_refund",
                    "version_no": "V2",
                    "effective_at": "2025-01-01",
                }],
                "reranked": True,
            }

        report = evaluate_cases(cases, search_fn=fake_search, top_k=3, recall_k=9)

        self.assertEqual(seen["top_k"], 3)
        self.assertEqual(seen["recall_k"], 9)
        self.assertEqual(report["summary"]["recall_k"], 9)


if __name__ == "__main__":
    unittest.main()
