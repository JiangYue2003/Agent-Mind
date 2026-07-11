import math
import unittest

from evaluation.retrieval_metrics import evaluate_retrieval_records


class RetrievalMetricsTests(unittest.TestCase):
    def test_evaluates_raw_and_reranked_candidate_layers(self):
        records = [
            {
                "metadata": {"case_id": "case-1", "unanswerable": False, "evaluation_mode": "knowledge"},
                "search": {
                    "raw_candidates": [
                        {"chunk_key": "noise"},
                        {"chunk_key": "gold-a"},
                    ],
                    "reranked_candidates": [
                        {"chunk_key": "gold-b"},
                        {"chunk_key": "noise"},
                        {"chunk_key": "gold-a"},
                    ],
                },
            },
            {
                "metadata": {"case_id": "case-unanswerable", "unanswerable": True, "evaluation_mode": "knowledge"},
                "search": {"raw_candidates": [], "reranked_candidates": []},
            },
        ]

        report = evaluate_retrieval_records(
            records,
            gold_chunk_keys={"case-1": ["gold-a", "gold-b"]},
            raw_k=15,
            final_k=3,
        )

        row = report["rows"][0]
        self.assertAlmostEqual(row["raw"]["recall_at_k"], 0.5)
        self.assertTrue(row["raw"]["hit_at_k"])
        self.assertAlmostEqual(row["raw"]["mrr_at_k"], 0.5)
        self.assertAlmostEqual(
            row["raw"]["ndcg_at_k"],
            (1 / math.log2(3)) / (1 + 1 / math.log2(3)),
        )
        self.assertEqual(row["reranked"]["recall_at_k"], 1.0)
        self.assertTrue(row["reranked"]["hit_at_k"])
        self.assertEqual(row["reranked"]["mrr_at_k"], 1.0)
        self.assertAlmostEqual(
            row["reranked"]["ndcg_at_k"],
            (1 + 1 / math.log2(4)) / (1 + 1 / math.log2(3)),
        )

        self.assertEqual(report["summary"]["answerable_cases"], 1)
        self.assertEqual(report["summary"]["unanswerable_cases"], 1)
        self.assertEqual(report["summary"]["unanswerable_noise_rate_raw"], 0.0)
        self.assertEqual(report["summary"]["unanswerable_noise_rate_reranked"], 0.0)


if __name__ == "__main__":
    unittest.main()
