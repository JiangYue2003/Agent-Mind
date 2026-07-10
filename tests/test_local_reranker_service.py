import unittest
from pathlib import Path

from fastapi.testclient import TestClient

from local_reranker.app import create_app


class _FakeReranker:
    def __init__(self):
        self.calls = []

    def compute_score(self, pairs, **kwargs):
        self.calls.append({"pairs": pairs, "kwargs": kwargs})
        return [0.18, 0.94, 0.51]


class LocalRerankerServiceTests(unittest.TestCase):
    def setUp(self):
        self.reranker = _FakeReranker()
        self.client = TestClient(create_app(reranker_factory=lambda: self.reranker))
        self.client.__enter__()

    def tearDown(self):
        self.client.__exit__(None, None, None)

    def test_health_reports_ready_cuda_backend(self):
        response = self.client.get("/health")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {
            "status": "ok",
            "backend": "FlagEmbedding",
            "device": "cuda:0",
        })

    def test_rerank_returns_tei_compatible_scores_in_descending_order(self):
        response = self.client.post("/rerank", json={
            "query": "退款多久到账",
            "texts": ["无关内容", "退款到账时间", "退款审核流程"],
            "return_text": False,
            "truncate": True,
        })

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), [
            {"index": 1, "score": 0.94},
            {"index": 2, "score": 0.51},
            {"index": 0, "score": 0.18},
        ])
        self.assertEqual(self.reranker.calls, [{
            "pairs": [
                ["退款多久到账", "无关内容"],
                ["退款多久到账", "退款到账时间"],
                ["退款多久到账", "退款审核流程"],
            ],
            "kwargs": {"normalize": True},
        }])

    def test_rerank_can_return_original_texts_when_requested(self):
        response = self.client.post("/rerank", json={
            "query": "退款多久到账",
            "texts": ["无关内容", "退款到账时间", "退款审核流程"],
            "return_text": True,
            "truncate": True,
        })

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()[0], {
            "index": 1,
            "score": 0.94,
            "text": "退款到账时间",
        })

    def test_requirements_pin_transformers_v4_for_flagreranker(self):
        requirements = Path("local_reranker/requirements.txt").read_text(encoding="utf-8")

        self.assertIn("transformers==4.57.3", requirements)


if __name__ == "__main__":
    unittest.main()
