import os
import unittest
from unittest.mock import patch

from core.dashscope_embedding import DashScopeEmbeddingClient, EmbeddingUnavailable


class _FakeResponse:
    status_code = 200
    message = ""
    output = {
        "embeddings": [
            {"embedding": [0.1, 0.2, 0.3, 0.4]},
            {"embedding": [0.5, 0.6, 0.7, 0.8]},
        ]
    }


class _FakeTextEmbedding:
    calls = []

    @classmethod
    def call(cls, **kwargs):
        cls.calls.append(kwargs)
        response = _FakeResponse()
        texts = kwargs["input"]
        if all(text.startswith("doc-") for text in texts):
            embeddings = [
                {"embedding": [float(int(text.removeprefix("doc-")))] * 4}
                for text in texts
            ]
        else:
            embeddings = _FakeResponse.output["embeddings"][:len(texts)]
        response.output = {
            "embeddings": embeddings
        }
        return response


class DashScopeEmbeddingClientTests(unittest.TestCase):
    def setUp(self):
        _FakeTextEmbedding.calls = []
        self.client = DashScopeEmbeddingClient(
            api_key="test-key",
            workspace="ws-test",
            model="text-embedding-v4",
            dimension=4,
            text_embedding=_FakeTextEmbedding,
        )

    def test_embed_documents_batches_texts_with_document_type(self):
        vectors = self.client.embed_documents(["退款规则", "物流时效"])

        self.assertEqual(vectors, [[0.1, 0.2, 0.3, 0.4], [0.5, 0.6, 0.7, 0.8]])
        self.assertEqual(_FakeTextEmbedding.calls, [{
            "api_key": "test-key",
            "workspace": "ws-test",
            "model": "text-embedding-v4",
            "input": ["退款规则", "物流时效"],
            "text_type": "document",
            "dimension": 4,
        }])

    def test_embed_documents_splits_requests_larger_than_ten_and_preserves_order(self):
        texts = [f"doc-{index}" for index in range(11)]

        vectors = self.client.embed_documents(texts)

        self.assertEqual(vectors, [[float(index)] * 4 for index in range(11)])
        self.assertEqual(
            [call["input"] for call in _FakeTextEmbedding.calls],
            [texts[:10], texts[10:]],
        )

    def test_embed_query_uses_query_type(self):
        vector = self.client.embed_query("退款多久到账")

        self.assertEqual(vector, [0.1, 0.2, 0.3, 0.4])
        self.assertEqual(_FakeTextEmbedding.calls[0]["text_type"], "query")

    def test_rejects_unexpected_embedding_dimension(self):
        self.client._text_embedding = type("WrongDimension", (), {
            "call": staticmethod(lambda **kwargs: _FakeResponse())
        })
        self.client._dimension = 3

        with self.assertRaises(EmbeddingUnavailable):
            self.client.embed_query("退款多久到账")

    def test_from_env_requires_workspace_scoped_dashscope_settings(self):
        with patch.dict(os.environ, {
            "DASHSCOPE_API_KEY": "test-key",
            "DASHSCOPE_HTTP_BASE_URL": "https://ws-test.example.com/api/v1",
            "DASHSCOPE_WORKSPACE": "ws-test",
            "EMBEDDING_MODEL": "text-embedding-v4",
            "EMBEDDING_DIMENSION": "1024",
        }, clear=True):
            client = DashScopeEmbeddingClient.from_env(text_embedding=_FakeTextEmbedding)

        self.assertEqual(client.model, "text-embedding-v4")
        self.assertEqual(client.dimension, 1024)

    def test_from_env_configures_sdk_base_url_from_workspace_endpoint(self):
        with patch.dict(os.environ, {
            "DASHSCOPE_API_KEY": "test-key",
            "DASHSCOPE_HTTP_BASE_URL": "https://ws-test.example.com/api/v1",
            "DASHSCOPE_WORKSPACE": "ws-test",
        }, clear=True):
            DashScopeEmbeddingClient.from_env(text_embedding=_FakeTextEmbedding)

            self.assertEqual(
                os.environ.get("DASHSCOPE_BASE_HTTP_API_URL"),
                "https://ws-test.example.com/api/v1",
            )


if __name__ == "__main__":
    unittest.main()
