import asyncio
import pathlib
import sys
import types
import unittest


if "dashscope" not in sys.modules:
    dashscope_stub = types.ModuleType("dashscope")

    class _StubTextEmbedding:
        @staticmethod
        def call(*args, **kwargs):
            raise NotImplementedError

    dashscope_stub.TextEmbedding = _StubTextEmbedding
    sys.modules["dashscope"] = dashscope_stub

ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.intent_recognizer import _TEMPLATES, IntentCategory, IntentRecognizer


class _EmbeddingClient:
    def __init__(self):
        self.document_calls = []
        self.query_calls = []

    def embed_documents(self, texts):
        self.document_calls.append(list(texts))
        return [[float(index), 1.0] for index, _ in enumerate(texts)]

    def embed_query(self, text):
        self.query_calls.append(text)
        return [1.0, 1.0]


class _UnavailableEmbeddingClient:
    def embed_documents(self, texts):
        raise RuntimeError("DashScope unavailable")

    def embed_query(self, text):
        raise RuntimeError("DashScope unavailable")


class IntentRecognizerEmbeddingTests(unittest.TestCase):
    def _recognizer(self):
        recognizer = IntentRecognizer(api_key="llm-key", base_url="https://example.invalid/anthropic")
        recognizer._embedding_enabled = True
        return recognizer

    def test_template_embeddings_are_batched_as_dashscope_documents(self):
        recognizer = self._recognizer()
        recognizer._embedding_client = _EmbeddingClient()

        asyncio.run(recognizer._load_template_embeddings())

        expected_count = sum(len(texts) for texts in _TEMPLATES.values())
        self.assertEqual(len(recognizer._embedding_client.document_calls), 1)
        self.assertEqual(len(recognizer._embedding_client.document_calls[0]), expected_count)

    def test_message_embedding_uses_dashscope_query_vector(self):
        recognizer = self._recognizer()
        recognizer._embedding_client = _EmbeddingClient()

        vector = asyncio.run(recognizer._embed_text("退款多久到账"))

        self.assertEqual(vector, [1.0, 1.0])
        self.assertEqual(recognizer._embedding_client.query_calls, ["退款多久到账"])

    def test_dashscope_failure_skips_embedding_branch_without_local_vector_fallback(self):
        recognizer = self._recognizer()
        recognizer._embedding_client = _UnavailableEmbeddingClient()
        local_vector_used = []

        def local_vector(text):
            local_vector_used.append(text)
            return [0.0, 0.0]

        recognizer._local_embedding = local_vector

        result = asyncio.run(recognizer._embedding_recognize("退款多久到账"))

        self.assertEqual(result["intent"], IntentCategory.OTHER)
        self.assertEqual(result["confidence"], 0.0)
        self.assertEqual(local_vector_used, [])


if __name__ == "__main__":
    unittest.main()
