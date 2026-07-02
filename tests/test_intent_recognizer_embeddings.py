import asyncio
import os
import pathlib
import sys
import types
import unittest
from unittest.mock import patch


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

from core.intent_recognizer import IntentRecognizer


class _FakeEmbeddingsResponse:
    status_code = 200
    output = {
        "embeddings": [
            {"embedding": [0.11, 0.22, 0.33]}
        ]
    }


class IntentRecognizerEmbeddingTests(unittest.TestCase):
    def test_dashscope_embedding_enabled_even_with_third_party_llm_base_url(self):
        with patch.dict(os.environ, {
            "DASHSCOPE_API_KEY": "test-key",
            "INTENT_EMBEDDING_PROVIDER": "dashscope",
            "INTENT_EMBEDDING_MODEL": "text-embedding-v3",
        }, clear=False):
            recognizer = IntentRecognizer(
                api_key="llm-key",
                base_url="https://api.deepseek.com/anthropic",
                model="deepseek-chat",
            )

        self.assertTrue(recognizer._embedding_enabled)

    def test_embed_text_prefers_dashscope_provider(self):
        with patch.dict(os.environ, {
            "DASHSCOPE_API_KEY": "test-key",
            "INTENT_EMBEDDING_PROVIDER": "dashscope",
            "INTENT_EMBEDDING_MODEL": "text-embedding-v3",
        }, clear=False):
            recognizer = IntentRecognizer(api_key="llm-key", base_url="https://api.deepseek.com/anthropic")

        with patch("core.intent_recognizer.TextEmbedding.call", return_value=_FakeEmbeddingsResponse()) as mocked_call:
            vector = asyncio.run(recognizer._embed_text("退款多久到账"))

        self.assertEqual(vector, [0.11, 0.22, 0.33])
        mocked_call.assert_called_once()

    def test_embed_text_falls_back_to_local_vector_when_dashscope_fails(self):
        with patch.dict(os.environ, {
            "DASHSCOPE_API_KEY": "test-key",
            "INTENT_EMBEDDING_PROVIDER": "dashscope",
            "INTENT_EMBEDDING_MODEL": "text-embedding-v3",
        }, clear=False):
            recognizer = IntentRecognizer(api_key="llm-key", base_url="https://api.deepseek.com/anthropic")

        with patch("core.intent_recognizer.TextEmbedding.call", side_effect=RuntimeError("dashscope down")):
            vector = asyncio.run(recognizer._embed_text("退款多久到账"))

        self.assertEqual(len(vector), 256)
        self.assertTrue(any(value != 0.0 for value in vector))


if __name__ == "__main__":
    unittest.main()
