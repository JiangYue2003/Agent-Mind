import asyncio
import pathlib
import sys
import types
import unittest
from datetime import datetime
from unittest.mock import AsyncMock, patch


if "chromadb" not in sys.modules:
    chromadb_stub = types.ModuleType("chromadb")
    chromadb_stub.HttpClient = lambda *args, **kwargs: None
    chromadb_stub.PersistentClient = lambda *args, **kwargs: None
    chromadb_stub.Settings = lambda *args, **kwargs: None
    sys.modules["chromadb"] = chromadb_stub

if "redis" not in sys.modules:
    redis_stub = types.ModuleType("redis")
    redis_stub.from_url = lambda *args, **kwargs: None
    sys.modules["redis"] = redis_stub

ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from memory.conversation_memory import MemoryManager, Message, MsgRole


class _RecordingCollection:
    def __init__(self):
        self.add_calls = []
        self.query_calls = []

    def add(self, **kwargs):
        self.add_calls.append(kwargs)

    def delete(self, **kwargs):
        return None

    def query(self, **kwargs):
        self.query_calls.append(kwargs)
        return {"documents": [["历史退款对话"]]}


class _EmbeddingClient:
    def __init__(self):
        self.document_calls = []
        self.query_calls = []

    def embed_documents(self, texts):
        self.document_calls.append(list(texts))
        return [[0.1, 0.2, 0.3, 0.4] for _ in texts]

    def embed_query(self, text):
        self.query_calls.append(text)
        return [0.5, 0.6, 0.7, 0.8]


class _UnavailableEmbeddingClient:
    def embed_query(self, text):
        raise RuntimeError("DashScope unavailable")


class _ProfileResponse:
    content = [type("Content", (), {
        "text": '{"preferences":["物流提醒"],"entities":{"产品":[],"问题类型":[]}}'
    })()]


class _ChromaClient:
    def __init__(self):
        self.collection_calls = []

    def heartbeat(self):
        return None

    def get_or_create_collection(self, name, **kwargs):
        self.collection_calls.append({"name": name, **kwargs})
        return _RecordingCollection()


class MemoryV4EmbeddingTests(unittest.TestCase):
    def _manager(self):
        manager = MemoryManager.__new__(MemoryManager)
        manager._episodic = _RecordingCollection()
        manager._profile = _RecordingCollection()
        manager._embedding_client = _EmbeddingClient()
        manager._model = "test-model"
        manager._client = type("Client", (), {
            "messages": type("Messages", (), {
                "create": AsyncMock(return_value=_ProfileResponse()),
            })(),
        })()
        return manager

    def test_memory_uses_v4_collection_names(self):
        self.assertTrue(hasattr(MemoryManager, "EPISODIC_COLLECTION_NAME"))
        self.assertTrue(hasattr(MemoryManager, "PROFILE_COLLECTION_NAME"))
        self.assertEqual(MemoryManager.EPISODIC_COLLECTION_NAME, "episodic_v4")
        self.assertEqual(MemoryManager.PROFILE_COLLECTION_NAME, "user_profile_v4")

    def test_memory_v4_collections_disable_chroma_default_embedding(self):
        chroma_client = _ChromaClient()
        with patch("memory.conversation_memory.redis.from_url"), patch(
            "memory.conversation_memory.AsyncAnthropic"
        ), patch("memory.conversation_memory.chromadb.HttpClient", return_value=chroma_client), patch(
            "memory.conversation_memory.DashScopeEmbeddingClient.from_env",
            return_value=_EmbeddingClient(),
        ):
            MemoryManager()

        self.assertEqual(
            [call["name"] for call in chroma_client.collection_calls],
            ["episodic_v4", "user_profile_v4"],
        )
        self.assertTrue(all("embedding_function" in call for call in chroma_client.collection_calls))
        self.assertTrue(all(call["embedding_function"] is None for call in chroma_client.collection_calls))

    def test_update_profile_writes_explicit_document_embedding(self):
        manager = self._manager()
        manager._get_working_memory = AsyncMock(return_value=[
            Message(role=MsgRole.USER, content="请提醒我物流状态", timestamp=datetime.now()),
        ])

        asyncio.run(manager.update_profile("user-1", "conv-1"))

        call = manager._profile.add_calls[0]
        self.assertEqual(call.get("embeddings"), [[0.1, 0.2, 0.3, 0.4]])
        self.assertEqual(manager._embedding_client.document_calls, [call["documents"]])

    def test_store_episodic_writes_explicit_document_embedding(self):
        manager = self._manager()

        asyncio.run(manager._store_episodic("user-1", "conv-1", "完整对话", "摘要"))

        call = manager._episodic.add_calls[0]
        self.assertEqual(call.get("embeddings"), [[0.1, 0.2, 0.3, 0.4]])
        self.assertEqual(manager._embedding_client.document_calls, [call["documents"]])

    def test_search_episodic_uses_explicit_query_embedding(self):
        manager = self._manager()

        results = asyncio.run(manager._search_episodic("user-1", "退款多久到账"))

        self.assertEqual(results, ["历史退款对话"])
        call = manager._episodic.query_calls[0]
        self.assertEqual(call.get("query_embeddings"), [[0.5, 0.6, 0.7, 0.8]])
        self.assertNotIn("query_texts", call)

    def test_search_episodic_returns_no_vector_results_when_embedding_unavailable(self):
        manager = self._manager()
        manager._embedding_client = _UnavailableEmbeddingClient()

        results = asyncio.run(manager._search_episodic("user-1", "退款多久到账"))

        self.assertEqual(results, [])
        self.assertEqual(manager._episodic.query_calls, [])


if __name__ == "__main__":
    unittest.main()
