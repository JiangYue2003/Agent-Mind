import importlib.util
import pathlib
import sys
import types
import unittest
from unittest.mock import patch

from telemetry.runtime import TraceContext


def _load_knowledge_base_module():
    if "chromadb" not in sys.modules:
        chromadb_stub = types.ModuleType("chromadb")
        chromadb_stub.HttpClient = object
        chromadb_stub.PersistentClient = object
        chromadb_stub.Settings = object
        sys.modules["chromadb"] = chromadb_stub

    module_path = pathlib.Path(__file__).resolve().parents[1] / "mcp" / "knowledge_base.py"
    spec = importlib.util.spec_from_file_location("knowledge_base_v4_test_module", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


knowledge_base_module = _load_knowledge_base_module()
KnowledgeBase = knowledge_base_module.KnowledgeBase


class _FakeEmbeddingClient:
    model = "text-embedding-v4"
    dimension = 1024

    def __init__(self):
        self.document_calls = []
        self.query_calls = []

    def embed_documents(self, texts):
        self.document_calls.append(list(texts))
        return [[float(index + 1)] * 4 for index, _ in enumerate(texts)]

    def embed_query(self, text):
        self.query_calls.append(text)
        return [0.1, 0.2, 0.3, 0.4]


class _RecordingCollection:
    def __init__(self):
        self.add_calls = []
        self.query_calls = []

    def add(self, ids, documents, metadatas, embeddings=None):
        self.add_calls.append({
            "ids": ids,
            "documents": documents,
            "metadatas": metadatas,
            "embeddings": embeddings,
        })

    def query(self, **kwargs):
        self.query_calls.append(kwargs)
        return {
            "documents": [["退款将在审核通过后退回原账户。"]],
            "metadatas": [[{
                "title": "退款政策",
                "doc_id": "doc-1",
                "parent_id": "parent-1",
                "chunk_index": 0,
                "child_chunk_index": 0,
                "section_title": "到账时效",
                "heading_path": "退款政策 > 到账时效",
                "total_chunks": 1,
            }]],
            "distances": [[0.1]],
        }

    def count(self):
        return 0


class _ChromaClient:
    def __init__(self):
        self.collection_calls = []

    def heartbeat(self):
        return None

    def get_or_create_collection(self, **kwargs):
        self.collection_calls.append(kwargs)
        return _RecordingCollection()


class KnowledgeBaseV4EmbeddingTests(unittest.TestCase):
    def setUp(self):
        self.kb = KnowledgeBase.__new__(KnowledgeBase)
        self.kb._child_records = []
        self.kb._bm25_doc_tokens = []
        self.kb._bm25_term_freqs = []
        self.kb._bm25_doc_freq = {}
        self.kb._bm25_avgdl = 0.0
        self.kb._embedding_client = _FakeEmbeddingClient()
        self.kb._parent_collection = _RecordingCollection()
        self.kb._child_collection = _RecordingCollection()
        self.kb._collection = self.kb._child_collection

    def test_add_documents_writes_explicit_v4_embeddings(self):
        self.kb.add_documents([{
            "title": "退款政策",
            "content": "退款审核通过后，款项将在五到七个工作日内退回原支付账户。",
        }])

        parent_call = self.kb._parent_collection.add_calls[0]
        child_call = self.kb._child_collection.add_calls[0]
        self.assertEqual(len(parent_call["embeddings"]), len(parent_call["documents"]))
        self.assertEqual(len(child_call["embeddings"]), len(child_call["documents"]))
        self.assertEqual(len(self.kb._embedding_client.document_calls), 2)

    def test_vector_recall_queries_chroma_with_v4_query_embedding(self):
        hits = self.kb._vector_recall("退款多久到账", top_n=3)

        self.assertEqual(len(hits), 1)
        self.assertEqual(self.kb._embedding_client.query_calls, ["退款多久到账"])
        query_call = self.kb._child_collection.query_calls[0]
        self.assertEqual(query_call["query_embeddings"], [[0.1, 0.2, 0.3, 0.4]])
        self.assertNotIn("query_texts", query_call)

    def test_v4_collections_disable_chroma_default_embedding(self):
        chroma_client = _ChromaClient()
        with patch.object(knowledge_base_module.chromadb, "HttpClient", return_value=chroma_client), patch.object(
            knowledge_base_module.DashScopeEmbeddingClient,
            "from_env",
            return_value=_FakeEmbeddingClient(),
        ):
            KnowledgeBase(chroma_host="chroma-test", chroma_port=8000)

        self.assertEqual(len(chroma_client.collection_calls), 2)
        self.assertTrue(all("embedding_function" in call for call in chroma_client.collection_calls))
        self.assertTrue(all(call["embedding_function"] is None for call in chroma_client.collection_calls))

    def test_vector_recall_trace_records_v4_provider_metadata(self):
        self.kb._attach_parent_context = lambda items: items
        trace = TraceContext(user_id="user-1", conv_id="conv-1", message="退款多久到账")

        self.kb.search_with_trace("退款多久到账", top_k=1, trace=trace)

        vector_stage = next(stage for stage in trace._stages if stage.name == "knowledge.vector_recall")
        self.assertEqual(vector_stage.meta, {
            "provider": "dashscope",
            "model": "text-embedding-v4",
            "dimension": 1024,
            "request_size": 1,
            "fallback_used": False,
        })


if __name__ == "__main__":
    unittest.main()
