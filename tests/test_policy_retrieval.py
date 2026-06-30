import importlib.util
import pathlib
import sys
import types
import unittest


def _load_knowledge_base_module():
    if "chromadb" not in sys.modules:
        chromadb_stub = types.ModuleType("chromadb")
        chromadb_stub.HttpClient = object
        chromadb_stub.PersistentClient = object
        chromadb_stub.Settings = object
        sys.modules["chromadb"] = chromadb_stub

    module_path = pathlib.Path(__file__).resolve().parents[1] / "mcp" / "knowledge_base.py"
    spec = importlib.util.spec_from_file_location("knowledge_base_policy_test_module", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


knowledge_base_module = _load_knowledge_base_module()
KnowledgeBase = knowledge_base_module.KnowledgeBase


class InMemoryCollection:
    def __init__(self):
        self.records = {}

    def add(self, ids, documents, metadatas):
        for item_id, document, metadata in zip(ids, documents, metadatas):
            self.records[str(item_id)] = {
                "document": document,
                "metadata": dict(metadata),
            }

    def get(self, ids=None, where=None, include=None):
        items = list(self.records.items())
        if ids is not None:
            wanted = {str(item_id) for item_id in ids}
            items = [item for item in items if item[0] in wanted]
        if where:
            items = [item for item in items if self._matches(item[1]["metadata"], where)]
        return {
            "ids": [[item_id for item_id, _ in items]],
            "documents": [[entry["document"] for _, entry in items]],
            "metadatas": [[entry["metadata"] for _, entry in items]],
        }

    def delete(self, ids=None, where=None):
        if ids is not None:
            for item_id in ids:
                self.records.pop(str(item_id), None)
            return
        if where:
            doomed = [
                item_id
                for item_id, entry in self.records.items()
                if self._matches(entry["metadata"], where)
            ]
            for item_id in doomed:
                self.records.pop(item_id, None)

    def count(self):
        return len(self.records)

    @staticmethod
    def _matches(metadata, where):
        for key, value in where.items():
            if metadata.get(key) != value:
                return False
        return True


class KnowledgeBasePolicyRetrievalTests(unittest.TestCase):
    def _make_kb(self):
        kb = KnowledgeBase.__new__(KnowledgeBase)
        kb._child_records = []
        kb._bm25_doc_tokens = []
        kb._bm25_term_freqs = []
        kb._bm25_doc_freq = {}
        kb._bm25_avgdl = 0.0
        kb._hybrid_recall_k = 20
        kb._rrf_k = 60
        kb._rerank_api_key = ""
        kb._rerank_model = "qwen3-rerank"
        kb._rerank_instruct = KnowledgeBase.DEFAULT_RERANK_INSTRUCT
        kb._parent_collection = InMemoryCollection()
        kb._child_collection = InMemoryCollection()
        kb._collection = kb._child_collection
        return kb

    def test_resolve_search_plan_always_returns_default_scope(self):
        kb = KnowledgeBase.__new__(KnowledgeBase)

        plan = kb._resolve_search_plan("最新请假制度是什么")

        self.assertEqual(plan["scope"], "active")
        self.assertEqual(plan["filters"], {})

    def test_sanitize_metadata_removes_none_values_for_chroma(self):
        kb = KnowledgeBase.__new__(KnowledgeBase)

        metadata = kb._sanitize_metadata({
            "title": "退款政策",
            "effective_at": None,
            "chunk_index": 0,
            "enabled": True,
        })

        self.assertNotIn("effective_at", metadata)
        self.assertEqual(metadata["title"], "退款政策")
        self.assertEqual(metadata["chunk_index"], 0)
        self.assertTrue(metadata["enabled"])

    def test_search_prefers_explicit_recall_k_over_default_resolution(self):
        kb = KnowledgeBase.__new__(KnowledgeBase)
        kb._resolve_search_plan = lambda query: {"scope": "active", "filters": {}}
        seen = {}

        def fake_vector_recall(query, top_n, scope="active", filters=None):
            seen["vector_top_n"] = top_n
            return [{
                "parent_id": "p1",
                "matched_child_chunk": 0,
                "policy_id": "policy-1",
                "score": 0.9,
            }]

        def fake_bm25_recall(query, top_n, scope="active", filters=None):
            seen["bm25_top_n"] = top_n
            return [{
                "parent_id": "p1",
                "matched_child_chunk": 0,
                "policy_id": "policy-1",
                "score": 0.8,
            }]

        kb._vector_recall = fake_vector_recall
        kb._bm25_recall = fake_bm25_recall
        kb._fuse_recall_results = lambda vector_hits, bm25_hits, top_n: vector_hits + bm25_hits
        kb._rerank_candidates = lambda query, candidates, top_n: candidates[:top_n]
        kb._attach_parent_context = lambda items, scope="active": items

        results = kb.search("退款多久到账", top_k=3, recall_k=11)

        self.assertEqual(seen["vector_top_n"], 11)
        self.assertEqual(seen["bm25_top_n"], 11)
        self.assertEqual(len(results), 2)

    def test_add_documents_writes_single_collection_without_version_metadata(self):
        kb = self._make_kb()

        added = kb.add_documents([{
            "title": "请假制度",
            "content": "员工年度可享受 10 天年假。",
        }])

        self.assertGreater(added, 0)
        self.assertGreater(kb._parent_collection.count(), 0)
        self.assertGreater(kb._child_collection.count(), 0)
        child_meta = next(iter(kb._child_collection.records.values()))["metadata"]
        self.assertNotIn("version_no", child_meta)
        self.assertNotIn("effective_at", child_meta)
        self.assertNotIn("policy_id", child_meta)


if __name__ == "__main__":
    unittest.main()
