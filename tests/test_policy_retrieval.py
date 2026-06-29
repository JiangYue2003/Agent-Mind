import importlib.util
import os
import pathlib
import sys
import tempfile
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
    def _make_kb(self, temp_dir):
        kb = KnowledgeBase.__new__(KnowledgeBase)
        kb._child_records = []
        kb._archive_child_records = []
        kb._bm25_doc_tokens = []
        kb._bm25_term_freqs = []
        kb._bm25_doc_freq = {}
        kb._bm25_avgdl = 0.0
        kb._archive_bm25_doc_tokens = []
        kb._archive_bm25_term_freqs = []
        kb._archive_bm25_doc_freq = {}
        kb._archive_bm25_avgdl = 0.0
        kb._hybrid_recall_k = 20
        kb._rrf_k = 60
        kb._rerank_api_key = ""
        kb._rerank_model = "qwen3-rerank"
        kb._rerank_instruct = KnowledgeBase.DEFAULT_RERANK_INSTRUCT
        kb._policy_catalog = knowledge_base_module.PolicyCatalog(
            os.path.join(temp_dir, "policy_catalog.sqlite3")
        )
        kb._active_parent_collection = InMemoryCollection()
        kb._active_child_collection = InMemoryCollection()
        kb._archive_parent_collection = InMemoryCollection()
        kb._archive_child_collection = InMemoryCollection()
        kb._parent_collection = kb._active_parent_collection
        kb._child_collection = kb._active_child_collection
        kb._collection = kb._active_child_collection
        return kb

    def test_add_documents_archives_previous_active_version_for_same_policy(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            kb = self._make_kb(temp_dir)

            kb.add_documents([{
                "title": "请假制度",
                "content": "员工年度可享受 5 天年假。",
                "policy_id": "policy-leave",
                "version_no": "v1",
                "issue_code": "人资〔2024〕15号",
                "effective_at": "2024-03-01",
            }])
            kb.add_documents([{
                "title": "请假制度",
                "content": "员工年度可享受 10 天年假。",
                "policy_id": "policy-leave",
                "version_no": "v2",
                "issue_code": "人资〔2025〕02号",
                "effective_at": "2025-01-01",
            }])

            active_versions = {
                entry["metadata"].get("version_no")
                for entry in kb._active_child_collection.records.values()
            }
            archive_versions = {
                entry["metadata"].get("version_no")
                for entry in kb._archive_child_collection.records.values()
            }

            self.assertEqual(active_versions, {"v2"})
            self.assertIn("v1", archive_versions)

            active_version = kb._policy_catalog.get_active_version("policy-leave")
            self.assertIsNotNone(active_version)
            self.assertEqual(active_version["version_no"], "v2")

    def test_resolve_search_plan_defaults_to_active_scope(self):
        kb = KnowledgeBase.__new__(KnowledgeBase)
        kb._policy_catalog = types.SimpleNamespace(
            resolve_versions=lambda target: [],
            match_policy_ids=lambda query: [],
        )

        plan = kb._resolve_search_plan("最新请假制度是什么")

        self.assertEqual(plan["scope"], "active")
        self.assertEqual(plan["filters"], {})

    def test_resolve_search_plan_uses_archive_scope_for_explicit_history_query(self):
        kb = KnowledgeBase.__new__(KnowledgeBase)
        kb._policy_catalog = types.SimpleNamespace(
            resolve_versions=lambda target: [{
                "version_id": "leave-v1",
                "policy_id": "policy-leave",
                "title": "请假制度",
                "version_no": "v1",
                "issue_code": "人资〔2024〕15号",
                "effective_at": "2024-03-01",
                "is_active": 0,
            }],
            match_policy_ids=lambda query: ["policy-leave"],
        )

        plan = kb._resolve_search_plan("请给我看人资〔2024〕15号的请假制度")

        self.assertEqual(plan["scope"], "archive")
        self.assertEqual(plan["filters"]["version_ids"], ["leave-v1"])

    def test_sanitize_metadata_removes_none_values_for_chroma(self):
        kb = KnowledgeBase.__new__(KnowledgeBase)

        metadata = kb._sanitize_metadata({
            "title": "退款政策",
            "policy_id": "policy-refund",
            "version_id": "v1",
            "effective_at": None,
            "chunk_index": 0,
            "enabled": True,
        })

        self.assertNotIn("effective_at", metadata)
        self.assertEqual(metadata["title"], "退款政策")
        self.assertEqual(metadata["chunk_index"], 0)
        self.assertTrue(metadata["enabled"])


if __name__ == "__main__":
    unittest.main()
