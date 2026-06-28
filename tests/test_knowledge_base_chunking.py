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
    spec = importlib.util.spec_from_file_location("knowledge_base_test_module", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


knowledge_base_module = _load_knowledge_base_module()
KnowledgeBase = knowledge_base_module.KnowledgeBase


class KnowledgeBaseChunkingTests(unittest.TestCase):
    def setUp(self):
        self.kb = KnowledgeBase.__new__(KnowledgeBase)

    def test_prefers_paragraph_boundaries_before_sentence_split(self):
        text = (
            "# 退款政策\n"
            "第一段第一句。第一段第二句。\n\n"
            "## 审核规则\n"
            "第二段第一句。第二段第二句。"
        )

        chunks = self.kb._chunk_text(text, chunk_size=24, overlap=0)

        self.assertGreaterEqual(len(chunks), 2)
        self.assertIn("第一段第一句", chunks[0])
        self.assertNotIn("第二段第一句", chunks[0])

    def test_splits_long_paragraph_with_multiple_punctuation(self):
        text = "第一句！第二句？Third sentence. Fourth sentence; 第五句；第六句。"

        chunks = self.kb._chunk_text(text, chunk_size=18, overlap=0)

        self.assertGreaterEqual(len(chunks), 3)
        self.assertTrue(any("第二句？" in chunk for chunk in chunks))
        self.assertTrue(any("Third sentence." in chunk for chunk in chunks))

    def test_adds_overlap_between_adjacent_chunks(self):
        text = (
            "第一部分内容比较长，需要拆分成多个块。"
            "第二部分继续补充说明。"
            "第三部分再补充一些细节，确保 overlap 有东西可以继承。"
        )

        chunks = self.kb._chunk_text(text, chunk_size=24, overlap=6)

        self.assertGreaterEqual(len(chunks), 2)
        self.assertTrue(chunks[1].startswith(chunks[0][-6:]))

    def test_build_chunks_extracts_heading_metadata(self):
        text = (
            "# 退款政策\n"
            "退款总则说明。\n\n"
            "## 审核规则\n"
            "审核通过后 1-3 个工作日内处理。"
        )

        chunks = self.kb._build_structured_chunks("退款政策", text)

        self.assertGreaterEqual(len(chunks), 2)
        self.assertEqual(chunks[0]["section_title"], "退款政策")
        self.assertEqual(chunks[1]["section_title"], "审核规则")
        self.assertEqual(chunks[1]["heading_path"], "退款政策 > 审核规则")
        self.assertEqual(chunks[0]["doc_id"], chunks[1]["doc_id"])

    def test_search_returns_structural_metadata(self):
        class FakeCollection:
            def query(self, query_texts, n_results):
                return {
                    "documents": [["审核通过后 1-3 个工作日内处理。"]],
                    "metadatas": [[{
                        "title": "退款政策",
                        "chunk_index": 0,
                        "total_chunks": 1,
                        "doc_id": "doc-1",
                        "section_title": "审核规则",
                        "heading_path": "退款政策 > 审核规则",
                    }]],
                    "distances": [[0.12]],
                }

        self.kb._collection = FakeCollection()

        items = self.kb.search("退款多久到账", top_k=1)

        self.assertEqual(items[0]["doc_id"], "doc-1")
        self.assertEqual(items[0]["section_title"], "审核规则")
        self.assertEqual(items[0]["heading_path"], "退款政策 > 审核规则")

    def test_search_uses_child_hits_to_fetch_parent_chunks(self):
        class FakeChildCollection:
            def query(self, query_texts, n_results):
                return {
                    "documents": [[
                        "到账需要多久",
                        "多久到账的审核规则",
                    ]],
                    "metadatas": [[
                        {
                            "title": "退款政策",
                            "doc_id": "doc-1",
                            "parent_id": "parent-1",
                            "chunk_index": 0,
                            "section_title": "到账时效",
                            "heading_path": "退款政策 > 到账时效",
                            "child_chunk_index": 0,
                        },
                        {
                            "title": "退款政策",
                            "doc_id": "doc-1",
                            "parent_id": "parent-1",
                            "chunk_index": 1,
                            "section_title": "到账时效",
                            "heading_path": "退款政策 > 到账时效",
                            "child_chunk_index": 1,
                        },
                    ]],
                    "distances": [[0.05, 0.08]],
                }

        class FakeParentCollection:
            def get(self, ids):
                return {
                    "ids": ["parent-1"],
                    "documents": [["退款将在审核通过后 5-7 个工作日内退回原账户。"]],
                    "metadatas": [[{
                        "title": "退款政策",
                        "doc_id": "doc-1",
                        "parent_id": "parent-1",
                        "chunk_index": 0,
                        "total_chunks": 1,
                        "section_title": "到账时效",
                        "heading_path": "退款政策 > 到账时效",
                    }]],
                }

        self.kb._collection = FakeChildCollection()
        self.kb._parent_collection = FakeParentCollection()
        self.kb._child_collection = FakeChildCollection()

        items = self.kb.search("退款多久到账", top_k=2)

        self.assertEqual(len(items), 1)
        self.assertEqual(items[0]["content"], "退款将在审核通过后 5-7 个工作日内退回原账户。")
        self.assertEqual(items[0]["matched_child_chunk"], 0)
        self.assertEqual(items[0]["matched_child_content"], "到账需要多久")

    def test_add_documents_writes_parent_and_child_collections(self):
        class RecordingCollection:
            def __init__(self):
                self.calls = []

            def add(self, ids, documents, metadatas):
                self.calls.append({
                    "ids": ids,
                    "documents": documents,
                    "metadatas": metadatas,
                })

            def count(self):
                return sum(len(call["ids"]) for call in self.calls)

        self.kb._parent_collection = RecordingCollection()
        self.kb._child_collection = RecordingCollection()
        self.kb._collection = self.kb._child_collection

        added = self.kb.add_documents([{
            "title": "退款政策",
            "content": "# 退款政策\n退款总则说明。\n\n## 到账时效\n退款将在审核通过后 5-7 个工作日内退回原账户。",
        }])

        self.assertGreaterEqual(added, 3)
        self.assertEqual(len(self.kb._parent_collection.calls), 1)
        self.assertEqual(len(self.kb._child_collection.calls), 1)
        self.assertTrue(all("parent_id" in meta for meta in self.kb._child_collection.calls[0]["metadatas"]))


if __name__ == "__main__":
    unittest.main()
