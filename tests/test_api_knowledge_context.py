import os
import unittest
from unittest import mock

from fastapi.testclient import TestClient

from api import main as api_main


class _FakeSearchResult:
    def __init__(self, data):
        self.success = True
        self.data = data
        self.reranked = True
        self.retrieval_debug = {
            "raw_candidates": [{"chunk_key": "refund-doc:parent:0:0", "rank": 1, "score": 0.5}],
            "reranked_candidates": [{"chunk_key": "refund-doc:parent:0:0", "rank": 1, "score": 0.9}],
        }


class _FakeToolManager:
    def __init__(self, result_data=None):
        self._tools = {}
        self.calls = []
        self.result_data = result_data

    async def search_with_rewrite(self, tool_name, message, top_k=3, recall_k=None, context=None):
        self.calls.append({"tool_name": tool_name, "message": message, "top_k": top_k})
        return _FakeSearchResult(self.result_data or [
            {
                "title": "退款政策",
                "heading_path": "制度中心 > 财务 > 退款政策",
                "score": 0.9321,
                "matched_child_content": "审核通过后，款项将在 5-7 个工作日内退回原支付账户。",
                "parent_content": "退款政策说明。审核通过后，款项将在 5-7 个工作日内退回原支付账户。",
            }
        ])


class _FakeKnowledgeBase:
    def __init__(self):
        self._child_records = [
            {
                "title": "退款政策",
                "doc_id": "refund-doc",
                "parent_id": "refund-doc:parent:0",
                "child_chunk_index": 0,
                "heading_path": "制度中心 > 财务 > 退款政策",
                "content": "审核通过后，款项将在 5-7 个工作日内退回原支付账户。",
            }
        ]
        self.replaced_documents = None

    async def search_handler(self, params, context):
        return []

    def replace_documents(self, documents):
        self.replaced_documents = documents
        return 2

    @property
    def doc_count(self):
        return 2


class _FakeKnowledgeTool:
    def __init__(self, kb):
        self.handler = kb.search_handler


class BuildKnowledgeContextTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self._original_tool_manager = api_main._tool_manager
        tool_manager = _FakeToolManager()
        self._fake_kb = _FakeKnowledgeBase()
        tool_manager._tools["knowledge_search"] = _FakeKnowledgeTool(self._fake_kb)
        api_main._tool_manager = tool_manager

    def tearDown(self):
        api_main._tool_manager = self._original_tool_manager

    async def test_build_knowledge_context_omits_version_tags(self):
        context, used = await api_main._build_knowledge_context("退款多久到账", top_k=3)

        self.assertTrue(used)
        self.assertNotIn("版本号:", context)
        self.assertNotIn("生效时间:", context)
        self.assertIn("命中片段: 审核通过后，款项将在 5-7 个工作日内退回原支付账户。", context)

    async def test_build_knowledge_context_uses_answer_top_k_default(self):
        with mock.patch.dict(os.environ, {"RAG_ANSWER_TOP_K": "5"}):
            context, used = await api_main._build_knowledge_context("退款多久到账")

        self.assertTrue(used)
        self.assertTrue(context)
        self.assertEqual(api_main._tool_manager.calls[-1]["top_k"], 5)

    async def test_build_knowledge_context_deduplicates_parents_only_when_building_prompt(self):
        manager = _FakeToolManager(result_data=[
            {
                "title": "退款政策",
                "parent_id": "refund-doc:parent:0",
                "heading_path": "制度中心 > 财务 > 退款政策",
                "score": 0.99,
                "matched_child_content": "退款审核阶段说明。",
                "parent_content": "父块 A：退款审核阶段和到账规则。",
            },
            {
                "title": "退款政策",
                "parent_id": "refund-doc:parent:0",
                "heading_path": "制度中心 > 财务 > 退款政策",
                "score": 0.98,
                "matched_child_content": "退款到账阶段说明。",
                "parent_content": "父块 A：退款审核阶段和到账规则。",
            },
            {
                "title": "发票规则",
                "parent_id": "invoice-doc:parent:0",
                "heading_path": "制度中心 > 财务 > 发票规则",
                "score": 0.97,
                "matched_child_content": "发票申请条件。",
                "parent_content": "父块 B：发票申请和补开规则。",
            },
        ])
        manager._tools["knowledge_search"] = _FakeKnowledgeTool(self._fake_kb)
        api_main._tool_manager = manager

        context, used = await api_main._build_knowledge_context("退款多久到账", top_k=5)

        self.assertTrue(used)
        self.assertEqual(context.count("父块 A：退款审核阶段和到账规则。"), 1)
        self.assertEqual(context.count("父块 B：发票申请和补开规则。"), 1)
        self.assertNotIn("退款到账阶段说明。", context)

    async def test_export_knowledge_chunks_returns_chunk_metadata(self):
        client = TestClient(api_main.app)

        response = client.get("/knowledge/chunks")

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["total"], 1)
        self.assertEqual(payload["items"][0]["parent_id"], "refund-doc:parent:0")
        self.assertEqual(payload["items"][0]["child_chunk_index"], 0)
        self.assertNotIn("version_no", payload["items"][0])
        self.assertNotIn("effective_at", payload["items"][0])
        self.assertIn("退回原支付账户", payload["items"][0]["content"])

    async def test_replace_knowledge_rebuilds_the_active_knowledge_base(self):
        client = TestClient(api_main.app)

        response = client.post("/knowledge/replace", json={
            "documents": [{"title": "退款政策", "content": "新的退款规则。"}],
        })

        self.assertEqual(response.status_code, 200)
        self.assertEqual(self._fake_kb.replaced_documents, [{
            "title": "退款政策",
            "content": "新的退款规则。",
        }])
        self.assertEqual(response.json()["total_chunks"], 2)

    async def test_search_exposes_retrieval_diagnostics_only_when_requested(self):
        client = TestClient(api_main.app)

        hidden_response = client.post("/search", params={"query": "退款多久到账", "top_k": 3, "recall_k": 15})
        debug_response = client.post(
            "/search",
            params={"query": "退款多久到账", "top_k": 3, "recall_k": 15, "include_debug": True},
        )

        self.assertEqual(hidden_response.status_code, 200)
        self.assertNotIn("retrieval_debug", hidden_response.json())
        self.assertEqual(debug_response.status_code, 200)
        self.assertEqual(
            debug_response.json()["retrieval_debug"]["raw_candidates"][0]["chunk_key"],
            "refund-doc:parent:0:0",
        )


if __name__ == "__main__":
    unittest.main()
