import unittest

from fastapi.testclient import TestClient

from api import main as api_main


class _FakeSearchResult:
    def __init__(self, data):
        self.success = True
        self.data = data
        self.reranked = True


class _FakeToolManager:
    def __init__(self):
        self._tools = {}

    async def search_with_rewrite(self, tool_name, message, top_k=3):
        return _FakeSearchResult([
            {
                "title": "退款政策",
                "heading_path": "制度中心 > 财务 > 退款政策",
                "score": 0.9321,
                "matched_child_content": "审核通过后，款项将在 5-7 个工作日内退回原支付账户。",
                "parent_content": "退款政策说明。审核通过后，款项将在 5-7 个工作日内退回原支付账户。",
                "version_no": "V2",
                "effective_at": "2025-01-01",
            }
        ])


class _FakeKnowledgeBase:
    def __init__(self):
        self._child_records = [
            {
                "title": "退款政策",
                "policy_id": "policy-refund",
                "version_id": "version-refund-v2",
                "version_no": "V2",
                "effective_at": "2025-01-01",
                "parent_id": "version-refund-v2:parent:0",
                "child_chunk_index": 0,
                "heading_path": "制度中心 > 财务 > 退款政策",
                "content": "审核通过后，款项将在 5-7 个工作日内退回原支付账户。",
            }
        ]
        self._archive_child_records = [
            {
                "title": "退款政策",
                "policy_id": "policy-refund",
                "version_id": "version-refund-v1",
                "version_no": "V1",
                "effective_at": "2024-01-01",
                "parent_id": "version-refund-v1:parent:0",
                "child_chunk_index": 0,
                "heading_path": "制度中心 > 财务 > 退款政策",
                "content": "旧版退款到账时间为 7-10 个工作日。",
            }
        ]

    async def search_handler(self, params, context):
        return []


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

    async def test_build_knowledge_context_includes_version_tags(self):
        context, used = await api_main._build_knowledge_context("退款多久到账", top_k=3)

        self.assertTrue(used)
        self.assertIn("版本号: V2", context)
        self.assertIn("生效时间: 2025-01-01", context)
        self.assertIn("命中片段: 审核通过后，款项将在 5-7 个工作日内退回原支付账户。", context)

    async def test_export_knowledge_chunks_returns_chunk_metadata(self):
        client = TestClient(api_main.app)

        response = client.get("/knowledge/chunks")

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["total"], 1)
        self.assertEqual(payload["scope"], "active")
        self.assertEqual(payload["items"][0]["parent_id"], "version-refund-v2:parent:0")
        self.assertEqual(payload["items"][0]["child_chunk_index"], 0)
        self.assertEqual(payload["items"][0]["version_no"], "V2")
        self.assertEqual(payload["items"][0]["effective_at"], "2025-01-01")
        self.assertIn("退回原支付账户", payload["items"][0]["content"])


if __name__ == "__main__":
    unittest.main()
