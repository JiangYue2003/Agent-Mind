import asyncio
import importlib.util
import pathlib
import types
import unittest


def _load_tool_manager_module():
    module_path = pathlib.Path(__file__).resolve().parents[1] / "mcp" / "tool_manager.py"
    spec = importlib.util.spec_from_file_location("tool_manager_test_module", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


tool_manager_module = _load_tool_manager_module()
MCPToolManager = tool_manager_module.MCPToolManager
ToolResult = tool_manager_module.ToolResult


class ToolManagerDedupTests(unittest.TestCase):
    def test_rewrite_query_prompt_prefers_conservative_entity_preserving_rewrites(self):
        manager = MCPToolManager.__new__(MCPToolManager)

        captured = {}

        class FakeContent:
            text = '["退款多久到账"]'

        class FakeResponse:
            content = [FakeContent()]

        class FakeMessages:
            async def create(self, **kwargs):
                captured.update(kwargs)
                return FakeResponse()

        class FakeClient:
            def __init__(self):
                self.messages = FakeMessages()

        manager._client = FakeClient()
        manager._model = "deepseek-chat"

        queries = asyncio.run(manager.rewrite_query("退款多久到账", n=3))

        prompt = captured["messages"][0]["content"]
        self.assertIn("不要引入原问题中没有的新条件", prompt)
        self.assertIn("保留订单号、金额、日期、制度名、版本号等关键实体", prompt)
        self.assertIn("如果原问题已经清晰，不要为了凑数而过度改写", prompt)
        self.assertEqual(queries, ["退款多久到账"])

    def test_rewrite_query_keeps_original_query_and_dedups(self):
        manager = MCPToolManager.__new__(MCPToolManager)

        class FakeContent:
            text = '["退款多久到账", "退款多久到账", "退款审核通过后多久到账"]'

        class FakeResponse:
            content = [FakeContent()]

        class FakeMessages:
            async def create(self, **kwargs):
                return FakeResponse()

        class FakeClient:
            def __init__(self):
                self.messages = FakeMessages()

        manager._client = FakeClient()
        manager._model = "deepseek-chat"

        queries = asyncio.run(manager.rewrite_query("退款多久到账", n=3))

        self.assertEqual(
            queries,
            ["退款多久到账", "退款审核通过后多久到账"],
        )

    def test_search_with_rewrite_dedups_identical_child_across_subqueries(self):
        manager = MCPToolManager.__new__(MCPToolManager)
        seen_params = []

        async def fake_rewrite_query(query: str, n: int = 3):
            return ["q1", "q2", "q3"]

        async def fake_call(tool_name, params, context=None, use_cache=True, rerank_top_k=0):
            seen_params.append(dict(params))
            scores = {"q1": 0.91, "q2": 0.83, "q3": 0.72}
            query = params["query"]
            return ToolResult(
                success=True,
                data=[{
                    "title": "退款政策",
                    "content": "退款将在审核通过后 5-7 个工作日内退回原账户。",
                    "score": scores[query],
                    "chunk": 0,
                    "doc_id": "doc-1",
                    "parent_id": "parent-1",
                    "section_title": "到账时效",
                    "heading_path": "退款政策 > 到账时效",
                }],
                tool_name=tool_name,
            )

        async def fake_rerank(query, items, top_k):
            return items[:top_k]

        manager.rewrite_query = fake_rewrite_query
        manager.call = fake_call
        manager._rerank = fake_rerank

        result = asyncio.run(manager.search_with_rewrite("knowledge_search", "退款多久到账", top_k=3))

        self.assertTrue(result.success)
        self.assertEqual(len(result.data), 1)
        self.assertEqual(result.data[0]["parent_id"], "parent-1")
        self.assertTrue(all(item.get("top_k") == 3 for item in seen_params))
        self.assertTrue(all(item.get("recall_k") == 20 for item in seen_params))

    def test_search_with_rewrite_keeps_distinct_children_from_one_parent_for_global_rerank(self):
        manager = MCPToolManager.__new__(MCPToolManager)
        rerank_inputs = []

        async def fake_rewrite_query(query: str, n: int = 3):
            return ["q1", "q2", "q3"]

        async def fake_call(tool_name, params, context=None, use_cache=True, rerank_top_k=0):
            child_index = {"q1": 0, "q2": 1, "q3": 0}[params["query"]]
            return ToolResult(
                success=True,
                data=[{
                    "title": "退款政策",
                    "content": f"退款规则子块 {child_index}",
                    "score": 0.9 - child_index * 0.1,
                    "doc_id": "doc-1",
                    "parent_id": "doc-1:parent:0",
                    "matched_child_chunk": child_index,
                }],
                tool_name=tool_name,
            )

        async def fake_local_rerank(query, items, top_k, context=None):
            rerank_inputs.extend(items)
            return [
                {
                    **item,
                    "rerank_applied": True,
                    "rerank_provider": "tei",
                    "rerank_score": item["score"],
                }
                for item in items[:top_k]
            ]

        manager.rewrite_query = fake_rewrite_query
        manager.call = fake_call
        manager._tools = {"knowledge_search": types.SimpleNamespace(rerank_handler=fake_local_rerank)}

        result = asyncio.run(manager.search_with_rewrite("knowledge_search", "退款多久到账", top_k=5))

        self.assertTrue(result.success)
        self.assertEqual([item["matched_child_chunk"] for item in rerank_inputs], [0, 1])
        self.assertEqual([item["matched_child_chunk"] for item in result.data], [0, 1])

    def test_search_with_rewrite_skips_outer_rerank_when_tool_handles_it(self):
        manager = MCPToolManager.__new__(MCPToolManager)

        async def fake_rewrite_query(query: str, n: int = 3):
            return [query]

        async def fake_call(tool_name, params, context=None, use_cache=True, rerank_top_k=0):
            return ToolResult(
                success=True,
                data=[
                    {
                        "title": "退款政策",
                        "content": "审核通过后 5-7 个工作日内退款会退回原支付账户。",
                        "score": 0.98,
                        "rerank_score": 0.98,
                        "rerank_applied": True,
                        "rerank_provider": "tei",
                        "chunk": 0,
                        "doc_id": "doc-1",
                        "parent_id": "parent-1",
                        "section_title": "到账时效",
                        "heading_path": "退款政策 > 到账时效",
                    }
                ],
                tool_name=tool_name,
                reranked=True,
            )

        async def fail_if_called(query, items, top_k):
            raise AssertionError("outer rerank should not be called")

        manager.rewrite_query = fake_rewrite_query
        manager.call = fake_call
        manager._rerank = fail_if_called

        result = asyncio.run(manager.search_with_rewrite("knowledge_search", "退款多久到账", top_k=1))

        self.assertTrue(result.success)
        self.assertEqual(result.data[0]["parent_id"], "parent-1")
        self.assertTrue(result.reranked)

    def test_search_with_rewrite_passes_explicit_recall_k_to_tool(self):
        manager = MCPToolManager.__new__(MCPToolManager)
        seen_params = []

        async def fake_rewrite_query(query: str, n: int = 3):
            return ["q1", "q2"]

        async def fake_call(tool_name, params, context=None, use_cache=True, rerank_top_k=0):
            seen_params.append(dict(params))
            return ToolResult(
                success=True,
                data=[{
                    "title": "退款政策",
                    "content": "退款将在审核通过后 5-7 个工作日内退回原账户。",
                    "score": 0.91,
                    "rerank_score": 0.91,
                    "rerank_applied": True,
                    "rerank_provider": "tei",
                    "chunk": 0,
                    "doc_id": "doc-1",
                    "parent_id": "parent-1",
                }],
                tool_name=tool_name,
                reranked=True,
            )

        async def fail_if_called(query, items, top_k):
            raise AssertionError("outer rerank should not be called")

        manager.rewrite_query = fake_rewrite_query
        manager.call = fake_call
        manager._rerank = fail_if_called

        result = asyncio.run(
            manager.search_with_rewrite("knowledge_search", "退款多久到账", top_k=3, recall_k=12)
        )

        self.assertTrue(result.success)
        self.assertEqual(len(seen_params), 2)
        self.assertTrue(all(item.get("top_k") == 3 for item in seen_params))
        self.assertTrue(all(item.get("recall_k") == 12 for item in seen_params))

    def test_search_with_rewrite_does_not_treat_rrf_score_as_real_reranking(self):
        manager = MCPToolManager.__new__(MCPToolManager)
        outer_rerank_calls = []

        async def fake_rewrite_query(query: str, n: int = 3):
            return [query]

        async def fake_call(tool_name, params, context=None, use_cache=True, rerank_top_k=0):
            return ToolResult(
                success=True,
                data=[
                    {
                        "title": "退款政策",
                        "content": "退款将在审核通过后 5-7 个工作日内退回原账户。",
                        "score": 0.91,
                        "rerank_score": 0.91,
                        "rerank_applied": False,
                        "rerank_provider": "rrf_fallback",
                        "parent_id": "parent-1",
                    },
                    {
                        "title": "物流说明",
                        "content": "标准配送需要 3-5 个工作日。",
                        "score": 0.42,
                        "rerank_score": 0.42,
                        "rerank_applied": False,
                        "rerank_provider": "rrf_fallback",
                        "parent_id": "parent-2",
                    },
                ],
                tool_name=tool_name,
            )

        async def fake_rerank(query, items, top_k):
            outer_rerank_calls.append((query, items, top_k))
            return items[:top_k]

        manager.rewrite_query = fake_rewrite_query
        manager.call = fake_call
        manager._rerank = fake_rerank

        result = asyncio.run(manager.search_with_rewrite("knowledge_search", "退款多久到账", top_k=1))

        self.assertEqual(len(outer_rerank_calls), 1)
        self.assertFalse(result.reranked)

    def test_search_with_rewrite_reranks_merged_candidates_once_with_the_knowledge_tool(self):
        manager = MCPToolManager.__new__(MCPToolManager)
        seen_params = []
        rerank_calls = []

        async def fake_rewrite_query(query: str, n: int = 3):
            return ["q1", "q2", "q3"]

        async def fake_call(tool_name, params, context=None, use_cache=True, rerank_top_k=0):
            seen_params.append(dict(params))
            return ToolResult(
                success=True,
                data=[{
                    "title": params["query"],
                    "content": f"内容 {params['query']}",
                    "score": 0.5,
                    "parent_id": params["query"],
                    "rerank_applied": False,
                    "rerank_provider": "rrf_pending",
                }],
                tool_name=tool_name,
            )

        async def fake_local_rerank(query, items, top_k, context=None):
            rerank_calls.append((query, items, top_k, context))
            return [
                {
                    **items[2],
                    "rerank_applied": True,
                    "rerank_provider": "tei",
                    "rerank_score": 0.9,
                    "score": 0.9,
                }
            ]

        async def fail_if_called(*args, **kwargs):
            raise AssertionError("LLM fallback rerank should not run when a tool reranker exists")

        manager.rewrite_query = fake_rewrite_query
        manager.call = fake_call
        manager._rerank = fail_if_called
        manager._tools = {"knowledge_search": types.SimpleNamespace(rerank_handler=fake_local_rerank)}

        result = asyncio.run(manager.search_with_rewrite("knowledge_search", "退款多久到账", top_k=1))

        self.assertEqual(len(rerank_calls), 1)
        self.assertEqual(rerank_calls[0][0], "退款多久到账")
        self.assertEqual(rerank_calls[0][2], 1)
        self.assertTrue(all(item["skip_rerank"] for item in seen_params))
        self.assertTrue(all(item["top_k"] == 20 for item in seen_params))
        self.assertTrue(result.reranked)
        self.assertEqual(result.data[0]["rerank_provider"], "tei")

    def test_search_with_rewrite_exposes_raw_and_reranked_candidate_diagnostics(self):
        manager = MCPToolManager.__new__(MCPToolManager)

        async def fake_rewrite_query(query: str, n: int = 3):
            return [query]

        async def fake_call(tool_name, params, context=None, use_cache=True, rerank_top_k=0):
            return ToolResult(
                success=True,
                data=[
                    {"parent_id": "parent-a", "matched_child_chunk": 0, "score": 0.8},
                    {"parent_id": "parent-b", "matched_child_chunk": 1, "score": 0.6},
                    {"parent_id": "parent-c", "matched_child_chunk": 2, "score": 0.4},
                ],
                tool_name=tool_name,
            )

        async def fake_local_rerank(query, items, top_k, context=None):
            return [
                {**items[2], "score": 0.9, "rerank_score": 0.9, "rerank_applied": True, "rerank_provider": "tei"},
                {**items[0], "score": 0.8, "rerank_score": 0.8, "rerank_applied": True, "rerank_provider": "tei"},
                {**items[1], "score": 0.7, "rerank_score": 0.7, "rerank_applied": True, "rerank_provider": "tei"},
            ][:top_k]

        manager.rewrite_query = fake_rewrite_query
        manager.call = fake_call
        manager._tools = {"knowledge_search": types.SimpleNamespace(rerank_handler=fake_local_rerank)}

        result = asyncio.run(
            manager.search_with_rewrite("knowledge_search", "退款多久到账", top_k=3, recall_k=15)
        )

        self.assertEqual(
            result.retrieval_debug["raw_candidates"],
            [
                {"chunk_key": "parent-a:0", "rank": 1, "score": 0.8},
                {"chunk_key": "parent-b:1", "rank": 2, "score": 0.6},
                {"chunk_key": "parent-c:2", "rank": 3, "score": 0.4},
            ],
        )
        self.assertEqual(
            result.retrieval_debug["reranked_candidates"],
            [
                {"chunk_key": "parent-c:2", "rank": 1, "score": 0.9},
                {"chunk_key": "parent-a:0", "rank": 2, "score": 0.8},
                {"chunk_key": "parent-b:1", "rank": 3, "score": 0.7},
            ],
        )


if __name__ == "__main__":
    unittest.main()
