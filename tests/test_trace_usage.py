import json
import tempfile
import unittest

from agents.agent_orchestrator import GeneralAgent, Request
from core.intent_recognizer import IntentCategory, IntentRecognizer
from mcp.tool_manager import MCPToolManager
from telemetry.runtime import TraceContext, TraceStore


class _FakeContentBlock:
    def __init__(self, text: str):
        self.text = text


class _FakeReasoningDetails:
    def __init__(self, reasoning_tokens: int):
        self.reasoning_tokens = reasoning_tokens


class _FakeUsage:
    def __init__(self):
        self.prompt_tokens = 11
        self.completion_tokens = 7
        self.total_tokens = 18
        self.prompt_cache_hit_tokens = 3
        self.prompt_cache_miss_tokens = 8
        self.completion_tokens_details = _FakeReasoningDetails(reasoning_tokens=2)


class _FakeResponse:
    def __init__(self):
        self.content = [_FakeContentBlock("测试回复")]
        self.usage = _FakeUsage()


class _FakeMessages:
    def __init__(self):
        self.calls = []

    async def create(self, **kwargs):
        self.calls.append(kwargs)
        return _FakeResponse()


class _FakeClient:
    def __init__(self):
        self.messages = _FakeMessages()


class _FakeIntentResponse:
    def __init__(self):
        self.content = [_FakeContentBlock('{"intent":"billing","confidence":0.91,"reasoning":"涉及退款"}')]
        self.usage = _FakeUsage()


class _FakeIntentMessages:
    async def create(self, **kwargs):
        return _FakeIntentResponse()


class _FakeIntentClient:
    def __init__(self):
        self.messages = _FakeIntentMessages()


class _FakeRewriteResponse:
    def __init__(self):
        self.content = [_FakeContentBlock('["退款多久到账","退款几天能到"]')]
        self.usage = _FakeUsage()


class _FakeRerankResponse:
    def __init__(self):
        self.content = [_FakeContentBlock("[1,0]")]
        self.usage = _FakeUsage()


class _FakeToolMessages:
    def __init__(self):
        self.calls = 0

    async def create(self, **kwargs):
        self.calls += 1
        if self.calls == 1:
            return _FakeRewriteResponse()
        return _FakeRerankResponse()


class _FakeToolClient:
    def __init__(self):
        self.messages = _FakeToolMessages()


class TraceUsageTests(unittest.IsolatedAsyncioTestCase):
    async def test_agent_llm_stage_records_usage_tokens(self):
        agent = GeneralAgent(_FakeClient(), "deepseek-chat")
        req = Request(
            message="退款多久到账",
            user_id="u123",
            conv_id="c123",
            context="背景信息",
        )
        trace = TraceContext(user_id="u123", conv_id="c123", message="退款多久到账")

        result = await agent.handle(req, context={"trace": trace})
        trace.finalize()

        self.assertTrue(result.success)
        stages = trace.to_dict()["stages"]
        llm_stage = next(item for item in stages if item["name"] == "agent.llm_call.general")
        usage = llm_stage["meta"]["usage"]
        self.assertEqual(usage["prompt_tokens"], 11)
        self.assertEqual(usage["completion_tokens"], 7)
        self.assertEqual(usage["total_tokens"], 18)
        self.assertEqual(usage["reasoning_tokens"], 2)

    async def test_knowledge_answer_prompt_requires_grounded_short_response(self):
        client = _FakeClient()
        agent = GeneralAgent(client, "deepseek-chat")
        req = Request(
            message="现货付款后多久发货？",
            user_id="u123",
            conv_id="c123",
            context="[知识库检索结果]\n现货订单通常在支付成功后 24-48 小时内发货。",
        )

        await agent.handle(req)

        call = client.messages.calls[0]
        self.assertEqual(call["max_tokens"], 450)
        self.assertIn("仅依据", call["system"])
        self.assertIn("先给出直接结论", call["system"])
        self.assertIn("不要补充", call["system"])
        self.assertIn("不是实时订单事实", call["system"])

    def test_trace_store_writes_jsonl_when_directory_enabled(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            store = TraceStore(capacity=10, jsonl_dir=tmpdir)
            trace = TraceContext(user_id="u123", conv_id="c123", message="退款多久到账")
            with trace.stage("chat.total"):
                pass
            trace.finalize()

            store.save(trace)

            files = list(store._jsonl_dir.glob("*.jsonl"))
            self.assertEqual(len(files), 1)
            payload = json.loads(files[0].read_text(encoding="utf-8").strip())
            self.assertEqual(payload["trace_id"], trace.trace_id)

    async def test_intent_recognizer_llm_stage_records_usage_tokens(self):
        recognizer = IntentRecognizer.__new__(IntentRecognizer)
        recognizer.client = _FakeIntentClient()
        recognizer.model = "deepseek-chat"
        recognizer.threshold = 0.5
        recognizer._embedding_enabled = False
        recognizer._tpl_embeddings = {}
        recognizer._cache = {}
        recognizer.cache_hits = 0
        recognizer.cache_misses = 0

        trace = TraceContext(user_id="u123", conv_id="c123", message="我要退款")
        payload = await recognizer._llm_recognize("我要退款", history=None, trace=trace)
        trace.finalize()

        self.assertEqual(payload["intent"], IntentCategory.BILLING)
        llm_stage = next(item for item in trace.to_dict()["stages"] if item["name"] == "intent.llm_recognize")
        usage = llm_stage["meta"]["usage"]
        self.assertEqual(usage["prompt_tokens"], 11)
        self.assertEqual(usage["completion_tokens"], 7)

    async def test_tool_manager_rewrite_and_rerank_record_usage_tokens(self):
        manager = MCPToolManager.__new__(MCPToolManager)
        manager._client = _FakeToolClient()
        manager._model = "deepseek-chat"
        manager._tools = {}
        manager._cache = {}

        trace = TraceContext(user_id="u123", conv_id="c123", message="退款多久到账")

        queries = await manager.rewrite_query("退款多久到账", trace=trace)
        reranked = await manager._rerank(
            "退款多久到账",
            [{"content": "A"}, {"content": "B"}],
            top_k=1,
            trace=trace,
        )
        trace.finalize()

        self.assertEqual(len(queries), 2)
        self.assertEqual(len(reranked), 1)
        stages = trace.to_dict()["stages"]
        rewrite_stage = next(item for item in stages if item["name"] == "tool_manager.rewrite_query_llm")
        rerank_stage = next(item for item in stages if item["name"] == "tool_manager.outer_rerank_llm")
        self.assertEqual(rewrite_stage["meta"]["usage"]["prompt_tokens"], 11)
        self.assertEqual(rerank_stage["meta"]["usage"]["completion_tokens"], 7)


if __name__ == "__main__":
    unittest.main()
