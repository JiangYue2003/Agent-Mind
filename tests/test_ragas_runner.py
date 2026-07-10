import asyncio
import json
import math
import pathlib
import sys
import tempfile
import types
import unittest
from unittest import mock

import evaluation.ragas_runner as ragas_runner

from evaluation.ragas_runner import (
    AnthropicMessagesRagasLLM,
    build_anthropic_evaluator,
    build_collection_metrics,
    build_openai_embeddings,
    build_ragas_samples,
    collect_deployed_records,
    evaluate_workflow_records,
    load_eval_cases,
    main,
    resume_evaluation_report,
    run_deployed_evaluation,
    run_ragas_evaluation,
    validate_deployed_knowledge_base,
)


class _FakeResponse:
    def __init__(self, payload):
        self._payload = payload

    def raise_for_status(self):
        return None

    def json(self):
        return self._payload


class _FakeHttpClient:
    def __init__(self):
        self.calls = []

    def post(self, path, params=None, json=None):
        self.calls.append({"path": path, "params": params, "json": json})
        if path == "/search":
            return _FakeResponse({
                "results": [
                    {"matched_child_content": "审核通过后，银行卡退款通常在 1-3 个工作日内到账。"},
                    {"content": "退款成功后可查看支付账户明细。"},
                ],
                "rerank_applied": True,
                "rerank_providers": ["tei"],
            })
        if path == "/chat":
            return _FakeResponse({
                "response": "审核通过后，银行卡退款通常在 1-3 个工作日内到账。",
                "knowledge_used": True,
                "latency_ms": 123.4,
                "trace_id": "trace-001",
            })
        raise AssertionError(f"unexpected request: {path}")

    def get(self, path, params=None):
        self.calls.append({"path": path, "params": params, "json": None})
        if path == "/knowledge/chunks":
            return _FakeResponse({"items": [{"title": "退款到账时间说明"}]})
        if path == "/traces/trace-001":
            return _FakeResponse({
                "stages": [{
                    "name": "knowledge_context.build",
                    "meta": {
                        "retrieved_contexts": [
                            "1. 标题: 退款到账时间说明\n   命中片段: 审核通过后，银行卡退款通常在 1-3 个工作日内到账。"
                        ],
                    },
                }],
            })
        raise AssertionError(f"unexpected request: {path}")


class RagasRunnerTests(unittest.TestCase):
    def test_module_entrypoint_is_declared_after_private_helpers(self):
        source = pathlib.Path(ragas_runner.__file__).read_text(encoding="utf-8")

        self.assertGreater(
            source.index('if __name__ == "__main__":'),
            source.index("def _extract_contexts"),
        )

    def test_load_eval_cases_reads_jsonl_and_rejects_missing_evidence(self):
        valid_case = {
            "case_id": "refund-arrival-001",
            "user_input": "银行卡退款审核通过后多久到账？",
            "reference": "审核通过后，银行卡退款通常在 1-3 个工作日内到账。",
            "reference_contexts": ["审核通过后，余额支付一般在 1 个工作日内到账，银行卡、支付宝、微信等原支付方式通常在 1-3 个工作日内到账。"],
            "source_file": "退款到账时间说明.md",
            "category": "direct",
            "difficulty": "easy",
            "unanswerable": False,
        }
        invalid_case = dict(valid_case, case_id="refund-arrival-002", reference_contexts=[])

        with tempfile.TemporaryDirectory() as temp_dir:
            path = pathlib.Path(temp_dir) / "cases.jsonl"
            path.write_text(
                "\n".join(json.dumps(item, ensure_ascii=False) for item in [valid_case, invalid_case]),
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "reference_contexts"):
                load_eval_cases(path)

            path.write_text(json.dumps(valid_case, ensure_ascii=False) + "\n", encoding="utf-8")
            cases = load_eval_cases(path)

        self.assertEqual(cases, [valid_case])

    def test_collect_deployed_records_uses_deployed_search_and_chat_contracts(self):
        case = {
            "case_id": "refund-arrival-001",
            "user_input": "银行卡退款审核通过后多久到账？",
            "reference": "审核通过后，银行卡退款通常在 1-3 个工作日内到账。",
            "reference_contexts": ["审核通过后，余额支付一般在 1 个工作日内到账，银行卡、支付宝、微信等原支付方式通常在 1-3 个工作日内到账。"],
            "source_file": "退款到账时间说明.md",
            "category": "direct",
            "difficulty": "easy",
            "unanswerable": False,
        }
        client = _FakeHttpClient()

        records = collect_deployed_records([case], client=client, top_k=2, run_id="test-run")

        self.assertEqual(len(records), 1)
        self.assertEqual(records[0]["user_input"], case["user_input"])
        self.assertEqual(records[0]["retrieved_contexts"], [
            "1. 标题: 退款到账时间说明\n   命中片段: 审核通过后，银行卡退款通常在 1-3 个工作日内到账。",
        ])
        self.assertEqual(records[0]["diagnostic_retrieved_contexts"], [
            "审核通过后，银行卡退款通常在 1-3 个工作日内到账。",
            "退款成功后可查看支付账户明细。",
        ])
        self.assertEqual(records[0]["response"], "审核通过后，银行卡退款通常在 1-3 个工作日内到账。")
        self.assertEqual(records[0]["reference"], case["reference"])
        self.assertEqual(records[0]["reference_contexts"], case["reference_contexts"])
        self.assertEqual(records[0]["metadata"]["case_id"], case["case_id"])
        self.assertEqual(client.calls[0], {
            "path": "/search",
            "params": {"query": case["user_input"], "top_k": 2},
            "json": None,
        })
        self.assertEqual(client.calls[1], {
            "path": "/chat",
            "params": None,
            "json": {
                "message": case["user_input"],
                "user_id": "ragas-eval-refund-arrival-001",
                "conv_id": "ragas-eval-test-run-refund-arrival-001",
            },
        })
        self.assertEqual(client.calls[2], {
            "path": "/traces/trace-001",
            "params": None,
            "json": None,
        })

    def test_collect_deployed_records_preserves_workflow_turns_in_one_conversation(self):
        case = {
            "case_id": "refund-workflow-001",
            "user_input": "我想退款",
            "reference": "需要先提供订单号。",
            "reference_contexts": [],
            "source_file": "",
            "category": "workflow",
            "difficulty": "medium",
            "unanswerable": False,
            "evaluation_mode": "workflow",
            "turns": [
                {
                    "user_input": "我想退款",
                    "reference": "需要先提供订单号。",
                    "expected_behavior": "clarify_order_id",
                },
                {
                    "user_input": "订单号是 ORD20250701001",
                    "reference": "系统应承接退款意图并处理该订单。",
                    "expected_behavior": "continue_workflow",
                },
            ],
        }

        client = _FakeHttpClient()
        records = collect_deployed_records([case], client=client, top_k=2, run_id="run-20260710")

        self.assertEqual([record["user_input"] for record in records], [
            "我想退款",
            "订单号是 ORD20250701001",
        ])
        self.assertEqual([record["metadata"]["turn_index"] for record in records], [0, 1])
        self.assertEqual([record["metadata"]["expected_behavior"] for record in records], [
            "clarify_order_id",
            "continue_workflow",
        ])
        self.assertEqual([record["metadata"]["evaluation_mode"] for record in records], [
            "workflow",
            "workflow",
        ])
        self.assertTrue(all(record["chat"]["knowledge_used"] for record in records))
        chat_calls = [call for call in client.calls if call["path"] == "/chat"]
        self.assertEqual([call["json"]["conv_id"] for call in chat_calls], [
            "ragas-eval-run-20260710-refund-workflow-001",
            "ragas-eval-run-20260710-refund-workflow-001",
        ])

    def test_collect_deployed_records_rejects_search_without_required_reranking(self):
        case = {
            "case_id": "refund-arrival-001",
            "user_input": "银行卡退款审核通过后多久到账？",
            "reference": "审核通过后，银行卡退款通常在 1-3 个工作日内到账。",
            "reference_contexts": ["审核通过后，银行卡退款通常在 1-3 个工作日内到账。"],
            "source_file": "退款到账时间说明.md",
            "category": "direct",
            "difficulty": "easy",
            "unanswerable": False,
        }

        class NoRerankClient(_FakeHttpClient):
            def post(self, path, params=None, json=None):
                if path == "/search":
                    self.calls.append({"path": path, "params": params, "json": json})
                    return _FakeResponse({
                        "results": [{"content": "退款将在审核后原路退回。"}],
                        "rerank_applied": False,
                        "rerank_providers": ["rrf_fallback"],
                    })
                return super().post(path, params=params, json=json)

        with self.assertRaisesRegex(ValueError, "real reranker"):
            collect_deployed_records(
                [case],
                client=NoRerankClient(),
                top_k=2,
                require_rerank=True,
                required_rerank_provider="tei",
                run_id="test-run",
            )

    def test_build_ragas_samples_excludes_unanswerable_cases(self):
        records = [
            {
                "user_input": "退款多久到账？",
                "retrieved_contexts": ["退款审核通过后会原路退回。"],
                "response": "通常会原路退回。",
                "reference": "审核通过后会原路退回。",
                "metadata": {"unanswerable": False},
            },
            {
                "user_input": "海外订单怎么寄？",
                "retrieved_contexts": [],
                "response": "请联系人工客服。",
                "reference": "当前知识库没有足够依据回答该问题。",
                "metadata": {"unanswerable": True},
            },
        ]

        samples = build_ragas_samples(records)

        self.assertEqual(samples, [{
            "user_input": "退款多久到账？",
            "retrieved_contexts": ["退款审核通过后会原路退回。"],
            "response": "通常会原路退回。",
            "reference": "审核通过后会原路退回。",
        }])

    def test_anthropic_adapter_uses_base_url_and_model_from_configuration(self):
        created_clients = []

        class FakeAsyncAnthropic:
            def __init__(self, **kwargs):
                created_clients.append(kwargs)

        class FakeInstructorClient:
            messages = object()

        class FakeInstructorLLM:
            def __init__(self, *, client, model, provider, **kwargs):
                self.client = client
                self.model = model
                self.provider = provider
                self.model_args = {
                    "temperature": 0.01,
                    "top_p": 0.1,
                    "max_tokens": 1024,
                    **kwargs,
                }

        def fake_from_anthropic(client):
            self.assertIsInstance(client, FakeAsyncAnthropic)
            return FakeInstructorClient()

        anthropic_module = types.ModuleType("anthropic")
        anthropic_module.AsyncAnthropic = FakeAsyncAnthropic
        instructor_module = types.ModuleType("instructor")
        instructor_module.from_anthropic = fake_from_anthropic
        ragas_module = types.ModuleType("ragas")
        ragas_llms_module = types.ModuleType("ragas.llms")
        ragas_llms_base_module = types.ModuleType("ragas.llms.base")
        ragas_llms_base_module.InstructorLLM = FakeInstructorLLM

        with mock.patch.dict(sys.modules, {
            "anthropic": anthropic_module,
            "instructor": instructor_module,
            "ragas": ragas_module,
            "ragas.llms": ragas_llms_module,
            "ragas.llms.base": ragas_llms_base_module,
        }):
            evaluator = build_anthropic_evaluator({
                "ANTHROPIC_API_KEY": "test-key",
                "ANTHROPIC_BASE_URL": "https://api.deepseek.example/anthropic",
                "ANTHROPIC_MODEL": "deepseek-chat",
            })

        self.assertEqual(created_clients, [{
            "api_key": "test-key",
            "base_url": "https://api.deepseek.example/anthropic",
        }])
        self.assertIsInstance(evaluator, AnthropicMessagesRagasLLM)
        self.assertIsInstance(evaluator, FakeInstructorLLM)
        self.assertEqual(evaluator.model, "deepseek-chat")

    def test_anthropic_adapter_uses_instructor_messages_api_for_async_scoring(self):
        created_clients = []
        create_calls = []

        class FakeAsyncAnthropic:
            def __init__(self, **kwargs):
                created_clients.append(kwargs)

        class FakeMessages:
            async def create(self, **kwargs):
                create_calls.append(kwargs)
                return "structured-result"

        class FakeInstructorClient:
            messages = FakeMessages()

        class FakeInstructorLLM:
            def __init__(self, *, client, model, provider, **kwargs):
                self.client = client
                self.model = model
                self.provider = provider
                self.model_args = {
                    "temperature": 0.01,
                    "top_p": 0.1,
                    "max_tokens": 1024,
                    **kwargs,
                }

        def fake_from_anthropic(client):
            self.assertIsInstance(client, FakeAsyncAnthropic)
            return FakeInstructorClient()

        anthropic_module = types.ModuleType("anthropic")
        anthropic_module.AsyncAnthropic = FakeAsyncAnthropic
        instructor_module = types.ModuleType("instructor")
        instructor_module.from_anthropic = fake_from_anthropic
        ragas_module = types.ModuleType("ragas")
        ragas_llms_module = types.ModuleType("ragas.llms")
        ragas_llms_base_module = types.ModuleType("ragas.llms.base")
        ragas_llms_base_module.InstructorLLM = FakeInstructorLLM

        with mock.patch.dict(sys.modules, {
            "anthropic": anthropic_module,
            "instructor": instructor_module,
            "ragas": ragas_module,
            "ragas.llms": ragas_llms_module,
            "ragas.llms.base": ragas_llms_base_module,
        }):
            evaluator = build_anthropic_evaluator({
                "ANTHROPIC_API_KEY": "test-key",
                "ANTHROPIC_BASE_URL": "https://api.deepseek.example/anthropic",
                "ANTHROPIC_MODEL": "deepseek-chat",
            })
            result = asyncio.run(evaluator.agenerate("judge this", str))

        self.assertEqual(result, "structured-result")
        self.assertIsInstance(evaluator, FakeInstructorLLM)
        self.assertEqual(created_clients, [{
            "api_key": "test-key",
            "base_url": "https://api.deepseek.example/anthropic",
        }])
        self.assertEqual(create_calls, [{
            "messages": [{"role": "user", "content": "judge this"}],
            "response_model": str,
            "model": "deepseek-chat",
            "temperature": 0.01,
            "top_p": 0.1,
            "max_tokens": 4096,
        }])

    def test_embedding_adapter_uses_openai_compatible_environment(self):
        created_clients = []

        class FakeAsyncOpenAI:
            def __init__(self, **kwargs):
                created_clients.append(kwargs)

        def fake_embedding_factory(provider, **kwargs):
            return {"provider": provider, **kwargs}

        openai_module = types.ModuleType("openai")
        openai_module.AsyncOpenAI = FakeAsyncOpenAI
        ragas_module = types.ModuleType("ragas")
        embeddings_module = types.ModuleType("ragas.embeddings")
        embeddings_base_module = types.ModuleType("ragas.embeddings.base")
        embeddings_base_module.embedding_factory = fake_embedding_factory

        with mock.patch.dict(sys.modules, {
            "openai": openai_module,
            "ragas": ragas_module,
            "ragas.embeddings": embeddings_module,
            "ragas.embeddings.base": embeddings_base_module,
        }):
            embeddings = build_openai_embeddings({
                "EVAL_EMBEDDING_API_KEY": "embedding-key",
                "EVAL_EMBEDDING_BASE_URL": "https://embedding.example/compatible-mode/v1",
                "EVAL_EMBEDDING_MODEL": "text-embedding-v4",
            })

        self.assertEqual(created_clients, [{
            "api_key": "embedding-key",
            "base_url": "https://embedding.example/compatible-mode/v1",
        }])
        self.assertEqual(embeddings["provider"], "openai")
        self.assertEqual(embeddings["model"], "text-embedding-v4")
        self.assertEqual(embeddings["interface"], "modern")

    def test_build_collection_metrics_uses_the_configured_llm_and_embeddings(self):
        class FakeMetric:
            def __init__(self, llm, embeddings=None):
                self.llm = llm
                self.embeddings = embeddings

        metrics_module = types.ModuleType("ragas.metrics.collections")
        metrics_module.Faithfulness = FakeMetric
        metrics_module.FactualCorrectness = FakeMetric
        metrics_module.AnswerRelevancy = FakeMetric

        with mock.patch.dict(sys.modules, {"ragas.metrics.collections": metrics_module}):
            metrics = build_collection_metrics(
                evaluator_llm="judge",
                evaluator_embeddings="embeddings",
            )

        self.assertEqual(set(metrics), {
            "faithfulness",
            "factual_correctness",
            "answer_relevancy",
        })
        self.assertTrue(all(metric.llm == "judge" for metric in metrics.values()))
        self.assertIsNone(metrics["faithfulness"].embeddings)
        self.assertEqual(metrics["answer_relevancy"].embeddings, "embeddings")

    def test_run_ragas_evaluation_only_scores_faithfulness_when_chat_used_knowledge(self):
        class FakeResult:
            def __init__(self, value):
                self.value = value

        calls = []

        class FakeMetric:
            def __init__(self, name):
                self.name = name

            async def ascore(self, **kwargs):
                calls.append((self.name, kwargs))
                return FakeResult(0.9)

        metrics = {
            "faithfulness": FakeMetric("faithfulness"),
            "factual_correctness": FakeMetric("factual_correctness"),
            "answer_relevancy": FakeMetric("answer_relevancy"),
        }

        records = [{
            "user_input": "退款多久到账？",
            "retrieved_contexts": ["退款审核通过后会原路退回。"],
            "response": "通常会原路退回。",
            "reference": "审核通过后会原路退回。",
            "reference_contexts": ["评测金标上下文，不应传给 Faithfulness。"],
            "chat": {"knowledge_used": True, "retrieved_context_source": "chat_trace"},
            "metadata": {"unanswerable": False, "evaluation_mode": "knowledge"},
        }, {
            "user_input": "我想退款",
            "retrieved_contexts": ["退款规则"],
            "response": "请提供订单号。",
            "reference": "需要先提供订单号。",
            "reference_contexts": [],
            "chat": {"knowledge_used": False},
            "metadata": {"unanswerable": False, "evaluation_mode": "workflow"},
        }]
        with mock.patch(
            "evaluation.ragas_runner.build_collection_metrics",
            return_value=metrics,
        ) as build_metrics:
            rows = run_ragas_evaluation(records, evaluator_llm="judge", evaluator_embeddings="embeddings")

        self.assertEqual(
            build_metrics.call_args.kwargs,
            {"evaluator_llm": "judge", "evaluator_embeddings": "embeddings"},
        )
        self.assertEqual(rows, [{
            "case_id": "",
            "turn_index": 0,
            "faithfulness": 0.9,
            "factual_correctness": 0.9,
            "answer_relevancy": 0.9,
        }])
        faithfulness_calls = [kwargs for name, kwargs in calls if name == "faithfulness"]
        self.assertEqual(faithfulness_calls, [{
            "user_input": "退款多久到账？",
            "response": "通常会原路退回。",
            "retrieved_contexts": ["退款审核通过后会原路退回。"],
        }])

    def test_run_ragas_evaluation_skips_faithfulness_for_legacy_diagnostic_contexts(self):
        class FakeResult:
            def __init__(self, value):
                self.value = value

        calls = []

        class FakeMetric:
            def __init__(self, name):
                self.name = name

            async def ascore(self, **kwargs):
                calls.append((self.name, kwargs))
                return FakeResult(0.9)

        metrics = {
            "faithfulness": FakeMetric("faithfulness"),
            "factual_correctness": FakeMetric("factual_correctness"),
            "answer_relevancy": FakeMetric("answer_relevancy"),
        }
        records = [{
            "user_input": "退款多久到账？",
            "retrieved_contexts": ["这是旧报告的诊断检索结果。"],
            "response": "通常会原路退回。",
            "reference": "审核通过后会原路退回。",
            "chat": {"knowledge_used": True},
            "metadata": {"unanswerable": False, "evaluation_mode": "knowledge"},
        }]

        with mock.patch("evaluation.ragas_runner.build_collection_metrics", return_value=metrics):
            rows = run_ragas_evaluation(records, evaluator_llm="judge", evaluator_embeddings="embeddings")

        self.assertIsNone(rows[0]["faithfulness"])
        self.assertNotIn("faithfulness", [name for name, _ in calls])

    def test_write_report_replaces_non_finite_values_with_null(self):
        report = {
            "ragas_rows": [{
                "faithfulness": math.nan,
                "answer_relevancy": math.inf,
                "factual_correctness": -math.inf,
            }],
        }

        with tempfile.TemporaryDirectory() as temp_dir:
            path = pathlib.Path(temp_dir) / "report.json"
            ragas_runner._write_report(path, report)
            raw = path.read_text(encoding="utf-8")
            saved = json.loads(raw)

        self.assertNotIn("NaN", raw)
        self.assertNotIn("Infinity", raw)
        self.assertEqual(saved["ragas_rows"][0], {
            "faithfulness": None,
            "answer_relevancy": None,
            "factual_correctness": None,
        })

    def test_run_ragas_evaluation_bounds_metric_concurrency_and_keeps_row_order(self):
        class FakeResult:
            def __init__(self, value):
                self.value = value

        active = 0
        peak = 0

        class FakeMetric:
            async def ascore(self, **kwargs):
                nonlocal active, peak
                active += 1
                peak = max(peak, active)
                try:
                    await asyncio.sleep(0.01)
                    return FakeResult(0.9)
                finally:
                    active -= 1

        metrics = {
            "faithfulness": FakeMetric(),
            "factual_correctness": FakeMetric(),
            "answer_relevancy": FakeMetric(),
        }
        records = [
            {
                "user_input": f"问题 {index}",
                "retrieved_contexts": [],
                "response": f"回答 {index}",
                "reference": f"参考 {index}",
                "reference_contexts": [],
                "chat": {"knowledge_used": False},
                "metadata": {
                    "case_id": f"case-{index}",
                    "turn_index": 0,
                    "unanswerable": False,
                    "evaluation_mode": "knowledge",
                },
            }
            for index in range(4)
        ]

        with mock.patch("evaluation.ragas_runner.build_collection_metrics", return_value=metrics):
            rows = run_ragas_evaluation(
                records,
                evaluator_llm="judge",
                evaluator_embeddings="embeddings",
                max_concurrency=2,
            )

        self.assertLessEqual(peak, 2)
        self.assertEqual([row["case_id"] for row in rows], ["case-0", "case-1", "case-2", "case-3"])

    def test_evaluate_workflow_records_checks_required_clarification_without_ragas(self):
        rows = evaluate_workflow_records([
            {
                "user_input": "我想退款",
                "response": "为了帮你准确查询订单或退款进度，请提供订单号。",
                "metadata": {
                    "case_id": "refund-workflow-001",
                    "turn_index": 0,
                    "evaluation_mode": "workflow",
                    "expected_behavior": "clarify_order_id",
                },
            },
            {
                "user_input": "订单号是 ORD20250701001",
                "response": "已根据订单号 ORD20250701001 查询到订单正在运输中。",
                "metadata": {
                    "case_id": "refund-workflow-001",
                    "turn_index": 1,
                    "evaluation_mode": "workflow",
                    "expected_behavior": "continue_workflow",
                },
            },
            {
                "user_input": "订单号是 ORD20250701001",
                "response": "为了帮你准确查询订单或退款进度，请提供订单号。",
                "metadata": {
                    "case_id": "refund-workflow-002",
                    "turn_index": 1,
                    "evaluation_mode": "workflow",
                    "expected_behavior": "continue_workflow",
                },
            },
            {
                "user_input": "退款多久到账？",
                "response": "通常会原路退回。",
                "metadata": {"evaluation_mode": "knowledge"},
            },
        ])

        self.assertEqual(rows, [{
            "case_id": "refund-workflow-001",
            "turn_index": 0,
            "expected_behavior": "clarify_order_id",
            "passed": True,
        }, {
            "case_id": "refund-workflow-001",
            "turn_index": 1,
            "expected_behavior": "continue_workflow",
            "passed": True,
        }, {
            "case_id": "refund-workflow-002",
            "turn_index": 1,
            "expected_behavior": "continue_workflow",
            "passed": False,
        }])

    def test_run_deployed_evaluation_persists_records_and_metric_rows(self):
        case = {
            "case_id": "refund-arrival-001",
            "user_input": "银行卡退款审核通过后多久到账？",
            "reference": "审核通过后，银行卡退款通常在 1-3 个工作日内到账。",
            "reference_contexts": ["审核通过后，余额支付一般在 1 个工作日内到账，银行卡、支付宝、微信等原支付方式通常在 1-3 个工作日内到账。"],
            "source_file": "退款到账时间说明.md",
            "category": "direct",
            "difficulty": "easy",
            "unanswerable": False,
        }
        client = _FakeHttpClient()

        with tempfile.TemporaryDirectory() as temp_dir:
            root = pathlib.Path(temp_dir)
            dataset_path = root / "cases.jsonl"
            dataset_path.write_text(json.dumps(case, ensure_ascii=False) + "\n", encoding="utf-8")
            with mock.patch(
                "evaluation.ragas_runner.run_ragas_evaluation",
                return_value=[{"context_precision": 0.8, "faithfulness": 0.9}],
            ):
                report = run_deployed_evaluation(
                    dataset_path=dataset_path,
                    output_dir=root / "output",
                    client=client,
                    evaluator_llm="judge",
                    evaluator_embeddings="embeddings",
                    run_id="test-run",
                    top_k=2,
                )

            saved = json.loads((root / "output" / "test-run.json").read_text(encoding="utf-8"))

        self.assertEqual(report["summary"], {
            "total_cases": 1,
            "answerable_cases": 1,
            "unanswerable_cases": 0,
            "workflow_cases": 0,
        })
        self.assertEqual(saved["ragas_rows"], [{"context_precision": 0.8, "faithfulness": 0.9}])
        self.assertEqual(saved["records"][0]["metadata"]["case_id"], "refund-arrival-001")
        self.assertEqual(saved["status"], "completed")

    def test_run_deployed_evaluation_saves_collected_records_when_scoring_fails(self):
        case = {
            "case_id": "refund-arrival-001",
            "user_input": "银行卡退款审核通过后多久到账？",
            "reference": "审核通过后，银行卡退款通常在 1-3 个工作日内到账。",
            "reference_contexts": ["审核通过后，余额支付一般在 1 个工作日内到账，银行卡、支付宝、微信等原支付方式通常在 1-3 个工作日内到账。"],
            "source_file": "退款到账时间说明.md",
            "category": "direct",
            "difficulty": "easy",
            "unanswerable": False,
        }
        client = _FakeHttpClient()

        with tempfile.TemporaryDirectory() as temp_dir:
            root = pathlib.Path(temp_dir)
            dataset_path = root / "cases.jsonl"
            dataset_path.write_text(json.dumps(case, ensure_ascii=False) + "\n", encoding="utf-8")
            with mock.patch(
                "evaluation.ragas_runner.run_ragas_evaluation",
                side_effect=TypeError("metric object required"),
            ), self.assertRaisesRegex(TypeError, "metric object required"):
                run_deployed_evaluation(
                    dataset_path=dataset_path,
                    output_dir=root / "output",
                    client=client,
                    evaluator_llm="judge",
                    evaluator_embeddings="embeddings",
                    run_id="failed-run",
                    top_k=2,
                )

            saved = json.loads((root / "output" / "failed-run.json").read_text(encoding="utf-8"))

        self.assertEqual(saved["status"], "scoring_failed")
        self.assertEqual(saved["error"], "metric object required")
        self.assertEqual(saved["records"][0]["metadata"]["case_id"], "refund-arrival-001")

    def test_resume_evaluation_report_reuses_saved_records(self):
        report = {
            "run_id": "failed-run",
            "status": "completed",
            "error": None,
            "summary": {"total_cases": 1, "answerable_cases": 1, "unanswerable_cases": 0},
            "records": [{
                "user_input": "退款多久到账？",
                "retrieved_contexts": ["退款审核通过后会原路退回。"],
                "response": "通常会原路退回。",
                "reference": "审核通过后会原路退回。",
                "metadata": {"case_id": "refund-001", "unanswerable": False},
            }],
            "ragas_rows": [{"context_precision": float("nan")}],
        }
        with tempfile.TemporaryDirectory() as temp_dir:
            path = pathlib.Path(temp_dir) / "failed-run.json"
            path.write_text(json.dumps(report), encoding="utf-8")
            with mock.patch(
                "evaluation.ragas_runner.run_ragas_evaluation",
                return_value=[{"case_id": "refund-001", "context_precision": 0.9}],
            ):
                resumed = resume_evaluation_report(
                    report_path=path,
                    evaluator_llm="judge",
                    evaluator_embeddings="embeddings",
                )

            saved = json.loads(path.read_text(encoding="utf-8"))

        self.assertEqual(resumed["status"], "completed")
        self.assertEqual(saved["ragas_rows"], [{"case_id": "refund-001", "context_precision": 0.9}])

    def test_validate_deployed_knowledge_base_requires_all_answerable_sources(self):
        cases = [
            {
                "case_id": "refund-arrival-001",
                "source_file": "退款到账时间说明.md",
                "unanswerable": False,
            },
            {
                "case_id": "no-answer-001",
                "source_file": "",
                "unanswerable": True,
            },
        ]
        client = _FakeHttpClient()

        validate_deployed_knowledge_base(cases, client=client)

        missing_client = _FakeHttpClient()
        missing_client.get = lambda path, params=None: _FakeResponse({"items": []})
        with self.assertRaisesRegex(ValueError, "退款到账时间说明"):
            validate_deployed_knowledge_base(cases, client=missing_client)

    def test_main_uses_container_environment_and_runs_one_evaluation(self):
        created_clients = []

        class FakeClient:
            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc_value, traceback):
                return False

        def fake_client_factory(**kwargs):
            created_clients.append(kwargs)
            return FakeClient()

        with mock.patch("evaluation.ragas_runner.httpx.Client", side_effect=fake_client_factory), \
             mock.patch("evaluation.ragas_runner.build_anthropic_evaluator", return_value="judge"), \
             mock.patch("evaluation.ragas_runner.build_openai_embeddings", return_value="embeddings"), \
             mock.patch("evaluation.ragas_runner.run_deployed_evaluation", return_value={
                 "summary": {"total_cases": 80, "answerable_cases": 75, "unanswerable_cases": 5}
             }) as run_evaluation:
            exit_code = main([
                "--base-url", "http://echomind:8000",
                "--dataset", "/app/evaluation/datasets/customer_service_v1.jsonl",
                "--output-dir", "/app/data/eval",
                "--run-id", "container-run",
                "--top-k", "4",
            ])

        self.assertEqual(exit_code, 0)
        self.assertEqual(created_clients, [{"base_url": "http://echomind:8000", "timeout": 60.0}])
        self.assertEqual(run_evaluation.call_args.kwargs["evaluator_llm"], "judge")
        self.assertEqual(run_evaluation.call_args.kwargs["evaluator_embeddings"], "embeddings")
        self.assertEqual(run_evaluation.call_args.kwargs["run_id"], "container-run")
        self.assertEqual(run_evaluation.call_args.kwargs["top_k"], 4)


if __name__ == "__main__":
    unittest.main()
