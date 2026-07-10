"""Collect deployed RAG interactions into the schema consumed by Ragas."""
import argparse
import asyncio
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Protocol

import httpx


class AnthropicMessagesRagasLLM:
    """Mixin that routes Ragas Instructor LLM calls to Anthropic messages."""

    def _check_client_async(self) -> bool:
        """Instructor's Anthropic client exposes ``messages.create``, not chat completions."""
        return True

    async def agenerate(self, prompt: str, response_model: Any) -> Any:
        """Return a structured response through Anthropic's messages endpoint."""
        return await self.client.messages.create(
            messages=[{"role": "user", "content": str(prompt)}],
            response_model=response_model,
            model=self.model,
            **self.model_args,
        )


class HttpClient(Protocol):
    def post(
        self,
        path: str,
        params: Dict[str, Any] | None = None,
        json: Dict[str, Any] | None = None,
    ) -> Any:
        """Submit an HTTP request and return an httpx-compatible response."""

    def get(self, path: str, params: Dict[str, Any] | None = None) -> Any:
        """Submit an HTTP GET request and return an httpx-compatible response."""


_REQUIRED_CASE_FIELDS = {
    "case_id",
    "user_input",
    "reference",
    "reference_contexts",
    "source_file",
    "category",
    "difficulty",
    "unanswerable",
}


def load_eval_cases(path: Path) -> List[Dict[str, Any]]:
    """Load and validate a JSONL customer-service evaluation dataset."""
    cases: List[Dict[str, Any]] = []
    for line_number, raw_line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not raw_line.strip():
            continue
        try:
            case = json.loads(raw_line)
        except json.JSONDecodeError as ex:
            raise ValueError(f"invalid JSON on line {line_number}: {ex.msg}") from ex
        _validate_case(case, line_number)
        cases.append(case)

    if not cases:
        raise ValueError("evaluation dataset is empty")
    return cases


def collect_deployed_records(
    cases: List[Dict[str, Any]],
    *,
    client: HttpClient,
    top_k: int,
    recall_k: int | None = None,
    require_rerank: bool = False,
    required_rerank_provider: str = "",
) -> List[Dict[str, Any]]:
    """Call the deployed search and chat endpoints for every evaluation case."""
    if top_k < 1:
        raise ValueError("top_k must be positive")
    if recall_k is not None and recall_k <= top_k:
        raise ValueError("recall_k must be greater than top_k when provided")

    records: List[Dict[str, Any]] = []
    for case in cases:
        case_id = case["case_id"]
        user_id = str(case.get("user_id") or f"ragas-eval-{case_id}")
        conv_id = str(case.get("conv_id") or f"ragas-eval-{case_id}")
        evaluation_mode = str(case.get("evaluation_mode") or "knowledge")
        for turn_index, turn in enumerate(_case_turns(case)):
            question = turn["user_input"]
            search_params: Dict[str, Any] = {"query": question, "top_k": top_k}
            if recall_k is not None:
                search_params["recall_k"] = recall_k
            search_response = client.post("/search", params=search_params)
            search_response.raise_for_status()
            search_payload = search_response.json()
            retrieved_contexts = _extract_contexts(search_payload.get("results", []))
            rerank_applied = bool(search_payload.get("rerank_applied", False))
            rerank_providers = _extract_rerank_providers(search_payload.get("rerank_providers"))
            is_knowledge_case = not case["unanswerable"] and evaluation_mode == "knowledge"
            if require_rerank and is_knowledge_case:
                if not rerank_applied:
                    raise ValueError(
                        f"search for {case_id} turn {turn_index} did not use a real reranker"
                    )
                if required_rerank_provider and required_rerank_provider not in rerank_providers:
                    raise ValueError(
                        f"search for {case_id} turn {turn_index} did not use required reranker "
                        f"{required_rerank_provider!r}; got {rerank_providers!r}"
                    )

            chat_response = client.post(
                "/chat",
                json={
                    "message": question,
                    "user_id": user_id,
                    "conv_id": conv_id,
                },
            )
            chat_response.raise_for_status()
            chat_payload = chat_response.json()
            response = str(chat_payload.get("response", "")).strip()
            if not response:
                raise ValueError(f"chat response is empty for case {case_id} turn {turn_index}")

            records.append({
                "user_input": question,
                "retrieved_contexts": retrieved_contexts,
                "response": response,
                "reference": turn["reference"],
                "reference_contexts": turn["reference_contexts"],
                "search": {
                    "top_k": top_k,
                    "recall_k": recall_k,
                    "rerank_applied": rerank_applied,
                    "rerank_providers": rerank_providers,
                },
                "chat": {
                    "knowledge_used": bool(chat_payload.get("knowledge_used", False)),
                    "latency_ms": chat_payload.get("latency_ms"),
                    "trace_id": str(chat_payload.get("trace_id", "")),
                },
                "metadata": {
                    "case_id": case_id,
                    "source_file": case["source_file"],
                    "category": case["category"],
                    "difficulty": case["difficulty"],
                    "unanswerable": case["unanswerable"],
                    "evaluation_mode": evaluation_mode,
                    "turn_index": turn_index,
                    "expected_behavior": turn["expected_behavior"],
                },
            })
    return records


def build_ragas_samples(records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Select answerable records and remove EchoMind-specific report metadata."""
    samples = []
    for record in records:
        metadata = record.get("metadata", {})
        if metadata.get("unanswerable") or metadata.get("evaluation_mode", "knowledge") != "knowledge":
            continue
        samples.append({
            "user_input": record["user_input"],
            "retrieved_contexts": record["retrieved_contexts"],
            "response": record["response"],
            "reference": record["reference"],
        })
    return samples


def validate_deployed_knowledge_base(cases: List[Dict[str, Any]], *, client: HttpClient) -> None:
    """Ensure every answerable evaluation source is present in the deployed ChromaDB."""
    response = client.get("/knowledge/chunks", params={"limit": 5000})
    response.raise_for_status()
    payload = response.json()
    items = payload.get("items", []) if isinstance(payload, dict) else []
    deployed_titles = {
        str(item.get("title", "")).strip()
        for item in items
        if isinstance(item, dict) and str(item.get("title", "")).strip()
    }
    expected_titles = {
        Path(str(case["source_file"])).stem
        for case in cases
        if (
            not case["unanswerable"]
            and case.get("evaluation_mode", "knowledge") == "knowledge"
            and case.get("source_file")
        )
    }
    missing = sorted(expected_titles - deployed_titles)
    if missing:
        raise ValueError(
            "deployed knowledge base is missing evaluation sources: " + ", ".join(missing)
        )


def build_anthropic_evaluator(environ: Dict[str, str] | None = None) -> Any:
    """Create a Ragas evaluator backed by the configured Anthropic-compatible API."""
    from anthropic import AsyncAnthropic
    import instructor
    from ragas.llms.base import InstructorLLM

    settings = environ if environ is not None else os.environ
    api_key = settings.get("ANTHROPIC_API_KEY", "").strip()
    model = settings.get("ANTHROPIC_MODEL", "").strip()
    base_url = settings.get("ANTHROPIC_BASE_URL", "").strip()
    if not api_key:
        raise ValueError("ANTHROPIC_API_KEY is required for Ragas evaluation")
    if not model:
        raise ValueError("ANTHROPIC_MODEL is required for Ragas evaluation")

    client_kwargs: Dict[str, str] = {"api_key": api_key}
    if base_url:
        client_kwargs["base_url"] = base_url
    client = AsyncAnthropic(**client_kwargs)

    class AnthropicMessagesInstructorLLM(AnthropicMessagesRagasLLM, InstructorLLM):
        """Ragas Instructor LLM with Anthropic-compatible message transport."""

    return AnthropicMessagesInstructorLLM(
        client=instructor.from_anthropic(client),
        model=model,
        provider="anthropic",
        max_tokens=4096,
    )


def build_openai_embeddings(environ: Dict[str, str] | None = None) -> Any:
    """Create Ragas embeddings backed by an OpenAI-compatible remote endpoint."""
    from openai import AsyncOpenAI
    from ragas.embeddings.base import embedding_factory

    settings = environ if environ is not None else os.environ
    api_key = settings.get("EVAL_EMBEDDING_API_KEY", "").strip()
    base_url = settings.get("EVAL_EMBEDDING_BASE_URL", "").strip()
    model = settings.get("EVAL_EMBEDDING_MODEL", "").strip()
    if not api_key:
        raise ValueError("EVAL_EMBEDDING_API_KEY is required for AnswerRelevancy")
    if not base_url:
        raise ValueError("EVAL_EMBEDDING_BASE_URL is required for AnswerRelevancy")
    if not model:
        raise ValueError("EVAL_EMBEDDING_MODEL is required for AnswerRelevancy")

    client = AsyncOpenAI(api_key=api_key, base_url=base_url)
    return embedding_factory("openai", model=model, client=client, interface="modern")


def build_collection_metrics(*, evaluator_llm: Any, evaluator_embeddings: Any) -> Dict[str, Any]:
    """Build single-turn scorers compatible with the Instructor LLM adapter."""
    from ragas.metrics.collections import (
        AnswerRelevancy,
        FactualCorrectness,
        Faithfulness,
    )

    return {
        "faithfulness": Faithfulness(llm=evaluator_llm),
        "factual_correctness": FactualCorrectness(llm=evaluator_llm),
        "answer_relevancy": AnswerRelevancy(
            llm=evaluator_llm,
            embeddings=evaluator_embeddings,
        ),
    }


def run_ragas_evaluation(
    records: List[Dict[str, Any]],
    *,
    evaluator_llm: Any,
    evaluator_embeddings: Any,
    progress_callback: Any = None,
    max_concurrency: int = 3,
) -> List[Dict[str, Any]]:
    """Score answerable records through Ragas collection metrics and checkpoint progress."""
    answerable_records = [
        record
        for record in records
        if not record.get("metadata", {}).get("unanswerable")
        and record.get("metadata", {}).get("evaluation_mode", "knowledge") == "knowledge"
    ]
    if not answerable_records:
        return []
    if max_concurrency < 1:
        raise ValueError("max_concurrency must be positive")

    async def score_all() -> List[Dict[str, Any]]:
        metrics = build_collection_metrics(
            evaluator_llm=evaluator_llm,
            evaluator_embeddings=evaluator_embeddings,
        )
        semaphore = asyncio.Semaphore(max_concurrency)
        rows: List[Dict[str, Any] | None] = [None] * len(answerable_records)

        async def score_metric(metric: Any, **kwargs: Any) -> Any:
            async with semaphore:
                return await metric.ascore(**kwargs)

        async def score_one(index: int, record: Dict[str, Any]) -> tuple[int, Dict[str, Any]]:
            return index, await _score_record(record, metrics, score_metric=score_metric)

        tasks = [
            asyncio.create_task(score_one(index, record))
            for index, record in enumerate(answerable_records)
        ]
        completed = 0
        try:
            for completed_task in asyncio.as_completed(tasks):
                index, row = await completed_task
                rows[index] = row
                completed += 1
                completed_rows = [item for item in rows if item is not None]
                if progress_callback is not None:
                    progress_callback(completed_rows)
                print(f"Ragas scoring: {completed}/{len(answerable_records)}")
        except Exception:
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            raise
        return [item for item in rows if item is not None]

    return asyncio.run(score_all())


def run_deployed_evaluation(
    *,
    dataset_path: Path,
    output_dir: Path,
    client: HttpClient,
    evaluator_llm: Any,
    evaluator_embeddings: Any,
    run_id: str,
    top_k: int,
    recall_k: int | None = None,
    require_rerank: bool = False,
    required_rerank_provider: str = "",
    max_concurrency: int = 3,
    workflow_dataset_path: Path | None = None,
) -> Dict[str, Any]:
    """Collect one deployed run, evaluate answerable records, and persist its report."""
    cases = load_eval_cases(dataset_path)
    if workflow_dataset_path is not None:
        cases.extend(load_eval_cases(workflow_dataset_path))
    validate_deployed_knowledge_base(cases, client=client)
    records = collect_deployed_records(
        cases,
        client=client,
        top_k=top_k,
        recall_k=recall_k,
        require_rerank=require_rerank,
        required_rerank_provider=required_rerank_provider,
    )
    unanswerable_cases = sum(1 for case in cases if case["unanswerable"])
    report = {
        "run_id": run_id,
        "status": "collected",
        "error": None,
        "summary": {
            "total_cases": len(cases),
            "answerable_cases": sum(
                1
                for case in cases
                if not case["unanswerable"] and case.get("evaluation_mode", "knowledge") == "knowledge"
            ),
            "unanswerable_cases": unanswerable_cases,
            "workflow_cases": sum(
                1 for case in cases if case.get("evaluation_mode") == "workflow"
            ),
        },
        "records": records,
        "ragas_rows": [],
        "evaluation_config": {
            "top_k": top_k,
            "recall_k": recall_k,
            "require_rerank": require_rerank,
            "required_rerank_provider": required_rerank_provider,
            "max_concurrency": max_concurrency,
        },
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / f"{run_id}.json"
    _write_report(report_path, report)
    return _score_report(
        report_path=report_path,
        report=report,
        evaluator_llm=evaluator_llm,
        evaluator_embeddings=evaluator_embeddings,
        max_concurrency=max_concurrency,
    )


def resume_evaluation_report(
    *,
    report_path: Path,
    evaluator_llm: Any,
    evaluator_embeddings: Any,
    max_concurrency: int = 3,
) -> Dict[str, Any]:
    """Re-score a saved collection report without calling the deployed chat service again."""
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if not isinstance(report, dict) or not isinstance(report.get("records"), list):
        raise ValueError("resume report must contain a records list")
    return _score_report(
        report_path=report_path,
        report=report,
        evaluator_llm=evaluator_llm,
        evaluator_embeddings=evaluator_embeddings,
        max_concurrency=max_concurrency,
    )


def _score_report(
    *,
    report_path: Path,
    report: Dict[str, Any],
    evaluator_llm: Any,
    evaluator_embeddings: Any,
    max_concurrency: int = 3,
) -> Dict[str, Any]:
    report["status"] = "scoring"
    report["error"] = None
    report["ragas_rows"] = []
    report["workflow_rows"] = evaluate_workflow_records(report["records"])
    report["scored_cases"] = 0
    _write_report(report_path, report)

    def checkpoint(rows: List[Dict[str, Any]]) -> None:
        report["ragas_rows"] = rows
        report["scored_cases"] = len(rows)
        _write_report(report_path, report)

    try:
        report["ragas_rows"] = run_ragas_evaluation(
            report["records"],
            evaluator_llm=evaluator_llm,
            evaluator_embeddings=evaluator_embeddings,
            progress_callback=checkpoint,
            max_concurrency=max_concurrency,
        )
    except Exception as ex:
        report["status"] = "scoring_failed"
        report["error"] = str(ex)
        _write_report(report_path, report)
        raise

    report["status"] = "completed"
    _write_report(report_path, report)
    return report


def main(argv: List[str] | None = None) -> int:
    """Run one containerized Ragas evaluation against the deployed EchoMind API."""
    parser = argparse.ArgumentParser(description="Evaluate the deployed EchoMind RAG pipeline with Ragas")
    parser.add_argument("--base-url", default=os.getenv("EVAL_APP_BASE_URL", "http://echomind:8000"))
    parser.add_argument("--dataset", default=os.getenv("EVAL_DATASET_PATH", "evaluation/datasets/customer_service_v1.jsonl"))
    parser.add_argument("--output-dir", default=os.getenv("EVAL_OUTPUT_DIR", "data/eval"))
    parser.add_argument("--run-id", default=os.getenv("EVAL_RUN_ID", ""))
    parser.add_argument("--top-k", type=int, default=int(os.getenv("EVAL_TOP_K", "3")))
    parser.add_argument("--recall-k", type=int, default=int(os.getenv("EVAL_RECALL_K", "12")))
    parser.add_argument("--max-concurrency", type=int, default=int(os.getenv("EVAL_MAX_CONCURRENCY", "3")))
    parser.add_argument("--require-rerank", default=os.getenv("EVAL_REQUIRE_RERANK", "false"))
    parser.add_argument("--required-rerank-provider", default=os.getenv("EVAL_REQUIRED_RERANK_PROVIDER", ""))
    parser.add_argument("--workflow-dataset", default=os.getenv("EVAL_WORKFLOW_DATASET", ""))
    parser.add_argument("--resume", default="", help="saved report path to score without recollecting")
    args = parser.parse_args(argv)

    run_id = args.run_id or f"ragas-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    require_rerank = _parse_bool(args.require_rerank, name="require-rerank")
    evaluator_llm = build_anthropic_evaluator()
    evaluator_embeddings = build_openai_embeddings()
    if args.resume:
        report = resume_evaluation_report(
            report_path=Path(args.resume),
            evaluator_llm=evaluator_llm,
            evaluator_embeddings=evaluator_embeddings,
            max_concurrency=args.max_concurrency,
        )
    else:
        with httpx.Client(base_url=args.base_url, timeout=60.0) as client:
            report = run_deployed_evaluation(
                dataset_path=Path(args.dataset),
                output_dir=Path(args.output_dir),
                client=client,
                evaluator_llm=evaluator_llm,
                evaluator_embeddings=evaluator_embeddings,
                run_id=run_id,
                top_k=args.top_k,
                recall_k=args.recall_k,
                require_rerank=require_rerank,
                required_rerank_provider=args.required_rerank_provider,
                max_concurrency=args.max_concurrency,
                workflow_dataset_path=Path(args.workflow_dataset) if args.workflow_dataset else None,
            )
    summary = report["summary"]
    print(
        f"Ragas evaluation complete: {summary['total_cases']} cases "
        f"({summary['answerable_cases']} answerable, {summary['unanswerable_cases']} unanswerable)"
    )
    return 0


def _validate_case(case: Any, line_number: int) -> None:
    if not isinstance(case, dict):
        raise ValueError(f"case on line {line_number} must be an object")
    missing = _REQUIRED_CASE_FIELDS - set(case)
    if missing:
        raise ValueError(f"case on line {line_number} is missing fields: {', '.join(sorted(missing))}")
    if not isinstance(case["reference_contexts"], list):
        raise ValueError(f"reference_contexts on line {line_number} must be a list")
    evaluation_mode = str(case.get("evaluation_mode") or "knowledge")
    if evaluation_mode not in {"knowledge", "workflow"}:
        raise ValueError(f"evaluation_mode on line {line_number} must be knowledge or workflow")
    if (
        evaluation_mode == "knowledge"
        and not case["unanswerable"]
        and not case["reference_contexts"]
    ):
        raise ValueError(f"reference_contexts on line {line_number} is required for answerable cases")
    for field in ("case_id", "user_input", "reference", "category", "difficulty"):
        if not isinstance(case[field], str) or not case[field].strip():
            raise ValueError(f"{field} on line {line_number} must be a non-empty string")
    if evaluation_mode == "workflow":
        turns = case.get("turns")
        if not isinstance(turns, list) or not turns:
            raise ValueError(f"turns on line {line_number} is required for workflow cases")
        for turn_index, turn in enumerate(turns, start=1):
            if not isinstance(turn, dict):
                raise ValueError(f"turn {turn_index} on line {line_number} must be an object")
            for field in ("user_input", "reference", "expected_behavior"):
                if not isinstance(turn.get(field), str) or not str(turn[field]).strip():
                    raise ValueError(
                        f"{field} for turn {turn_index} on line {line_number} must be a non-empty string"
                    )


def _case_turns(case: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Normalize single-turn knowledge cases and multi-turn workflow cases."""
    if case.get("evaluation_mode", "knowledge") != "workflow":
        return [{
            "user_input": case["user_input"],
            "reference": case["reference"],
            "reference_contexts": case["reference_contexts"],
            "expected_behavior": "knowledge_answer",
        }]
    return [{
        "user_input": turn["user_input"],
        "reference": turn["reference"],
        "reference_contexts": turn.get("reference_contexts", case["reference_contexts"]),
        "expected_behavior": turn["expected_behavior"],
    } for turn in case["turns"]]


def _extract_contexts(results: Any) -> List[str]:
    if not isinstance(results, list):
        return []
    contexts: List[str] = []
    for item in results:
        if not isinstance(item, dict):
            continue
        content = str(item.get("matched_child_content") or item.get("content") or "").strip()
        if content:
            contexts.append(content)
    return contexts


def _extract_rerank_providers(value: Any) -> List[str]:
    if not isinstance(value, list):
        return []
    return sorted({str(item).strip() for item in value if str(item).strip()})


async def _score_record(
    record: Dict[str, Any],
    metrics: Dict[str, Any],
    *,
    score_metric: Any,
) -> Dict[str, Any]:
    user_input = record["user_input"]
    response = record["response"]
    reference = record["reference"]
    metric_tasks = {
        "factual_correctness": score_metric(
            metrics["factual_correctness"],
            response=response,
            reference=reference,
        ),
        "answer_relevancy": score_metric(
            metrics["answer_relevancy"],
            user_input=user_input,
            response=response,
        ),
    }
    reference_contexts = record.get("reference_contexts", [])
    if record.get("chat", {}).get("knowledge_used") and reference_contexts:
        metric_tasks["faithfulness"] = score_metric(
            metrics["faithfulness"],
            user_input=user_input,
            response=response,
            retrieved_contexts=reference_contexts,
        )
    metric_names = list(metric_tasks)
    metric_results = await asyncio.gather(*(metric_tasks[name] for name in metric_names))
    results = dict(zip(metric_names, metric_results))
    return {
        "case_id": record.get("metadata", {}).get("case_id", ""),
        "turn_index": record.get("metadata", {}).get("turn_index", 0),
        "faithfulness": results["faithfulness"].value if "faithfulness" in results else None,
        **{
            name: result.value
            for name, result in results.items()
            if name != "faithfulness"
        },
    }


def evaluate_workflow_records(records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Score slot-collection turns separately from knowledge-answer metrics."""
    rows: List[Dict[str, Any]] = []
    for record in records:
        metadata = record.get("metadata", {})
        if metadata.get("evaluation_mode") != "workflow":
            continue
        expected_behavior = metadata.get("expected_behavior", "")
        response = str(record.get("response", ""))
        if expected_behavior == "clarify_order_id":
            passed = "订单号" in response
        elif expected_behavior == "continue_workflow":
            repeat_slot_prompts = ("请提供订单号", "提供订单号", "补充订单号")
            passed = not any(prompt in response for prompt in repeat_slot_prompts)
        else:
            passed = False
        rows.append({
            "case_id": metadata.get("case_id", ""),
            "turn_index": metadata.get("turn_index", 0),
            "expected_behavior": expected_behavior,
            "passed": passed,
        })
    return rows


def _write_report(path: Path, report: Dict[str, Any]) -> None:
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")


def _parse_bool(value: Any, *, name: str) -> bool:
    normalized = str(value).strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off", ""}:
        return False
    raise ValueError(f"{name} must be true or false")


if __name__ == "__main__":
    raise SystemExit(main())
