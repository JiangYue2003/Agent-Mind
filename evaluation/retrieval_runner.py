"""Run deterministic retrieval-only evaluation against EchoMind's /search endpoint."""
import argparse
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Protocol

import httpx

from evaluation.retrieval_metrics import (
    content_sha256,
    evaluate_retrieval_records,
    load_gold_chunk_hashes,
    load_gold_chunk_keys,
)


class HttpResponse(Protocol):
    def raise_for_status(self) -> None: ...
    def json(self) -> Any: ...


class HttpClient(Protocol):
    def get(self, path: str, params: Dict[str, Any] | None = None) -> HttpResponse: ...
    def post(
        self,
        path: str,
        params: Dict[str, Any] | None = None,
        json: Dict[str, Any] | None = None,
    ) -> HttpResponse: ...


def load_retrieval_cases(path: Path) -> List[Dict[str, Any]]:
    """Load search-only cases that do not depend on chat workflow behavior."""
    cases: List[Dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            case = json.loads(line)
        except json.JSONDecodeError as ex:
            raise ValueError(f"invalid JSON at {path}:{line_number}") from ex
        if not isinstance(case, dict):
            raise ValueError(f"retrieval case at {path}:{line_number} must be an object")
        _validate_case(case, line_number=line_number)
        cases.append(case)
    if not cases:
        raise ValueError("retrieval dataset is empty")
    return cases


def collect_retrieval_records(
    cases: List[Dict[str, Any]],
    *,
    client: HttpClient,
    top_k: int,
    recall_k: int,
    require_rerank: bool,
    required_rerank_provider: str,
) -> List[Dict[str, Any]]:
    """Call /search exactly once for every case; /chat is intentionally not used."""
    if top_k < 1 or recall_k <= top_k:
        raise ValueError("recall_k must be greater than positive top_k")

    records: List[Dict[str, Any]] = []
    for case in cases:
        response = client.post(
            "/search",
            params={
                "query": case["query"],
                "top_k": top_k,
                "recall_k": recall_k,
                "include_debug": True,
            },
        )
        response.raise_for_status()
        payload = response.json()
        if not isinstance(payload, dict):
            raise ValueError(f"search for {case['case_id']} returned a non-object payload")
        debug = payload.get("retrieval_debug")
        if not isinstance(debug, dict):
            raise ValueError(f"search for {case['case_id']} did not return retrieval diagnostics")
        raw_candidates = _extract_candidates(debug.get("raw_candidates"), field_name="raw_candidates")
        reranked_candidates = _extract_candidates(
            debug.get("reranked_candidates"), field_name="reranked_candidates"
        )
        answerable = bool(case["expected_answerable"])
        applied = bool(payload.get("rerank_applied", False))
        providers = _as_string_set(payload.get("rerank_providers"))
        if require_rerank and answerable:
            if not applied:
                raise ValueError(f"search for {case['case_id']} did not use a real reranker")
            if required_rerank_provider and required_rerank_provider not in providers:
                raise ValueError(
                    f"search for {case['case_id']} did not use required reranker "
                    f"{required_rerank_provider!r}; got {providers!r}"
                )
        records.append({
            "user_input": case["query"],
            "search": {
                "top_k": top_k,
                "recall_k": recall_k,
                "rerank_applied": applied,
                "rerank_providers": providers,
                "raw_candidates": raw_candidates,
                "reranked_candidates": reranked_candidates,
            },
            "metadata": {
                "case_id": case["case_id"],
                "unanswerable": not answerable,
                "evaluation_mode": "knowledge",
                "case_type": case["case_type"],
                "category": case["category"],
                "difficulty": case["difficulty"],
                "source_file": case.get("source_file", ""),
            },
        })
    return records


def run_retrieval_evaluation(
    *,
    dataset_path: Path,
    gold_path: Path,
    output_dir: Path,
    client: HttpClient,
    run_id: str,
    top_k: int,
    recall_k: int,
    require_rerank: bool,
    required_rerank_provider: str,
) -> Dict[str, Any]:
    cases = load_retrieval_cases(dataset_path)
    gold_chunk_keys = load_gold_chunk_keys(gold_path)
    gold_chunk_hashes = load_gold_chunk_hashes(gold_path)
    validate_deployed_gold(cases, gold_chunk_keys=gold_chunk_keys, gold_chunk_hashes=gold_chunk_hashes, client=client)
    records = collect_retrieval_records(
        cases,
        client=client,
        top_k=top_k,
        recall_k=recall_k,
        require_rerank=require_rerank,
        required_rerank_provider=required_rerank_provider,
    )
    report = {
        "run_id": run_id,
        "status": "completed",
        "error": None,
        "summary": {
            "total_cases": len(cases),
            "answerable_cases": sum(case["expected_answerable"] for case in cases),
            "unanswerable_cases": sum(not case["expected_answerable"] for case in cases),
        },
        "records": records,
        "retrieval": evaluate_retrieval_records(
            records,
            gold_chunk_keys=gold_chunk_keys,
            raw_k=recall_k,
            final_k=top_k,
        ),
        "evaluation_config": {
            "dataset_path": str(dataset_path),
            "gold_path": str(gold_path),
            "top_k": top_k,
            "recall_k": recall_k,
            "require_rerank": require_rerank,
            "required_rerank_provider": required_rerank_provider,
        },
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / f"{run_id}.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return report


def validate_deployed_gold(
    cases: List[Dict[str, Any]],
    *,
    gold_chunk_keys: Dict[str, List[str]],
    gold_chunk_hashes: Dict[str, str],
    client: HttpClient,
) -> None:
    answerable_case_ids = [case["case_id"] for case in cases if case["expected_answerable"]]
    missing_cases = sorted(case_id for case_id in answerable_case_ids if not gold_chunk_keys.get(case_id))
    if missing_cases:
        raise ValueError("missing retrieval gold for answerable cases: " + ", ".join(missing_cases))
    required_keys = {key for case_id in answerable_case_ids for key in gold_chunk_keys[case_id]}
    missing_hashes = sorted(required_keys - set(gold_chunk_hashes))
    if missing_hashes:
        raise ValueError("missing content hashes for gold chunks: " + ", ".join(missing_hashes[:5]))

    response = client.get("/knowledge/chunks", params={"limit": 5000})
    response.raise_for_status()
    payload = response.json()
    items = payload.get("items", []) if isinstance(payload, dict) else []
    deployed_hashes: Dict[str, str] = {}
    for item in items:
        if not isinstance(item, dict):
            continue
        key = str(item.get("chunk_key", "")).strip()
        content = item.get("content")
        if key and isinstance(content, str):
            deployed_hashes[key] = content_sha256(content)
    missing_keys = sorted(required_keys - set(deployed_hashes))
    if missing_keys:
        raise ValueError("deployed knowledge base is missing gold chunks: " + ", ".join(missing_keys[:5]))
    changed_keys = sorted(key for key in required_keys if deployed_hashes[key] != gold_chunk_hashes[key])
    if changed_keys:
        raise ValueError("deployed gold chunk content changed: " + ", ".join(changed_keys[:5]))


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Evaluate EchoMind retrieval with /search only")
    parser.add_argument("--base-url", default=os.getenv("SEARCH_EVAL_APP_BASE_URL", "http://127.0.0.1:8000"))
    parser.add_argument("--dataset", default="evaluation/datasets/customer_service_search_v1.jsonl")
    parser.add_argument("--gold", default="evaluation/datasets/customer_service_search_gold_v2.json")
    parser.add_argument("--output-dir", default="data/eval")
    parser.add_argument("--run-id", default="")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--recall-k", type=int, default=20)
    parser.add_argument("--require-rerank", default="true")
    parser.add_argument("--required-rerank-provider", default="tei")
    args = parser.parse_args(argv)
    require_rerank = _parse_bool(args.require_rerank)
    run_id = args.run_id or f"search-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    with httpx.Client(base_url=args.base_url, timeout=60.0) as client:
        report = run_retrieval_evaluation(
            dataset_path=Path(args.dataset),
            gold_path=Path(args.gold),
            output_dir=Path(args.output_dir),
            client=client,
            run_id=run_id,
            top_k=args.top_k,
            recall_k=args.recall_k,
            require_rerank=require_rerank,
            required_rerank_provider=args.required_rerank_provider,
        )
    summary = report["retrieval"]["summary"]
    print(
        f"Retrieval evaluation complete: {report['summary']['answerable_cases']} answerable, "
        f"{report['summary']['unanswerable_cases']} OOD; "
        f"Hit@{args.recall_k}={summary['raw']['hit_at_k']:.4f}, "
        f"Hit@{args.top_k}={summary['reranked']['hit_at_k']:.4f}"
    )
    return 0


def _validate_case(case: Dict[str, Any], *, line_number: int) -> None:
    required = ("case_id", "query", "expected_answerable", "case_type", "category", "difficulty")
    missing = [field for field in required if field not in case]
    if missing:
        raise ValueError(f"retrieval case at line {line_number} is missing: {', '.join(missing)}")
    if not isinstance(case["case_id"], str) or not case["case_id"].strip():
        raise ValueError(f"retrieval case at line {line_number} has invalid case_id")
    if not isinstance(case["query"], str) or not case["query"].strip():
        raise ValueError(f"retrieval case at line {line_number} has invalid query")
    if not isinstance(case["expected_answerable"], bool):
        raise ValueError(f"retrieval case at line {line_number} expected_answerable must be boolean")
    if case["expected_answerable"]:
        for field in ("source_file", "target_fact", "evidence_text"):
            if not isinstance(case.get(field), str) or not case[field].strip():
                raise ValueError(f"retrieval case at line {line_number} requires {field}")
    elif case["case_type"] != "ood":
        raise ValueError(f"retrieval OOD case at line {line_number} must use case_type=ood")


def _extract_candidates(value: Any, *, field_name: str) -> List[Dict[str, Any]]:
    if not isinstance(value, list):
        raise ValueError(f"retrieval diagnostics field {field_name!r} must be a list")
    candidates: List[Dict[str, Any]] = []
    for index, item in enumerate(value, start=1):
        if not isinstance(item, dict):
            raise ValueError(f"retrieval diagnostics field {field_name!r} has a non-object candidate")
        key = str(item.get("chunk_key", "")).strip()
        if not key:
            raise ValueError(f"retrieval diagnostics field {field_name!r} candidate {index} lacks chunk_key")
        candidates.append({
            "chunk_key": key,
            "rank": int(item.get("rank", index)),
            "score": float(item.get("score", 0.0)),
        })
    return candidates


def _as_string_set(value: Any) -> List[str]:
    if not isinstance(value, list):
        return []
    return sorted({str(item).strip() for item in value if str(item).strip()})


def _parse_bool(value: str) -> bool:
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"invalid boolean value: {value!r}")


if __name__ == "__main__":
    raise SystemExit(main())
