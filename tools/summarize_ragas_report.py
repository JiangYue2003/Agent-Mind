"""Summarize one EchoMind Ragas evaluation report without third-party packages."""
import argparse
import json
import math
import statistics
from pathlib import Path
from typing import Any, Dict, Iterable, List


METRICS = ("faithfulness", "factual_correctness", "answer_relevancy")


def build_summary(report: Dict[str, Any]) -> Dict[str, Any]:
    """Return aggregate metrics, coverage, workflow results, and lowest-scoring cases."""
    records = _as_list(report.get("records"))
    ragas_rows = _as_list(report.get("ragas_rows"))
    workflow_rows = _as_list(report.get("workflow_rows"))
    record_index = _index_records(records)

    return {
        "run_id": str(report.get("run_id", "")),
        "status": str(report.get("status", "")),
        "evaluation_config": report.get("evaluation_config", {}),
        "counts": {
            "records": len(records),
            "ragas_rows": len(ragas_rows),
            "workflow_rows": len(workflow_rows),
        },
        "ragas_metrics": _summarize_rows(ragas_rows),
        "coverage": _coverage(records),
        "workflow": _workflow_summary(workflow_rows),
        "by_category": _grouped_summary(ragas_rows, record_index, "category"),
        "by_difficulty": _grouped_summary(ragas_rows, record_index, "difficulty"),
        "worst_cases": _worst_cases(ragas_rows, record_index),
    }


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Summarize one EchoMind Ragas evaluation report")
    parser.add_argument("report", type=Path, help="path to data/eval/ragas-*.json")
    parser.add_argument("--output", type=Path, help="optional JSON output path")
    args = parser.parse_args(argv)

    report = json.loads(args.report.read_text(encoding="utf-8"))
    if not isinstance(report, dict):
        raise ValueError("Ragas report must be a JSON object")
    summary = build_summary(report)
    rendered = json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)
    return 0


def _as_list(value: Any) -> List[Dict[str, Any]]:
    return [item for item in value if isinstance(item, dict)] if isinstance(value, list) else []


def _index_records(records: Iterable[Dict[str, Any]]) -> Dict[tuple[str, int], Dict[str, Any]]:
    index: Dict[tuple[str, int], Dict[str, Any]] = {}
    for record in records:
        metadata = record.get("metadata", {})
        if not isinstance(metadata, dict):
            continue
        index[(str(metadata.get("case_id", "")), _as_int(metadata.get("turn_index")))] = record
    return index


def _summarize_rows(rows: Iterable[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    rows = list(rows)
    return {metric: _metric_summary(row.get(metric) for row in rows) for metric in METRICS}


def _metric_summary(values: Iterable[Any]) -> Dict[str, Any]:
    raw_values = list(values)
    scores = [float(value) for value in raw_values if _is_finite_number(value)]
    missing_count = len(raw_values) - len(scores)
    if not scores:
        return {
            "valid_count": 0,
            "missing_count": missing_count,
            "mean": None,
            "median": None,
            "p10": None,
            "p90": None,
            "min": None,
            "max": None,
            "stddev": None,
        }
    scores.sort()
    return {
        "valid_count": len(scores),
        "missing_count": missing_count,
        "mean": _round(statistics.mean(scores)),
        "median": _round(statistics.median(scores)),
        "p10": _round(_percentile(scores, 0.1)),
        "p90": _round(_percentile(scores, 0.9)),
        "min": _round(scores[0]),
        "max": _round(scores[-1]),
        "stddev": _round(statistics.pstdev(scores)),
    }


def _coverage(records: Iterable[Dict[str, Any]]) -> Dict[str, Any]:
    records = list(records)
    total = len(records)
    return {
        "knowledge_used_count": sum(bool(_as_dict(record.get("chat")).get("knowledge_used")) for record in records),
        "chat_trace_context_count": sum(
            _as_dict(record.get("chat")).get("retrieved_context_source") == "chat_trace"
            for record in records
        ),
        "rerank_applied_count": sum(bool(_as_dict(record.get("search")).get("rerank_applied")) for record in records),
        "knowledge_used_rate": _rate(
            sum(bool(_as_dict(record.get("chat")).get("knowledge_used")) for record in records), total
        ),
        "rerank_applied_rate": _rate(
            sum(bool(_as_dict(record.get("search")).get("rerank_applied")) for record in records), total
        ),
    }


def _workflow_summary(rows: Iterable[Dict[str, Any]]) -> Dict[str, Any]:
    rows = list(rows)
    passed = sum(row.get("passed") is True for row in rows)
    by_behavior: Dict[str, Dict[str, Any]] = {}
    for row in rows:
        behavior = str(row.get("expected_behavior", "unknown"))
        bucket = by_behavior.setdefault(behavior, {"total": 0, "passed": 0, "pass_rate": None})
        bucket["total"] += 1
        bucket["passed"] += int(row.get("passed") is True)
    for bucket in by_behavior.values():
        bucket["pass_rate"] = _rate(bucket["passed"], bucket["total"])
    return {"total": len(rows), "passed": passed, "pass_rate": _rate(passed, len(rows)), "by_expected_behavior": by_behavior}


def _grouped_summary(
    rows: Iterable[Dict[str, Any]],
    record_index: Dict[tuple[str, int], Dict[str, Any]],
    field: str,
) -> Dict[str, Dict[str, Dict[str, Any]]]:
    groups: Dict[str, List[Dict[str, Any]]] = {}
    for row in rows:
        record = record_index.get(_row_key(row), {})
        metadata = _as_dict(record.get("metadata"))
        groups.setdefault(str(metadata.get(field, "unknown")), []).append(row)
    return {name: _summarize_rows(group_rows) for name, group_rows in sorted(groups.items())}


def _worst_cases(
    rows: Iterable[Dict[str, Any]],
    record_index: Dict[tuple[str, int], Dict[str, Any]],
) -> Dict[str, List[Dict[str, Any]]]:
    results: Dict[str, List[Dict[str, Any]]] = {}
    for metric in METRICS:
        ranked = [row for row in rows if _is_finite_number(row.get(metric))]
        ranked.sort(key=lambda row: float(row[metric]))
        results[metric] = [_case_summary(row, record_index, metric) for row in ranked[:10]]
    return results


def _case_summary(row: Dict[str, Any], record_index: Dict[tuple[str, int], Dict[str, Any]], metric: str) -> Dict[str, Any]:
    record = record_index.get(_row_key(row), {})
    metadata = _as_dict(record.get("metadata"))
    return {
        "case_id": str(row.get("case_id", "")),
        "turn_index": _as_int(row.get("turn_index")),
        "user_input": str(record.get("user_input", "")),
        "category": str(metadata.get("category", "unknown")),
        "difficulty": str(metadata.get("difficulty", "unknown")),
        "score": _round(float(row[metric])),
    }


def _row_key(row: Dict[str, Any]) -> tuple[str, int]:
    return str(row.get("case_id", "")), _as_int(row.get("turn_index"))


def _is_finite_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _as_dict(value: Any) -> Dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _as_int(value: Any) -> int:
    return value if isinstance(value, int) else 0


def _percentile(values: List[float], percentile: float) -> float:
    if len(values) == 1:
        return values[0]
    index = (len(values) - 1) * percentile
    lower = math.floor(index)
    upper = math.ceil(index)
    if lower == upper:
        return values[lower]
    return values[lower] + (values[upper] - values[lower]) * (index - lower)


def _rate(numerator: int, denominator: int) -> float | None:
    return _round(numerator / denominator) if denominator else None


def _round(value: float) -> float:
    return round(value, 4)


if __name__ == "__main__":
    raise SystemExit(main())
