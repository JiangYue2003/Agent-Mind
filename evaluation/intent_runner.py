"""Run versioned intent and workflow evaluations against real EchoMind components."""
import argparse
import asyncio
import inspect
import json
import os
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from core.intent_recognizer import IntentCategory, IntentDomain, IntentRecognizer, SpeechAct, UrgencyLevel
from evaluation.intent_metrics import compute_dual_axis_metrics, compute_intent_metrics, compute_workflow_metrics
from workflow.intent_decider import DecisionMode, WorkflowIntentDecider


_WORKFLOW_TOOLS = {"knowledge_search", "order_lookup", "shipment_track", "refund_create"}


@dataclass
class IntentEvalCase:
    case_id: str
    user_input: str
    expected_intent: str
    history: List[Dict[str, str]] = field(default_factory=list)
    expected_speech_act: Optional[str] = None
    expected_domain: Optional[str] = None
    expected_urgency: Optional[str] = None
    expected_entities: Optional[Dict[str, List[str]]] = None
    expected_workflow_mode: Optional[str] = None
    expected_tools: Optional[List[str]] = None
    expected_should_clarify: Optional[bool] = None
    category: str = ""
    difficulty: str = ""


def load_intent_cases(path: Path) -> List[IntentEvalCase]:
    cases: List[IntentEvalCase] = []
    seen_case_ids = set()
    for line_number, raw_line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not raw_line.strip():
            continue
        try:
            payload = json.loads(raw_line)
        except json.JSONDecodeError as ex:
            raise ValueError(f"invalid JSON on line {line_number}") from ex
        if not isinstance(payload, dict):
            raise ValueError(f"case on line {line_number} must be an object")

        case_id = str(payload.get("case_id", "")).strip()
        user_input = str(payload.get("user_input", "")).strip()
        expected_intent = str(payload.get("expected_intent", "")).strip()
        if not case_id or case_id in seen_case_ids:
            raise ValueError(f"case_id on line {line_number} must be unique and non-empty")
        if not user_input:
            raise ValueError(f"user_input for {case_id} must be non-empty")
        if expected_intent not in {category.value for category in IntentCategory}:
            raise ValueError(f"expected_intent for {case_id} is invalid: {expected_intent}")
        expected_speech_act = payload.get("expected_speech_act")
        expected_domain = payload.get("expected_domain")
        if expected_speech_act is not None and expected_speech_act not in {item.value for item in SpeechAct}:
            raise ValueError(f"expected_speech_act for {case_id} is invalid: {expected_speech_act}")
        if expected_domain is not None and expected_domain not in {item.value for item in IntentDomain}:
            raise ValueError(f"expected_domain for {case_id} is invalid: {expected_domain}")
        if (expected_speech_act is None) != (expected_domain is None):
            raise ValueError(f"dual-axis expectations for {case_id} must be declared together")

        history = payload.get("history", [])
        if not isinstance(history, list) or not all(
            isinstance(item, dict) and isinstance(item.get("role"), str) and isinstance(item.get("content"), str)
            for item in history
        ):
            raise ValueError(f"history for {case_id} must be role/content objects")
        mode = payload.get("expected_workflow_mode")
        if mode is not None and mode not in {item.value for item in DecisionMode}:
            raise ValueError(f"expected_workflow_mode for {case_id} is invalid: {mode}")
        tools = payload.get("expected_tools")
        if tools is not None and (
            not isinstance(tools, list) or not all(isinstance(tool, str) and tool in _WORKFLOW_TOOLS for tool in tools)
        ):
            raise ValueError(f"expected_tools for {case_id} contains an unknown tool")
        clarify = payload.get("expected_should_clarify")
        if clarify is not None and not isinstance(clarify, bool):
            raise ValueError(f"expected_should_clarify for {case_id} must be boolean")
        if any(value is not None for value in (mode, tools, clarify)) and not all(
            value is not None for value in (mode, tools, clarify)
        ):
            raise ValueError(f"workflow expectations for {case_id} must be declared together")

        seen_case_ids.add(case_id)
        cases.append(IntentEvalCase(
            case_id=case_id,
            user_input=user_input,
            expected_intent=expected_intent,
            history=history,
            expected_speech_act=expected_speech_act,
            expected_domain=expected_domain,
            expected_urgency=_optional_string(payload.get("expected_urgency")),
            expected_entities=payload.get("expected_entities"),
            expected_workflow_mode=mode,
            expected_tools=tools,
            expected_should_clarify=clarify,
            category=_optional_string(payload.get("category")) or "",
            difficulty=_optional_string(payload.get("difficulty")) or "",
        ))
    if not cases:
        raise ValueError("intent dataset must contain at least one case")
    return cases


async def evaluate_cases(
    cases: Sequence[IntentEvalCase],
    *,
    recognizer: IntentRecognizer,
    decider: Optional[WorkflowIntentDecider],
    scope: str,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for case in cases:
        # History-sensitive utterances must not reuse a previous case's message-only cache entry.
        cache = getattr(recognizer, "_cache", None)
        if isinstance(cache, dict):
            cache.clear()
        row = asdict(case)
        try:
            result = await recognizer.recognize(case.user_input, history=case.history or None)
            speech_act = getattr(getattr(result, "speech_act", None), "value", None)
            domain = getattr(getattr(result, "domain", None), "value", None)
            row.update({
                "predicted_intent": result.intent.value,
                "predicted_speech_act": speech_act,
                "predicted_domain": domain,
                "intent_confidence": result.confidence,
                "predicted_urgency": result.urgency.name.lower(),
                "predicted_entities": result.entities,
                "intent_reasoning": result.reasoning,
                "latency_ms": result.latency_ms,
                "intent_error": None,
            })
        except Exception as ex:  # Keep the report useful when a remote dependency fails.
            result = None
            row.update({
                "predicted_intent": None,
                "intent_error": str(ex),
            })

        if (
            scope in {"all", "workflow"}
            and decider is not None
            and result is not None
            and case.expected_workflow_mode is not None
        ):
            try:
                decision_kwargs = {
                    "message": case.user_input,
                    "intent": result.intent,
                    "entities": result.entities,
                    "history": case.history or None,
                }
                if "speech_act" in inspect.signature(decider.decide).parameters:
                    decision_kwargs["speech_act"] = getattr(result, "speech_act", None)
                    decision_kwargs["domain"] = getattr(result, "domain", None)
                decision = await decider.decide(**decision_kwargs)
                row["workflow"] = decision.to_dict()
                row["workflow_error"] = None
            except Exception as ex:
                row["workflow"] = None
                row["workflow_error"] = str(ex)
        rows.append(row)
    return rows


def build_report(dataset: Path, rows: List[Dict[str, Any]], scope: str) -> Dict[str, Any]:
    latencies = [float(row["latency_ms"]) for row in rows if isinstance(row.get("latency_ms"), (int, float))]
    sorted_latencies = sorted(latencies)
    return {
        "schema_version": "intent-eval-v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "dataset": str(dataset),
        "scope": scope,
        "summary": {
            "intent": compute_intent_metrics(rows),
            "dual_axis": compute_dual_axis_metrics(rows),
            "workflow": compute_workflow_metrics(rows) if scope in {"all", "workflow"} else None,
            "intent_failure_count": sum(1 for row in rows if row.get("intent_error")),
            "workflow_failure_count": sum(1 for row in rows if row.get("workflow_error")),
            "latency_ms": _latency_summary(sorted_latencies),
        },
        "cases": rows,
    }


def _latency_summary(latencies: List[float]) -> Dict[str, float]:
    if not latencies:
        return {"count": 0, "p50": 0.0, "p95": 0.0}
    return {
        "count": len(latencies),
        "p50": round(_percentile(latencies, 0.5), 2),
        "p95": round(_percentile(latencies, 0.95), 2),
    }


def _percentile(values: List[float], percentile: float) -> float:
    index = max(0, min(len(values) - 1, round((len(values) - 1) * percentile)))
    return values[index]


def _optional_string(value: Any) -> Optional[str]:
    return value if isinstance(value, str) else None


def _runtime_components(scope: str) -> tuple[IntentRecognizer, Optional[WorkflowIntentDecider]]:
    api_key = os.getenv("ANTHROPIC_API_KEY", "").strip()
    if not api_key:
        raise ValueError("ANTHROPIC_API_KEY is required for intent evaluation")
    base_url = os.getenv("ANTHROPIC_BASE_URL") or None
    model = os.getenv("ANTHROPIC_MODEL", "deepseek-chat")
    recognizer = IntentRecognizer(api_key=api_key, base_url=base_url, model=model)
    decider = (
        WorkflowIntentDecider(api_key=api_key, base_url=base_url, model=model)
        if scope in {"all", "workflow"} else None
    )
    return recognizer, decider


async def _run(args: argparse.Namespace) -> Path:
    dataset = Path(args.dataset)
    cases = load_intent_cases(dataset)
    recognizer, decider = _runtime_components(args.scope)
    rows = await evaluate_cases(cases, recognizer=recognizer, decider=decider, scope=args.scope)
    report = build_report(dataset, rows, args.scope)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    output_path = output_dir / f"intent-{timestamp}.json"
    output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return output_path


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Evaluate EchoMind intent recognition and workflow decisions")
    parser.add_argument("--dataset", required=True, help="Versioned JSONL intent dataset")
    parser.add_argument("--output-dir", default="data/eval", help="Directory for timestamped JSON reports")
    parser.add_argument("--scope", choices=("intent", "workflow", "all"), default="all")
    args = parser.parse_args(argv)
    output_path = asyncio.run(_run(args))
    print(f"Intent evaluation complete: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
