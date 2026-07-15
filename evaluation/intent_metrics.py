from collections import Counter
from typing import Any, Dict, Iterable, List, Tuple

from core.intent_recognizer import IntentCategory, IntentDomain, SpeechAct


INTENT_LABELS: Tuple[str, ...] = tuple(category.value for category in IntentCategory)
SPEECH_ACT_LABELS: Tuple[str, ...] = tuple(item.value for item in SpeechAct)
DOMAIN_LABELS: Tuple[str, ...] = tuple(item.value for item in IntentDomain)


def compute_intent_metrics(rows: Iterable[Dict[str, Any]]) -> Dict[str, Any]:
    """Compute deterministic single-label classification metrics for intent cases."""
    return _compute_label_metrics(rows, "expected_intent", "predicted_intent", INTENT_LABELS)


def compute_dual_axis_metrics(rows: Iterable[Dict[str, Any]]) -> Dict[str, Any]:
    """Score speech act and business domain independently plus their joint match."""
    rows = list(rows)
    evaluated = [
        row for row in rows
        if (
            row.get("expected_speech_act") in SPEECH_ACT_LABELS
            and row.get("expected_domain") in DOMAIN_LABELS
        )
    ]
    joint_correct = sum(
        row.get("predicted_speech_act") == row["expected_speech_act"]
        and row.get("predicted_domain") == row["expected_domain"]
        for row in evaluated
    )
    total = len(evaluated)
    return {
        "total": total,
        "speech_act": _compute_label_metrics(
            evaluated, "expected_speech_act", "predicted_speech_act", SPEECH_ACT_LABELS,
        ),
        "domain": _compute_label_metrics(
            evaluated, "expected_domain", "predicted_domain", DOMAIN_LABELS,
        ),
        "joint_exact_match": round(joint_correct / total, 4) if total else 0.0,
    }


def _compute_label_metrics(
    rows: Iterable[Dict[str, Any]],
    expected_key: str,
    predicted_key: str,
    labels: Tuple[str, ...],
) -> Dict[str, Any]:
    evaluated = [
        row for row in rows
        if isinstance(row.get(expected_key), str) and row[expected_key] in labels
    ]
    confusion = {
        expected: {predicted: 0 for predicted in labels}
        for expected in labels
    }
    correct = 0
    for row in evaluated:
        expected = row[expected_key]
        predicted = str(row.get(predicted_key) or "")
        if predicted in labels:
            confusion[expected][predicted] += 1
        if predicted == expected:
            correct += 1

    per_class: Dict[str, Dict[str, float]] = {}
    for label in labels:
        tp = confusion[label][label]
        fp = sum(confusion[expected][label] for expected in labels if expected != label)
        fn = sum(confusion[label][predicted] for predicted in labels if predicted != label)
        support = sum(confusion[label].values())
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        per_class[label] = {
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
            "support": support,
        }

    total = len(evaluated)
    macro_f1 = sum(item["f1"] for item in per_class.values()) / len(labels)
    return {
        "total": total,
        "correct": correct,
        "accuracy": round(correct / total, 4) if total else 0.0,
        "macro_f1": round(macro_f1, 4),
        "per_class": per_class,
        "confusion_matrix": confusion,
    }


def compute_workflow_metrics(rows: Iterable[Dict[str, Any]]) -> Dict[str, Any]:
    """Evaluate mode, tool set, and clarification independently and jointly."""
    evaluated = [
        row for row in rows
        if (
            isinstance(row.get("expected_workflow_mode"), str)
            and isinstance(row.get("expected_tools"), list)
            and isinstance(row.get("expected_should_clarify"), bool)
        )
    ]
    mode_correct = 0
    tools_correct = 0
    clarify_correct = 0
    exact_correct = 0
    for row in evaluated:
        workflow = row.get("workflow") if isinstance(row.get("workflow"), dict) else {}
        mode_ok = workflow.get("mode") == row["expected_workflow_mode"]
        tools_ok = set(workflow.get("tools") or []) == set(row["expected_tools"])
        clarify_ok = workflow.get("should_clarify") is row["expected_should_clarify"]
        mode_correct += int(mode_ok)
        tools_correct += int(tools_ok)
        clarify_correct += int(clarify_ok)
        exact_correct += int(mode_ok and tools_ok and clarify_ok)

    total = len(evaluated)
    return {
        "total": total,
        "mode_accuracy": round(mode_correct / total, 4) if total else 0.0,
        "tool_set_accuracy": round(tools_correct / total, 4) if total else 0.0,
        "clarify_accuracy": round(clarify_correct / total, 4) if total else 0.0,
        "exact_match": round(exact_correct / total, 4) if total else 0.0,
    }
