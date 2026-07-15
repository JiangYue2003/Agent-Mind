# Intent And Workflow Evaluation Design

## Goal

Add a reproducible evaluation baseline for EchoMind's existing intent recognition
and workflow decision components. It must measure the two layers separately:

1. Intent classification: the `IntentRecognizer` output against human-labelled
   `IntentCategory` values.
2. Workflow decision: the `WorkflowIntentDecider` output against the expected
   evidence mode, tool set, and clarification behaviour.

The evaluation does not change production routing and does not replace RAGAS.
RAGAS remains responsible for retrieval and answer-quality measurement.

## Dataset Contract

The versioned JSONL datasets contain one object per case:

```json
{
  "case_id": "intent-billing-001",
  "user_input": "钱扣走了但订单还是待付款，要不要再付？",
  "history": [],
  "expected_intent": "billing",
  "expected_urgency": "low",
  "expected_entities": {"order_id": []},
  "expected_workflow_mode": "knowledge",
  "expected_tools": ["knowledge_search"],
  "expected_should_clarify": false,
  "category": "boundary",
  "difficulty": "medium"
}
```

All labels use the current production enums and tool names. The full baseline
contains balanced examples for all ten intent labels, OOD cases, boundary cases,
and multi-turn cases. The smoke dataset is a stable, representative subset.

For overlapping messages, labels use this precedence: explicit transfer or
complaint remains `escalation` or `complaint`; an explicit operation is
`request`; payment, invoice, and refund rule/status questions are `billing`;
account settings questions without an explicit operation are `account`.

## Evaluation Flow

`evaluation.intent_runner` loads and validates the dataset, creates the real
`IntentRecognizer` and `WorkflowIntentDecider` from the current
`ANTHROPIC_*` configuration, and evaluates each case in isolation. This avoids
the recognizer's message-only cache contaminating one test case with another.

For each case it records prediction, intent confidence, urgency, entities,
latency, workflow decision, and any component failure. It writes a timestamped
JSON artifact under `data/eval`; existing reports are never overwritten.

## Metrics

Intent metrics: accuracy, macro-F1, fixed-label per-class precision/recall/F1,
and a confusion matrix. Workflow metrics: exact mode match, exact tool-set
match, exact clarification match, and full exact match. Metrics are reported
only for cases declaring their corresponding expected fields.

Confidence calibration is intentionally out of scope for v1: the current
recognizer publishes the LLM confidence while the final category is selected by
a weighted multi-source vote, so that value is not yet the final prediction
probability.

## Verification

Tests validate dataset records, deterministic metric calculations, and runner
behaviour with fake recognizer/decider components. A real run is performed with
the current Anthropic-compatible DeepSeek configuration from the target runtime.
