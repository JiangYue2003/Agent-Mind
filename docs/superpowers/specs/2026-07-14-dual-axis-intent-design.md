# Dual-Axis Intent Recognition Design

## Goal

Keep the existing `IntentCategory` contract stable while adding two independent
semantic axes:

- `speech_act`: `information`, `operation`, `complaint`, `escalation`, `social`,
  or `ood`.
- `domain`: `billing`, `account`, `technical`, `order`, `general`, or `unknown`.

The axes resolve the current conflict where one legacy label must represent both
the user's requested action and the business domain.

## Compatibility

`IntentResult.intent` remains the current weighted three-source category and all
existing Agent, skill, and API consumers continue to use it. The LLM classifier
adds the axes to its JSON response. If it fails or returns an invalid axis, the
recognizer derives a deterministic fallback from the legacy category.

## Workflow Use

`WorkflowIntentDecider` accepts optional axes and includes them in its decision
prompt as auxiliary evidence. The prompt explicitly says that the original
message, resolved order ID, and history remain authoritative. This strengthens
the existing independent workflow decision rather than making it depend on an
intent label.

The API conditionally passes axes only when the injected decider accepts the
new parameters, preserving existing fake deciders and third-party test doubles.

## Verification

Focused tests verify axis parsing, fallback derivation, and that a workflow
decision prompt receives both axes. Existing workflow and chat tests validate
legacy compatibility.
