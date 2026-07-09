from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional

from workflow.action_models import (
    ActionItem,
    ActionStatus,
    ActionType,
    EvidenceItem,
    EvidenceStore,
    RoutePlan,
    WorkflowPlan,
)
from workflow.tool_registry import ToolRegistry


@dataclass
class ExecutionResult:
    evidence_store: EvidenceStore = field(default_factory=EvidenceStore)
    action_statuses: Dict[str, str] = field(default_factory=dict)
    context_blocks: List[str] = field(default_factory=list)
    route_plan: Optional[RoutePlan] = None
    knowledge_used: bool = False
    degraded: bool = False
    failed_actions: List[str] = field(default_factory=list)


class ActionExecutor:
    def __init__(self, registry: ToolRegistry):
        self._registry = registry

    async def execute(self, plan: WorkflowPlan, runtime: Any = None) -> ExecutionResult:
        evidence_store = EvidenceStore()
        statuses = {action.id: ActionStatus.PENDING.value for action in plan.actions}
        failed_actions: List[str] = []
        failures = 0
        executed = 0
        degraded = False

        while self._has_pending(statuses):
            ready = [
                action for action in plan.actions
                if statuses[action.id] == ActionStatus.PENDING.value
                and self._deps_satisfied(action, statuses)
                and self._inputs_satisfied(action, evidence_store)
                and self._slots_satisfied(action, runtime)
            ]
            if not ready:
                for action in plan.actions:
                    if statuses[action.id] == ActionStatus.PENDING.value:
                        statuses[action.id] = ActionStatus.BLOCKED.value
                break

            groups = self._build_groups(ready)
            for group in groups:
                if executed >= plan.stop_conditions.max_actions:
                    degraded = True
                    for action in plan.actions:
                        if statuses[action.id] == ActionStatus.PENDING.value:
                            statuses[action.id] = ActionStatus.SKIPPED.value
                    break

                outcomes = await self._run_group(group, runtime, evidence_store)
                for action, outcome in zip(group, outcomes):
                    statuses[action.id] = outcome["status"]
                    for item in outcome["evidence_items"]:
                        evidence_store.put(item)
                    if outcome["status"] == ActionStatus.FAILED.value:
                        failures += 1
                        failed_actions.append(action.id)
                    executed += 1

                if failures > plan.stop_conditions.max_failures:
                    degraded = True
                    for action in plan.actions:
                        if statuses[action.id] == ActionStatus.PENDING.value:
                            statuses[action.id] = ActionStatus.BLOCKED.value
                    break

            if degraded:
                break

        return ExecutionResult(
            evidence_store=evidence_store,
            action_statuses=statuses,
            context_blocks=self._build_context_blocks(evidence_store),
            route_plan=plan.route_plan,
            knowledge_used=any(item.source == "knowledge_search" for item in evidence_store.values()),
            degraded=degraded,
            failed_actions=failed_actions,
        )

    async def _run_group(
        self,
        actions: List[ActionItem],
        runtime: Any,
        evidence_store: EvidenceStore,
    ) -> List[Dict[str, Any]]:
        if len(actions) == 1:
            return [await self._run_action(actions[0], runtime, evidence_store)]
        return await asyncio.gather(*(self._run_action(action, runtime, evidence_store) for action in actions))

    async def _run_action(
        self,
        action: ActionItem,
        runtime: Any,
        evidence_store: EvidenceStore,
    ) -> Dict[str, Any]:
        trace = self._runtime_get(runtime, "trace")
        attempts = max(action.retry_limit, 0) + 1
        last_error = ""
        for _ in range(attempts):
            try:
                if trace is None:
                    raw = await self._registry.execute(action, runtime, evidence_store)
                else:
                    with trace.stage(f"workflow.action.{action.id}", action_type=action.type.value):
                        raw = await self._registry.execute(action, runtime, evidence_store)
                items = self._normalize_evidence(raw)
                return {"status": ActionStatus.SUCCEEDED.value, "evidence_items": items}
            except Exception as ex:  # pragma: no cover - failures are asserted by caller state
                last_error = str(ex)
        return {
            "status": ActionStatus.FAILED.value,
            "evidence_items": [
                EvidenceItem(
                    key=f"{action.id}.error",
                    source="workflow",
                    value={"error": last_error},
                )
            ] if last_error else [],
        }

    def _build_groups(self, ready: List[ActionItem]) -> List[List[ActionItem]]:
        parallel = [action for action in ready if action.can_parallel]
        sequential = [action for action in ready if not action.can_parallel]
        groups: List[List[ActionItem]] = []
        if parallel:
            groups.append(parallel)
        groups.extend([[action] for action in sequential])
        return groups

    def _deps_satisfied(self, action: ActionItem, statuses: Dict[str, str]) -> bool:
        return all(statuses.get(dep_id) == ActionStatus.SUCCEEDED.value for dep_id in action.depends_on)

    def _inputs_satisfied(self, action: ActionItem, evidence_store: EvidenceStore) -> bool:
        if not action.input_keys:
            return True
        return evidence_store.has_all(action.input_keys)

    def _slots_satisfied(self, action: ActionItem, runtime: Any) -> bool:
        if not action.required_slots:
            return True
        entities = self._runtime_get(runtime, "entities", {}) or {}
        for slot in action.required_slots:
            values = entities.get(slot) or []
            if not values:
                return False
        return True

    def _build_context_blocks(self, evidence_store: EvidenceStore) -> List[str]:
        knowledge_blocks: List[str] = []
        tool_blocks: List[str] = []
        called_tools: List[str] = []

        for item in evidence_store.values():
            if not item.prompt_block:
                continue
            if item.source == "knowledge_search":
                knowledge_blocks.append(item.prompt_block)
            else:
                tool_blocks.append(item.prompt_block)
                if item.tool_name:
                    called_tools.append(item.tool_name)

        context_blocks = list(knowledge_blocks)
        if tool_blocks:
            summary = ["[工具增强上下文]", "", "[工具执行摘要]"]
            summary.append(f"- 已调用工具: {', '.join(dict.fromkeys(called_tools)) if called_tools else '无'}")
            summary.append("- 事实优先级: 实时订单/物流/退款结果优先于通用知识说明；人工转接结果优先于推测性答复")
            context_blocks.append("\n".join(summary + [""] + tool_blocks))
        return context_blocks

    def _normalize_evidence(self, raw: Any) -> List[EvidenceItem]:
        if raw is None:
            return []
        if isinstance(raw, EvidenceItem):
            return [raw]
        if isinstance(raw, list):
            return [item for item in raw if isinstance(item, EvidenceItem)]
        return []

    def _has_pending(self, statuses: Dict[str, str]) -> bool:
        return any(status == ActionStatus.PENDING.value for status in statuses.values())

    @staticmethod
    def _runtime_get(runtime: Any, key: str, default: Any = None) -> Any:
        if isinstance(runtime, dict):
            return runtime.get(key, default)
        return getattr(runtime, key, default)
