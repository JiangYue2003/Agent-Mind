from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List

from workflow.action_models import WorkflowPlan


class WorkflowState(Enum):
    INTAKE = "intake"
    PLAN = "plan"
    CLARIFY = "clarify"
    RETRIEVE = "retrieve"
    ACT = "act"
    EXECUTE = "execute"
    VERIFY = "verify"
    RESPOND = "respond"
    HANDOFF = "handoff"
    CLOSE = "close"


@dataclass
class WorkflowPath:
    states: List[WorkflowState]
    action_steps: int
    degraded: bool = False

    def to_dict(self) -> Dict[str, object]:
        return {
            "states": [state.value for state in self.states],
            "action_steps": self.action_steps,
            "degraded": self.degraded,
        }

    def includes(self, state: WorkflowState) -> bool:
        return state in self.states


class WorkflowStateMachine:
    def __init__(self, max_action_steps: int = 3):
        self._max_action_steps = max_action_steps

    def build_path(self, target: Any) -> WorkflowPath:
        if isinstance(target, WorkflowPlan):
            return self._build_plan_path(target)
        return self._build_legacy_path(str(target))

    def _build_legacy_path(self, next_action: str) -> WorkflowPath:
        mapping = {
            "clarify": [WorkflowState.INTAKE, WorkflowState.CLARIFY, WorkflowState.RESPOND, WorkflowState.CLOSE],
            "retrieve": [WorkflowState.INTAKE, WorkflowState.RETRIEVE, WorkflowState.RESPOND, WorkflowState.CLOSE],
            "act": [WorkflowState.INTAKE, WorkflowState.ACT, WorkflowState.RESPOND, WorkflowState.CLOSE],
            "handoff": [WorkflowState.INTAKE, WorkflowState.HANDOFF, WorkflowState.RESPOND, WorkflowState.CLOSE],
            "retrieve_act": [
                WorkflowState.INTAKE,
                WorkflowState.RETRIEVE,
                WorkflowState.ACT,
                WorkflowState.RESPOND,
                WorkflowState.CLOSE,
            ],
            "respond": [WorkflowState.INTAKE, WorkflowState.RESPOND, WorkflowState.CLOSE],
        }
        states = mapping.get(next_action, mapping["respond"])
        action_steps = self._count_action_steps(states)
        if action_steps > self._max_action_steps:
            return WorkflowPath(
                states=[WorkflowState.INTAKE, WorkflowState.RESPOND, WorkflowState.CLOSE],
                action_steps=0,
                degraded=True,
            )
        return WorkflowPath(states=states, action_steps=action_steps, degraded=False)

    def _build_plan_path(self, plan: WorkflowPlan) -> WorkflowPath:
        states: List[WorkflowState] = [WorkflowState.INTAKE, WorkflowState.PLAN]
        if plan.need_clarify:
            states.extend([WorkflowState.CLARIFY, WorkflowState.RESPOND, WorkflowState.CLOSE])
            return WorkflowPath(states=states, action_steps=1, degraded=False)

        if plan.need_knowledge:
            states.append(WorkflowState.RETRIEVE)
        if plan.need_action_tool:
            states.append(WorkflowState.ACT)
        if plan.need_handoff:
            states.append(WorkflowState.HANDOFF)

        if plan.actions:
            states.extend([WorkflowState.EXECUTE, WorkflowState.VERIFY])
        states.extend([WorkflowState.RESPOND, WorkflowState.CLOSE])
        states = self._dedupe(states)

        action_steps = plan.action_count()
        if action_steps > self._max_action_steps:
            return WorkflowPath(
                states=[WorkflowState.INTAKE, WorkflowState.PLAN, WorkflowState.RESPOND, WorkflowState.CLOSE],
                action_steps=0,
                degraded=True,
            )
        return WorkflowPath(states=states, action_steps=action_steps, degraded=False)

    def _count_action_steps(self, states: List[WorkflowState]) -> int:
        return sum(
            1
            for state in states
            if state in {WorkflowState.CLARIFY, WorkflowState.RETRIEVE, WorkflowState.ACT, WorkflowState.HANDOFF}
        )

    def _dedupe(self, states: List[WorkflowState]) -> List[WorkflowState]:
        seen = set()
        result = []
        for state in states:
            if state in seen:
                continue
            seen.add(state)
            result.append(state)
        return result
