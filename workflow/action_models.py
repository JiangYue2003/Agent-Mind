from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Iterable, List, Optional


class ComplexityLevel(str, Enum):
    SIMPLE = "simple"
    SINGLE_DOMAIN = "single_domain"
    MULTI_SOURCE = "multi_source"
    CROSS_DOMAIN = "cross_domain"
    EXCEPTION = "exception"


class ActionType(str, Enum):
    CLARIFY_SLOT = "clarify_slot"
    CONFIRM_GOAL = "confirm_goal"
    RETRIEVE_POLICY = "retrieve_policy"
    RETRIEVE_FAQ = "retrieve_faq"
    LOOKUP_ORDER = "lookup_order"
    TRACK_SHIPMENT = "track_shipment"
    CREATE_REFUND = "create_refund"
    DECIDE_HANDOFF = "decide_handoff"
    CREATE_HANDOFF = "create_handoff"
    SYNTHESIZE_ANSWER = "synthesize_answer"
    SYNTHESIZE_MULTI_AGENT = "synthesize_multi_agent"


class ActionStatus(str, Enum):
    PENDING = "pending"
    READY = "ready"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    SKIPPED = "skipped"
    BLOCKED = "blocked"


class AgentRole(str, Enum):
    GENERAL = "general"
    TECHNICAL = "technical"
    BILLING = "billing"
    ESCALATION = "escalation"


class MergeMode(str, Enum):
    SINGLE_AGENT = "single_agent"
    PRIMARY_SUMMARIZE = "primary_summarize"
    PARALLEL_SECTIONS = "parallel_sections"


class FailurePolicy(str, Enum):
    FAIL_FAST = "fail_fast"
    CONTINUE = "continue"
    HANDOFF = "handoff"
    ASK_USER = "ask_user"


_RETRIEVE_ACTIONS = {ActionType.RETRIEVE_POLICY, ActionType.RETRIEVE_FAQ}
_BUSINESS_ACTIONS = {ActionType.LOOKUP_ORDER, ActionType.TRACK_SHIPMENT, ActionType.CREATE_REFUND}
_HANDOFF_ACTIONS = {ActionType.DECIDE_HANDOFF, ActionType.CREATE_HANDOFF}
_SYNTHESIZE_ACTIONS = {ActionType.SYNTHESIZE_ANSWER, ActionType.SYNTHESIZE_MULTI_AGENT}


@dataclass
class ActionItem:
    id: str
    type: ActionType | str
    objective: str
    domain: str
    required_slots: List[str] = field(default_factory=list)
    optional_slots: List[str] = field(default_factory=list)
    depends_on: List[str] = field(default_factory=list)
    can_parallel: bool = False
    tool_name: str = ""
    agent_hint: Optional[AgentRole | str] = None
    input_keys: List[str] = field(default_factory=list)
    output_key: str = ""
    timeout_ms: int = 8000
    retry_limit: int = 1
    failure_policy: FailurePolicy | str = FailurePolicy.FAIL_FAST
    reason: str = ""

    def __post_init__(self) -> None:
        self.type = ActionType(self.type)
        if self.agent_hint:
            self.agent_hint = AgentRole(self.agent_hint)
        self.failure_policy = FailurePolicy(self.failure_policy)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "type": self.type.value,
            "objective": self.objective,
            "domain": self.domain,
            "required_slots": list(self.required_slots),
            "optional_slots": list(self.optional_slots),
            "depends_on": list(self.depends_on),
            "can_parallel": self.can_parallel,
            "tool_name": self.tool_name,
            "agent_hint": self.agent_hint.value if self.agent_hint else "",
            "input_keys": list(self.input_keys),
            "output_key": self.output_key,
            "timeout_ms": self.timeout_ms,
            "retry_limit": self.retry_limit,
            "failure_policy": self.failure_policy.value,
            "reason": self.reason,
        }


@dataclass
class RoutePlan:
    primary_agent: AgentRole | str
    supporting_agents: List[AgentRole | str] = field(default_factory=list)
    merge_mode: MergeMode | str = MergeMode.SINGLE_AGENT
    final_writer: Optional[AgentRole | str] = None
    reason: str = ""

    def __post_init__(self) -> None:
        self.primary_agent = AgentRole(self.primary_agent)
        self.supporting_agents = [AgentRole(agent) for agent in self.supporting_agents]
        self.merge_mode = MergeMode(self.merge_mode)
        if self.final_writer:
            self.final_writer = AgentRole(self.final_writer)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "primary_agent": self.primary_agent.value,
            "supporting_agents": [agent.value for agent in self.supporting_agents],
            "merge_mode": self.merge_mode.value,
            "final_writer": self.final_writer.value if self.final_writer else "",
            "reason": self.reason,
        }


@dataclass
class StopConditions:
    max_actions: int = 5
    max_parallel_groups: int = 2
    max_failures: int = 2
    handoff_on_conflict: bool = True
    ask_user_on_missing_required_slot: bool = True

    def to_dict(self) -> Dict[str, Any]:
        return {
            "max_actions": self.max_actions,
            "max_parallel_groups": self.max_parallel_groups,
            "max_failures": self.max_failures,
            "handoff_on_conflict": self.handoff_on_conflict,
            "ask_user_on_missing_required_slot": self.ask_user_on_missing_required_slot,
        }


@dataclass
class EvidenceItem:
    key: str
    source: str
    value: Any
    confidence: float = 1.0
    stale: bool = False
    conflict_with: List[str] = field(default_factory=list)
    prompt_block: str = ""
    tool_name: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "key": self.key,
            "source": self.source,
            "value": self.value,
            "confidence": self.confidence,
            "stale": self.stale,
            "conflict_with": list(self.conflict_with),
            "prompt_block": self.prompt_block,
            "tool_name": self.tool_name,
        }


@dataclass
class EvidenceStore:
    items: Dict[str, EvidenceItem] = field(default_factory=dict)

    def put(self, item: EvidenceItem) -> None:
        self.items[item.key] = item

    def get(self, key: str) -> Optional[EvidenceItem]:
        return self.items.get(key)

    def has_all(self, keys: Iterable[str]) -> bool:
        return all(key in self.items for key in keys)

    def values(self) -> List[EvidenceItem]:
        return list(self.items.values())

    def to_dict(self) -> Dict[str, Any]:
        return {key: item.to_dict() for key, item in self.items.items()}


@dataclass
class WorkflowPlan:
    complexity: ComplexityLevel | str
    primary_intent: str
    primary_goal: str = ""
    secondary_intents: List[str] = field(default_factory=list)
    actions: List[ActionItem] = field(default_factory=list)
    route_plan: Optional[RoutePlan] = None
    stop_conditions: StopConditions = field(default_factory=StopConditions)
    reason: str = ""
    confidence: float = 0.0
    missing_slots: List[str] = field(default_factory=list)
    clarify_prompt: str = ""

    def __post_init__(self) -> None:
        self.complexity = ComplexityLevel(self.complexity)

    @property
    def need_clarify(self) -> bool:
        return any(action.type == ActionType.CLARIFY_SLOT for action in self.actions)

    @property
    def need_knowledge(self) -> bool:
        return any(action.type in _RETRIEVE_ACTIONS for action in self.actions)

    @property
    def need_order_lookup(self) -> bool:
        return any(action.type == ActionType.LOOKUP_ORDER for action in self.actions)

    @property
    def need_action_tool(self) -> bool:
        return any(action.type in _BUSINESS_ACTIONS for action in self.actions)

    @property
    def need_handoff(self) -> bool:
        if self.route_plan and self.route_plan.primary_agent == AgentRole.ESCALATION:
            return True
        return any(action.type in _HANDOFF_ACTIONS for action in self.actions)

    @property
    def next_action(self) -> str:
        if self.need_clarify:
            return "clarify"
        if self.need_handoff and not (self.need_knowledge or self.need_action_tool):
            return "handoff"
        if self.need_knowledge and self.need_action_tool:
            return "retrieve_act"
        if self.need_action_tool:
            return "act"
        if self.need_knowledge:
            return "retrieve"
        return "respond"

    @property
    def state(self) -> str:
        return self.next_action

    def action_count(self) -> int:
        return sum(1 for action in self.actions if action.type not in _SYNTHESIZE_ACTIONS)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "complexity": self.complexity.value,
            "primary_intent": self.primary_intent,
            "primary_goal": self.primary_goal,
            "secondary_intents": list(self.secondary_intents),
            "state": self.state,
            "next_action": self.next_action,
            "need_clarify": self.need_clarify,
            "need_knowledge": self.need_knowledge,
            "need_order_lookup": self.need_order_lookup,
            "need_action_tool": self.need_action_tool,
            "need_handoff": self.need_handoff,
            "missing_slots": list(self.missing_slots),
            "clarify_prompt": self.clarify_prompt,
            "reason": self.reason,
            "confidence": self.confidence,
            "actions": [action.to_dict() for action in self.actions],
            "route_plan": self.route_plan.to_dict() if self.route_plan else None,
            "stop_conditions": self.stop_conditions.to_dict(),
        }
