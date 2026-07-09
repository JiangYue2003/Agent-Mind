from typing import Dict, List, Optional

from core.intent_recognizer import IntentCategory
from workflow.action_models import (
    ActionItem,
    ActionType,
    AgentRole,
    ComplexityLevel,
    MergeMode,
    RoutePlan,
    StopConditions,
    WorkflowPlan,
)
from workflow.slot_manager import SlotAssessment, SlotManager


class ActionPlanner:
    _POLICY_KEYWORDS = ("多久到账", "多久到", "规则", "政策", "流程", "条件", "怎么退", "如何退")
    _BILLING_KEYWORDS = ("退款", "退货", "支付", "扣款", "账单", "发票", "订单")
    _TECHNICAL_KEYWORDS = ("报错", "错误", "崩溃", "500", "401", "无法登录", "登录失败", "error", "crash")
    _SHIPMENT_KEYWORDS = ("物流", "快递", "配送", "发货", "运输")
    _REFUND_ACTION_KEYWORDS = ("我要退款", "申请退款", "帮我退款", "退款一下", "给我退款", "直接退款")

    def __init__(self, slot_manager: Optional[SlotManager] = None):
        self._slot_manager = slot_manager or SlotManager()

    def plan(
        self,
        message: str,
        intent: Optional[IntentCategory],
        entities: Optional[Dict[str, List[str]]],
        slot_assessment: Optional[SlotAssessment] = None,
        intent_confidence: float = 0.0,
        intent_reasoning: str = "",
    ) -> WorkflowPlan:
        assessment = slot_assessment or self._slot_manager.assess(message, intent, entities)
        confidence = self._confidence(intent_confidence, assessment)
        normalized_entities = self._normalize_entities(entities)
        route_plan = self._route_plan(message, intent, assessment)

        if assessment.requires_clarification:
            actions = [
                ActionItem(
                    id="a1",
                    type=ActionType.CLARIFY_SLOT,
                    objective="向用户追问缺失槽位",
                    domain="conversation",
                    required_slots=list(assessment.missing_required_slots),
                    output_key="clarify.prompt",
                    reason=assessment.reason or "缺少关键槽位，需要先追问",
                )
            ]
            return WorkflowPlan(
                complexity=ComplexityLevel.SINGLE_DOMAIN,
                primary_intent=intent.value if intent else "other",
                primary_goal=assessment.primary_goal,
                actions=actions,
                route_plan=route_plan,
                stop_conditions=StopConditions(max_actions=1, max_parallel_groups=1, max_failures=0),
                reason=assessment.reason or "缺少关键槽位，需要先追问",
                confidence=confidence,
                missing_slots=list(assessment.missing_required_slots),
                clarify_prompt=assessment.clarify_question,
            )

        if assessment.should_handoff:
            actions = []
            if normalized_entities.get("order_id"):
                actions.append(self._lookup_order_action("a1"))
            handoff_depends = [actions[-1].id] if actions else []
            actions.append(
                ActionItem(
                    id=f"a{len(actions) + 1}",
                    type=ActionType.CREATE_HANDOFF,
                    objective="创建人工转接记录",
                    domain="escalation",
                    depends_on=handoff_depends,
                    tool_name="human_handoff",
                    input_keys=["order.snapshot"] if handoff_depends else [],
                    output_key="handoff.result",
                    reason=assessment.reason or "用户明确要求转人工",
                )
            )
            actions.append(
                self._synthesize_action(
                    action_id=f"a{len(actions) + 1}",
                    depends_on=[actions[-1].id],
                    multi_agent=False,
                )
            )
            return WorkflowPlan(
                complexity=ComplexityLevel.EXCEPTION,
                primary_intent=intent.value if intent else "other",
                primary_goal=assessment.primary_goal,
                actions=actions,
                route_plan=RoutePlan(
                    primary_agent=AgentRole.ESCALATION,
                    merge_mode=MergeMode.SINGLE_AGENT,
                    reason=assessment.reason or "升级到人工客服处理",
                ),
                reason=assessment.reason or "用户明确要求转人工",
                confidence=max(confidence, 0.9),
            )

        supports_multi_agent = bool(route_plan and route_plan.supporting_agents)
        needs_policy = self._needs_policy_lookup(message, assessment)
        needs_order = assessment.should_lookup_order and bool(normalized_entities.get("order_id"))
        needs_shipment = self._needs_shipment_tracking(message, assessment, normalized_entities)
        needs_refund_action = self._needs_refund_creation(message, assessment, normalized_entities)
        actions: List[ActionItem] = []

        if needs_policy and needs_refund_action:
            actions.extend([
                self._retrieve_policy_action("a1", assessment.primary_goal),
                self._create_refund_action("a2"),
                self._synthesize_action("a3", depends_on=["a1", "a2"], multi_agent=supports_multi_agent),
            ])
            complexity = ComplexityLevel.CROSS_DOMAIN if supports_multi_agent else ComplexityLevel.MULTI_SOURCE
            return WorkflowPlan(
                complexity=complexity,
                primary_intent=intent.value if intent else "other",
                primary_goal=assessment.primary_goal,
                secondary_intents=self._secondary_intents(route_plan),
                actions=actions,
                route_plan=route_plan,
                reason=assessment.reason or "用户既在咨询规则，也希望直接提交退款申请",
                confidence=max(confidence, 0.9),
            )

        if needs_policy and needs_order:
            actions.extend([
                self._retrieve_policy_action("a1", assessment.primary_goal),
                self._lookup_order_action("a2"),
                self._synthesize_action("a3", depends_on=["a1", "a2"], multi_agent=supports_multi_agent),
            ])
            complexity = ComplexityLevel.CROSS_DOMAIN if supports_multi_agent else ComplexityLevel.MULTI_SOURCE
            return WorkflowPlan(
                complexity=complexity,
                primary_intent=intent.value if intent else "other",
                primary_goal=assessment.primary_goal,
                secondary_intents=self._secondary_intents(route_plan),
                actions=actions,
                route_plan=route_plan,
                reason=assessment.reason or "复合问题需要同时查规则和订单实时状态",
                confidence=max(confidence, 0.9),
            )

        if needs_refund_action:
            actions.extend([
                self._create_refund_action("a1"),
                self._synthesize_action("a2", depends_on=["a1"], multi_agent=supports_multi_agent),
            ])
            complexity = ComplexityLevel.CROSS_DOMAIN if supports_multi_agent else ComplexityLevel.SINGLE_DOMAIN
            return WorkflowPlan(
                complexity=complexity,
                primary_intent=intent.value if intent else "other",
                primary_goal=assessment.primary_goal,
                secondary_intents=self._secondary_intents(route_plan),
                actions=actions,
                route_plan=route_plan,
                reason=assessment.reason or "用户明确希望直接发起退款申请",
                confidence=max(confidence, 0.88),
            )

        if needs_shipment:
            actions.extend([
                self._track_shipment_action("a1"),
                self._synthesize_action("a2", depends_on=["a1"], multi_agent=supports_multi_agent),
            ])
            complexity = ComplexityLevel.CROSS_DOMAIN if supports_multi_agent else ComplexityLevel.SINGLE_DOMAIN
            return WorkflowPlan(
                complexity=complexity,
                primary_intent=intent.value if intent else "other",
                primary_goal=assessment.primary_goal,
                secondary_intents=self._secondary_intents(route_plan),
                actions=actions,
                route_plan=route_plan,
                reason=assessment.reason or "用户在查询物流轨迹，需要实时物流信息",
                confidence=max(confidence, 0.86),
            )

        if needs_order:
            actions.extend([
                self._lookup_order_action("a1"),
                self._synthesize_action("a2", depends_on=["a1"], multi_agent=supports_multi_agent),
            ])
            complexity = ComplexityLevel.CROSS_DOMAIN if supports_multi_agent else ComplexityLevel.SINGLE_DOMAIN
            return WorkflowPlan(
                complexity=complexity,
                primary_intent=intent.value if intent else "other",
                primary_goal=assessment.primary_goal,
                secondary_intents=self._secondary_intents(route_plan),
                actions=actions,
                route_plan=route_plan,
                reason=assessment.reason or "问题依赖订单实时状态",
                confidence=max(confidence, 0.85),
            )

        if needs_policy:
            actions.extend([
                self._retrieve_policy_action("a1", assessment.primary_goal),
                self._synthesize_action("a2", depends_on=["a1"], multi_agent=supports_multi_agent),
            ])
            complexity = ComplexityLevel.CROSS_DOMAIN if supports_multi_agent else ComplexityLevel.SINGLE_DOMAIN
            return WorkflowPlan(
                complexity=complexity,
                primary_intent=intent.value if intent else "other",
                primary_goal=assessment.primary_goal,
                secondary_intents=self._secondary_intents(route_plan),
                actions=actions,
                route_plan=route_plan,
                reason=assessment.reason or intent_reasoning or "问题更适合检索知识库后回答",
                confidence=max(confidence, 0.8),
            )

        actions.append(self._synthesize_action("a1", depends_on=[], multi_agent=supports_multi_agent))
        return WorkflowPlan(
            complexity=ComplexityLevel.CROSS_DOMAIN if supports_multi_agent else ComplexityLevel.SIMPLE,
            primary_intent=intent.value if intent else "other",
            primary_goal=assessment.primary_goal,
            secondary_intents=self._secondary_intents(route_plan),
            actions=actions,
            route_plan=route_plan,
            reason=assessment.reason or intent_reasoning or "直接回复即可",
            confidence=confidence,
        )

    def _confidence(self, intent_confidence: float, assessment: SlotAssessment) -> float:
        base = intent_confidence if intent_confidence > 0 else 0.75
        if assessment.requires_clarification or assessment.should_handoff:
            base = max(base, 0.88)
        if assessment.should_lookup_order:
            base = max(base, 0.85)
        return round(min(base, 0.99), 2)

    def _normalize_entities(self, entities: Optional[Dict[str, List[str]]]) -> Dict[str, List[str]]:
        normalized: Dict[str, List[str]] = {}
        for key, values in (entities or {}).items():
            items = values if isinstance(values, list) else [values]
            cleaned = [str(item).strip() for item in items if str(item).strip()]
            normalized[str(key)] = cleaned
        normalized.setdefault("order_id", [])
        return normalized

    def _needs_policy_lookup(self, message: str, assessment: SlotAssessment) -> bool:
        lowered = (message or "").lower()
        if assessment.primary_goal in {"refund_policy", "faq"}:
            return True
        return any(keyword in lowered for keyword in self._POLICY_KEYWORDS) and any(
            keyword in lowered for keyword in self._BILLING_KEYWORDS
        )

    def _needs_shipment_tracking(
        self,
        message: str,
        assessment: SlotAssessment,
        entities: Dict[str, List[str]],
    ) -> bool:
        if assessment.primary_goal != "order_status":
            return False
        if not entities.get("order_id"):
            return False
        lowered = (message or "").lower()
        return any(keyword in lowered for keyword in self._SHIPMENT_KEYWORDS)

    def _needs_refund_creation(
        self,
        message: str,
        assessment: SlotAssessment,
        entities: Dict[str, List[str]],
    ) -> bool:
        if assessment.primary_goal != "refund_consult":
            return False
        if not entities.get("order_id"):
            return False
        lowered = (message or "").lower()
        if any(keyword in lowered for keyword in ("退款到哪", "退款进度", "退款状态", "查退款", "查询退款")):
            return False
        return any(keyword in lowered for keyword in self._REFUND_ACTION_KEYWORDS)

    def _route_plan(
        self,
        message: str,
        intent: Optional[IntentCategory],
        assessment: SlotAssessment,
    ) -> RoutePlan:
        lowered = (message or "").lower()
        billing_signal = intent in {IntentCategory.BILLING, IntentCategory.ACCOUNT} or any(
            keyword in lowered for keyword in self._BILLING_KEYWORDS
        )
        technical_signal = intent == IntentCategory.TECHNICAL or any(
            keyword in lowered for keyword in self._TECHNICAL_KEYWORDS
        )

        if assessment.should_handoff:
            return RoutePlan(
                primary_agent=AgentRole.ESCALATION,
                merge_mode=MergeMode.SINGLE_AGENT,
                reason=assessment.reason or "人工升级场景",
            )

        if billing_signal and technical_signal:
            return RoutePlan(
                primary_agent=AgentRole.BILLING,
                supporting_agents=[AgentRole.TECHNICAL],
                merge_mode=MergeMode.PRIMARY_SUMMARIZE,
                final_writer=AgentRole.BILLING,
                reason="同时涉及账单/售后和技术问题，由账单 Agent 主答，技术 Agent 提供辅助意见",
            )

        if technical_signal:
            return RoutePlan(primary_agent=AgentRole.TECHNICAL, reason="主问题为技术问题")
        if billing_signal:
            return RoutePlan(primary_agent=AgentRole.BILLING, reason="主问题为账单或订单售后问题")
        return RoutePlan(primary_agent=AgentRole.GENERAL, reason="通用咨询场景")

    def _secondary_intents(self, route_plan: RoutePlan) -> List[str]:
        return [agent.value for agent in route_plan.supporting_agents] if route_plan else []

    def _retrieve_policy_action(self, action_id: str, primary_goal: str) -> ActionItem:
        action_type = ActionType.RETRIEVE_POLICY if primary_goal != "faq" else ActionType.RETRIEVE_FAQ
        return ActionItem(
            id=action_id,
            type=action_type,
            objective="检索客服知识库中的规则说明",
            domain="knowledge",
            can_parallel=True,
            tool_name="knowledge_search",
            output_key="knowledge.policy",
        )

    def _lookup_order_action(self, action_id: str) -> ActionItem:
        return ActionItem(
            id=action_id,
            type=ActionType.LOOKUP_ORDER,
            objective="查询订单实时状态",
            domain="orders",
            required_slots=["order_id"],
            can_parallel=True,
            tool_name="order_lookup",
            output_key="order.snapshot",
        )

    def _track_shipment_action(self, action_id: str) -> ActionItem:
        return ActionItem(
            id=action_id,
            type=ActionType.TRACK_SHIPMENT,
            objective="查询物流实时轨迹",
            domain="shipping",
            required_slots=["order_id"],
            can_parallel=True,
            tool_name="shipment_track",
            output_key="shipment.snapshot",
        )

    def _create_refund_action(self, action_id: str) -> ActionItem:
        return ActionItem(
            id=action_id,
            type=ActionType.CREATE_REFUND,
            objective="提交退款申请",
            domain="billing",
            required_slots=["order_id"],
            tool_name="refund_create",
            output_key="refund.result",
        )

    def _synthesize_action(self, action_id: str, depends_on: List[str], multi_agent: bool) -> ActionItem:
        return ActionItem(
            id=action_id,
            type=ActionType.SYNTHESIZE_MULTI_AGENT if multi_agent else ActionType.SYNTHESIZE_ANSWER,
            objective="整理证据并生成最终答复",
            domain="response",
            depends_on=list(depends_on),
            input_keys=["knowledge.policy", "order.snapshot"] if len(depends_on) > 1 else [],
            output_key="final.answer",
        )
