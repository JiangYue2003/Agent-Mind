from dataclasses import dataclass, field
from typing import Dict, List, Optional

from core.intent_recognizer import IntentCategory


@dataclass
class SlotAssessment:
    primary_goal: str
    required_slots: List[str] = field(default_factory=list)
    optional_slots: List[str] = field(default_factory=list)
    missing_required_slots: List[str] = field(default_factory=list)
    missing_optional_slots: List[str] = field(default_factory=list)
    clarify_question: str = ""
    reason: str = ""
    should_lookup_order: bool = False
    should_handoff: bool = False

    @property
    def requires_clarification(self) -> bool:
        return bool(self.missing_required_slots)

    def to_dict(self) -> Dict[str, object]:
        return {
            "primary_goal": self.primary_goal,
            "required_slots": list(self.required_slots),
            "optional_slots": list(self.optional_slots),
            "missing_required_slots": list(self.missing_required_slots),
            "missing_optional_slots": list(self.missing_optional_slots),
            "clarify_question": self.clarify_question,
            "reason": self.reason,
            "should_lookup_order": self.should_lookup_order,
            "should_handoff": self.should_handoff,
        }


class SlotManager:
    _HANDOFF_KEYWORDS = ("转人工", "人工客服", "人工处理", "找人工", "客服专员", "投诉专员")
    _ORDER_STATUS_KEYWORDS = ("订单", "物流", "发货", "配送", "快递", "到哪", "进度", "状态", "查询订单")
    _REFUND_POLICY_KEYWORDS = ("多久到账", "多久到", "几天到账", "什么时候到账", "规则", "政策", "流程")
    _REFUND_REQUEST_KEYWORDS = ("我要退款", "申请退款", "帮我退款", "退款一下", "给我退款")

    def assess(
        self,
        message: str,
        intent: Optional[IntentCategory],
        entities: Optional[Dict[str, List[str]]],
    ) -> SlotAssessment:
        msg = (message or "").strip()
        lowered = msg.lower()
        normalized_entities = self._normalize_entities(entities)
        order_ids = normalized_entities.get("order_id", [])

        if self._is_handoff_request(lowered, intent):
            missing_optional_slots = []
            if not order_ids:
                missing_optional_slots.append("order_id")
            if len(msg) <= 10:
                missing_optional_slots.append("problem_summary")
            return SlotAssessment(
                primary_goal="handoff_request",
                optional_slots=["order_id", "problem_summary"],
                missing_optional_slots=missing_optional_slots,
                reason="用户明确要求转人工处理",
                should_handoff=True,
            )

        if self._is_refund_policy_question(lowered):
            return SlotAssessment(
                primary_goal="refund_policy",
                reason="用户在咨询退款规则或到账时效，优先走知识检索",
            )

        if self._is_order_status_request(lowered, order_ids):
            missing_required = [] if order_ids else ["order_id"]
            return SlotAssessment(
                primary_goal="order_status",
                required_slots=["order_id"],
                missing_required_slots=missing_required,
                clarify_question=self._clarify_question("order_id") if missing_required else "",
                reason="用户在查询订单或退款进度，需要定位具体订单",
                should_lookup_order=bool(order_ids),
            )

        if self._is_refund_consult(lowered, intent):
            missing_required = [] if order_ids else ["order_id"]
            missing_optional = [] if self._extract_refund_reason(msg) else ["refund_reason"]
            return SlotAssessment(
                primary_goal="refund_consult",
                required_slots=["order_id"],
                optional_slots=["refund_reason"],
                missing_required_slots=missing_required,
                missing_optional_slots=missing_optional,
                clarify_question=self._clarify_question("order_id") if missing_required else "",
                reason="用户想处理退款，但需要先定位订单",
                should_lookup_order=bool(order_ids),
            )

        if self._looks_like_business_faq(lowered, intent):
            return SlotAssessment(
                primary_goal="faq",
                reason="问题更像规则说明或常见问答，优先检索知识库",
            )

        return SlotAssessment(
            primary_goal="direct_response",
            reason="问题不依赖知识检索或业务工具，可直接回复",
        )

    def _normalize_entities(self, entities: Optional[Dict[str, List[str]]]) -> Dict[str, List[str]]:
        normalized: Dict[str, List[str]] = {}
        for key, values in (entities or {}).items():
            items = values if isinstance(values, list) else [values]
            cleaned = [str(item).strip() for item in items if str(item).strip()]
            normalized[str(key)] = cleaned
        normalized.setdefault("order_id", [])
        return normalized

    def _is_handoff_request(self, message: str, intent: Optional[IntentCategory]) -> bool:
        if intent == IntentCategory.ESCALATION:
            return True
        return any(keyword in message for keyword in self._HANDOFF_KEYWORDS)

    def _is_refund_policy_question(self, message: str) -> bool:
        if "退款" not in message:
            return False
        if any(keyword in message for keyword in self._REFUND_REQUEST_KEYWORDS):
            return False
        if "订单" in message and ("查" in message or "查询" in message or "到哪" in message):
            return False
        return any(keyword in message for keyword in self._REFUND_POLICY_KEYWORDS)

    def _is_order_status_request(self, message: str, order_ids: List[str]) -> bool:
        refund_progress_keywords = ("查", "查询", "进度", "状态", "到哪", "现在", "结果")
        if order_ids and "退款" in message and any(keyword in message for keyword in refund_progress_keywords):
            return True
        if any(keyword in message for keyword in self._REFUND_REQUEST_KEYWORDS):
            return False
        if order_ids and any(keyword in message for keyword in self._ORDER_STATUS_KEYWORDS):
            return True
        return any(keyword in message for keyword in self._ORDER_STATUS_KEYWORDS)

    def _is_refund_consult(self, message: str, intent: Optional[IntentCategory]) -> bool:
        if "退款" not in message and intent != IntentCategory.BILLING:
            return False
        if self._is_refund_policy_question(message):
            return False
        if any(keyword in message for keyword in self._REFUND_REQUEST_KEYWORDS):
            return True
        action_keywords = ("帮我查", "查询", "订单", "申请", "进度", "处理")
        return "退款" in message and any(keyword in message for keyword in action_keywords)

    def _looks_like_business_faq(self, message: str, intent: Optional[IntentCategory]) -> bool:
        business_keywords = (
            "退款", "订单", "物流", "配送", "发票", "扣款", "支付", "账单",
            "订阅", "登录", "报错", "错误", "崩溃", "会员", "积分", "账户", "密码", "地址",
        )
        if any(keyword in message for keyword in business_keywords):
            return True
        return intent in {
            IntentCategory.BILLING,
            IntentCategory.TECHNICAL,
            IntentCategory.ACCOUNT,
            IntentCategory.REQUEST,
        }

    def _clarify_question(self, slot_name: str) -> str:
        if slot_name == "order_id":
            return "为了帮你准确查询订单或退款进度，请提供订单号。"
        if slot_name == "refund_reason":
            return "请补充一下退款原因，我再结合规则帮你判断。"
        return "为了继续帮你处理，请补充关键信息。"

    def _extract_refund_reason(self, message: str) -> str:
        markers = ("因为", "原因是", "不想要了", "买错", "重复下单", "商品有问题")
        for marker in markers:
            if marker in message:
                return marker
        return ""
