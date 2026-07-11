import json
import logging
import re
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional

from anthropic import AsyncAnthropic

from core.resilience import call_llm
from core.intent_recognizer import IntentCategory
from workflow.tool_schema import required_slots_for_tools

logger = logging.getLogger(__name__)


class DecisionMode(str, Enum):
    KNOWLEDGE = "knowledge"
    LIVE_RECORD = "live_record"
    ACTION = "action"


@dataclass
class WorkflowDecision:
    mode: DecisionMode
    tools: List[str]
    confidence: float
    reason: str
    specific_order: bool = False
    order_id: str = ""
    used_fallback: bool = False

    @property
    def required_slots(self) -> List[str]:
        return required_slots_for_tools(self.tools)

    @property
    def requires_order_id(self) -> bool:
        return "order_id" in self.required_slots

    @property
    def should_clarify(self) -> bool:
        return self.requires_order_id and not self.order_id and "knowledge_search" not in self.tools

    def to_dict(self) -> Dict[str, Any]:
        return {
            "goal": self.mode.value,
            "mode": self.mode.value,
            "tools": list(self.tools),
            "specific_order": self.specific_order,
            "required_slots": self.required_slots,
            "confidence": self.confidence,
            "reason": self.reason,
            "order_id_resolved": bool(self.order_id),
            "requires_order_id": self.requires_order_id,
            "should_clarify": self.should_clarify,
            "used_fallback": self.used_fallback,
        }


class WorkflowIntentDecider:
    """Select the minimum evidence needed to answer a customer-service request."""

    _ALLOWED_TOOLS = {"knowledge_search", "order_lookup", "shipment_track", "refund_create"}
    _ORDER_ID_PATTERN = re.compile(r"\bORD[0-9A-Za-z_-]+\b", re.IGNORECASE)
    _MIN_CONFIDENCE = 0.6

    def __init__(
        self,
        api_key: str,
        base_url: Optional[str] = None,
        model: str = "claude-3-5-sonnet-20241022",
        client: Optional[Any] = None,
    ):
        if client is not None:
            self.client = client
        else:
            kwargs: Dict[str, Any] = {"api_key": api_key}
            if base_url:
                kwargs["base_url"] = base_url
            self.client = AsyncAnthropic(**kwargs)
        self.model = model

    async def decide(
        self,
        message: str,
        intent: Optional[IntentCategory],
        entities: Optional[Dict[str, List[str]]],
        history: Optional[List[Dict[str, str]]],
    ) -> WorkflowDecision:
        order_id = self._resolve_order_id(entities, history)
        prompt = self._build_prompt(message, intent, order_id, history)
        try:
            response = await call_llm(
                lambda: self.client.messages.create(
                    model=self.model,
                    max_tokens=320,
                    temperature=0.0,
                    messages=[{"role": "user", "content": prompt}],
                )
            )
            raw = response.content[0].text
            payload = self._parse_json(raw)
            return self._from_payload(payload, order_id)
        except Exception as ex:
            logger.warning("工作流语义决策失败，降级知识检索: %s", ex)
            return self._knowledge_fallback("决策服务不可用，优先检索知识库", order_id)

    def _build_prompt(
        self,
        message: str,
        intent: Optional[IntentCategory],
        order_id: str,
        history: Optional[List[Dict[str, str]]],
    ) -> str:
        recent_history = "\n".join(
            f"{item.get('role', 'user')}: {item.get('content', '')}"
            for item in (history or [])[-5:]
        ) or "无"
        return f"""你是客服工作流决策器。判断回答当前问题所需的最小证据，不能只根据关键词猜测工具。

粗粒度意图: {getattr(intent, 'value', 'other')}
当前可用订单号: {order_id or '无'}
最近对话:
{recent_history}
当前问题: {message}

可用工具:
- knowledge_search: 查询通用规则、资格、时效、正常性、处理条件、处理阈值和操作流程；不需要订单号。
- order_lookup: 查询某一笔订单的实时状态；需要订单号。
- shipment_track: 查询某一笔订单的实时物流轨迹；需要订单号。
- refund_create: 为某一笔订单提交退款申请；需要订单号。

决策规则:
- goal 表示用户想解决的问题类型，不表示订单号是否已经给出。缺订单号时，仍必须保留用户需要的实时查询或执行工具，后续由槽位机制追问。
- 规则、资格、时效、正常性、处理阈值和流程选择 goal=knowledge、tools=["knowledge_search"]。
- 用户想查询自己订单的当前状态或当前物流，选择 goal=live_record 和对应实时工具；即使未给订单号也不能改成 knowledge。
- 用户想提交退款或要求系统执行操作，选择 goal=action 和对应执行工具；即使未给订单号也不能改成 knowledge。
- 只有用户同时明确询问通用规则和个人订单时，才同时选择 knowledge_search 和对应实时工具。

示例:
- “现货付款后多久发货？” -> goal=knowledge, tools=["knowledge_search"]
- “我想退款” -> goal=action, tools=["refund_create"]
- “我的订单还没到” -> goal=live_record, tools=["order_lookup"]
- “帮我查一下物流进度” -> goal=live_record, tools=["shipment_track"]

仅返回 JSON:
{{"goal":"knowledge|live_record|action","tools":["knowledge_search|order_lookup|shipment_track|refund_create"],"confidence":0到1,"reason":"简短中文理由"}}"""

    def _from_payload(self, payload: Dict[str, Any], order_id: str) -> WorkflowDecision:
        try:
            mode = DecisionMode(str(payload.get("goal", payload.get("mode"))))
            tools = [str(tool) for tool in payload["tools"]]
            confidence = float(payload["confidence"])
        except (KeyError, TypeError, ValueError) as ex:
            raise ValueError("决策结果字段不完整") from ex

        if not tools or any(tool not in self._ALLOWED_TOOLS for tool in tools):
            raise ValueError("决策结果包含未知工具")
        if not 0.0 <= confidence <= 1.0:
            raise ValueError("决策置信度非法")
        if not self._tools_match_mode(mode, tools):
            return self._knowledge_fallback("决策模式与工具能力不一致", order_id)

        if confidence < self._MIN_CONFIDENCE:
            return self._knowledge_fallback("决策置信度不足", order_id)

        return WorkflowDecision(
            mode=mode,
            tools=list(dict.fromkeys(tools)),
            confidence=confidence,
            reason=str(payload.get("reason", ""))[:200],
            specific_order=bool(payload.get("specific_order", mode != DecisionMode.KNOWLEDGE)),
            order_id=order_id,
        )

    def _knowledge_fallback(self, reason: str, order_id: str) -> WorkflowDecision:
        return WorkflowDecision(
            mode=DecisionMode.KNOWLEDGE,
            tools=["knowledge_search"],
            specific_order=False,
            confidence=0.0,
            reason=reason,
            order_id=order_id,
            used_fallback=True,
        )

    @staticmethod
    def _tools_match_mode(mode: DecisionMode, tools: List[str]) -> bool:
        tool_set = set(tools)
        if mode == DecisionMode.KNOWLEDGE:
            return tool_set == {"knowledge_search"}
        if mode == DecisionMode.LIVE_RECORD:
            return tool_set.issubset({"knowledge_search", "order_lookup", "shipment_track"}) and bool(
                tool_set & {"order_lookup", "shipment_track"}
            )
        return tool_set.issubset({"knowledge_search", "refund_create"}) and "refund_create" in tool_set

    def _resolve_order_id(
        self,
        entities: Optional[Dict[str, List[str]]],
        history: Optional[List[Dict[str, str]]],
    ) -> str:
        current_ids = self._unique_order_ids((entities or {}).get("order_id") or [])
        if len(current_ids) == 1:
            return current_ids[0]
        if current_ids:
            return ""

        history_ids: List[str] = []
        for item in (history or [])[-5:]:
            history_ids.extend(self._ORDER_ID_PATTERN.findall(str(item.get("content", ""))))
        unique_history_ids = self._unique_order_ids(history_ids)
        return unique_history_ids[0] if len(unique_history_ids) == 1 else ""

    def _unique_order_ids(self, values: List[Any]) -> List[str]:
        unique: List[str] = []
        for value in values:
            text = str(value).strip().upper()
            if self._ORDER_ID_PATTERN.fullmatch(text) and text not in unique:
                unique.append(text)
        return unique

    @staticmethod
    def _parse_json(raw: str) -> Dict[str, Any]:
        start, end = raw.find("{"), raw.rfind("}") + 1
        if start < 0 or end <= start:
            raise ValueError("决策结果不是 JSON")
        payload = json.loads(raw[start:end])
        if not isinstance(payload, dict):
            raise ValueError("决策结果不是对象")
        return payload
