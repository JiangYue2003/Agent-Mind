import os
from typing import Any, Dict, Optional

import httpx


class HumanHandoffService:
    """转人工工具，负责将会话上下文写入外部客服/工单系统。"""

    def __init__(
        self,
        base_url: Optional[str] = None,
        timeout_s: float = 5.0,
        transport: Optional[httpx.AsyncBaseTransport] = None,
    ):
        self._base_url = (base_url or os.getenv("HANDOFF_BASE_URL", "http://localhost:8000")).rstrip("/")
        self._timeout_s = timeout_s
        self._transport = transport

    async def handoff_handler(self, params: Dict[str, Any], context: Any) -> Dict[str, Any]:
        payload = {
            "user_id": str(params.get("user_id", "") or "").strip(),
            "conv_id": str(params.get("conv_id", "") or "").strip(),
            "latest_message": str(params.get("latest_message", "") or "").strip(),
            "intent": str(params.get("intent", "") or "").strip(),
            "urgency": str(params.get("urgency", "") or "").strip(),
            "reason": str(params.get("reason", "") or "").strip(),
            "summary": str(params.get("summary", "") or "").strip(),
            "recent_messages": list(params.get("recent_messages") or []),
            "user_profile": dict(params.get("user_profile") or {}),
            "order_snapshot": dict(params.get("order_snapshot") or {}),
            "knowledge_context": list(params.get("knowledge_context") or []),
        }
        if not payload["user_id"]:
            raise ValueError("缺少 user_id")
        if not payload["conv_id"]:
            raise ValueError("缺少 conv_id")
        if not payload["latest_message"]:
            raise ValueError("缺少 latest_message")

        async with httpx.AsyncClient(
            base_url=self._base_url,
            timeout=self._timeout_s,
            transport=self._transport,
        ) as client:
            response = await client.post("/mock/external/handoffs", json=payload)
            response.raise_for_status()
            return response.json()
