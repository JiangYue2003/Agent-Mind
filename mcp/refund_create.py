import os
from typing import Any, Dict, Optional

import httpx

from mcp.http_utils import raise_for_protected_resource


class RefundCreateService:
    """退款申请工具，默认通过 HTTP 调用外部售后/订单系统。"""

    def __init__(
        self,
        base_url: Optional[str] = None,
        timeout_s: float = 5.0,
        transport: Optional[httpx.AsyncBaseTransport] = None,
    ):
        self._base_url = (base_url or os.getenv("REFUND_CREATE_BASE_URL", "http://localhost:8000")).rstrip("/")
        self._timeout_s = timeout_s
        self._transport = transport

    async def create_handler(self, params: Dict[str, Any], context: Any) -> Dict[str, Any]:
        payload = {
            "user_id": str(params.get("user_id", "") or "").strip(),
            "order_id": str(params.get("order_id", "") or "").strip(),
            "reason": str(params.get("reason", "") or "").strip(),
        }
        if not payload["user_id"]:
            raise ValueError("缺少 user_id")
        if not payload["order_id"]:
            raise ValueError("缺少 order_id")

        async with httpx.AsyncClient(
            base_url=self._base_url,
            timeout=self._timeout_s,
            transport=self._transport,
        ) as client:
            response = await client.post("/mock/external/refunds", json=payload)
            raise_for_protected_resource(response, "退款申请")
            return response.json()
