import os
from typing import Any, Dict, Optional

import httpx


class OrderLookupService:
    """订单查询工具，默认通过 HTTP 调用外部 OMS。"""

    def __init__(
        self,
        base_url: Optional[str] = None,
        timeout_s: float = 5.0,
        transport: Optional[httpx.AsyncBaseTransport] = None,
    ):
        self._base_url = (base_url or os.getenv("ORDER_LOOKUP_BASE_URL", "http://localhost:8000")).rstrip("/")
        self._timeout_s = timeout_s
        self._transport = transport

    async def lookup_handler(self, params: Dict[str, Any], context: Any) -> Dict[str, Any]:
        user_id = str(params.get("user_id", "") or "").strip()
        order_id = str(params.get("order_id", "") or "").strip()
        if not user_id:
            raise ValueError("缺少 user_id")
        if not order_id:
            raise ValueError("缺少 order_id")

        async with httpx.AsyncClient(
            base_url=self._base_url,
            timeout=self._timeout_s,
            transport=self._transport,
        ) as client:
            response = await client.get(
                f"/mock/external/orders/{order_id}",
                params={"user_id": user_id},
            )
            response.raise_for_status()
            return response.json()
