from __future__ import annotations

import httpx


def raise_for_protected_resource(response: httpx.Response, resource_name: str) -> None:
    if response.status_code in {403, 404}:
        raise httpx.HTTPStatusError(
            f"{resource_name}不存在或无权访问",
            request=response.request,
            response=response,
        )
    response.raise_for_status()
