from __future__ import annotations

import asyncio
import os
import random
import time
from dataclasses import dataclass
from typing import Awaitable, Callable, Generic, TypeVar


T = TypeVar("T")


@dataclass(frozen=True)
class RetryPolicy:
    """A total time budget shared by every attempt and retry delay."""

    timeout_ms: int
    retry_limit: int = 0
    base_delay_ms: int = 200
    max_delay_ms: int = 2_000

    def __post_init__(self) -> None:
        if self.timeout_ms <= 0:
            raise ValueError("timeout_ms must be positive")
        if self.retry_limit < 0:
            raise ValueError("retry_limit must not be negative")
        if self.base_delay_ms < 0 or self.max_delay_ms < self.base_delay_ms:
            raise ValueError("invalid retry delay configuration")


@dataclass(frozen=True)
class RetryResult(Generic[T]):
    value: T
    attempts: int
    retry_delays_s: tuple[float, ...]


def is_transient_error(error: Exception) -> bool:
    """Classify network, timeout, throttling, and server failures as retryable."""
    if isinstance(error, (asyncio.TimeoutError, OSError)):
        return True
    status_code = getattr(error, "status_code", None)
    response = getattr(error, "response", None)
    if status_code is None and response is not None:
        status_code = getattr(response, "status_code", None)
    return isinstance(status_code, int) and (status_code in {408, 429} or status_code >= 500)


async def run_with_budget(
    operation: Callable[[], Awaitable[T]],
    *,
    policy: RetryPolicy,
    should_retry: Callable[[Exception], bool] = is_transient_error,
    sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
    random_value: Callable[[], float] = random.random,
    clock: Callable[[], float] = time.monotonic,
) -> RetryResult[T]:
    """Run an operation within one deadline, with exponential backoff and full jitter."""
    deadline = clock() + policy.timeout_ms / 1_000
    delays: list[float] = []
    attempts = 0

    while True:
        remaining = deadline - clock()
        if remaining <= 0:
            raise asyncio.TimeoutError("retry budget exhausted before attempt")

        attempts += 1
        try:
            async with asyncio.timeout(remaining):
                value = await operation()
            return RetryResult(value=value, attempts=attempts, retry_delays_s=tuple(delays))
        except Exception as error:
            if attempts > policy.retry_limit or not should_retry(error):
                raise

            cap_s = min(
                policy.max_delay_ms,
                policy.base_delay_ms * (2 ** (attempts - 1)),
            ) / 1_000
            delay_s = cap_s * min(max(random_value(), 0.0), 1.0)
            if delay_s >= deadline - clock():
                raise asyncio.TimeoutError("retry budget exhausted before retry") from error
            delays.append(delay_s)
            await sleep(delay_s)


def llm_retry_policy_from_env() -> RetryPolicy:
    return RetryPolicy(
        timeout_ms=int(os.getenv("LLM_TIMEOUT_MS", "12000")),
        retry_limit=int(os.getenv("LLM_RETRY_LIMIT", "2")),
        base_delay_ms=int(os.getenv("LLM_RETRY_BASE_DELAY_MS", "200")),
        max_delay_ms=int(os.getenv("LLM_RETRY_MAX_DELAY_MS", "2000")),
    )


async def call_llm(operation: Callable[[], Awaitable[T]]) -> T:
    result = await run_with_budget(operation, policy=llm_retry_policy_from_env())
    return result.value
