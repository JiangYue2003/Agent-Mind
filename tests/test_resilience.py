import asyncio
import unittest

from core.resilience import RetryPolicy, run_with_budget


class ResilienceTests(unittest.TestCase):
    def test_retries_transient_error_with_exponential_full_jitter(self):
        attempts = 0
        delays = []

        async def operation():
            nonlocal attempts
            attempts += 1
            if attempts < 3:
                raise OSError("temporary network failure")
            return "ok"

        async def sleep(delay):
            delays.append(delay)

        result = asyncio.run(run_with_budget(
            operation,
            policy=RetryPolicy(
                timeout_ms=1_000,
                retry_limit=2,
                base_delay_ms=100,
                max_delay_ms=1_000,
            ),
            should_retry=lambda error: isinstance(error, OSError),
            sleep=sleep,
            random_value=lambda: 0.5,
        ))

        self.assertEqual(result.value, "ok")
        self.assertEqual(result.attempts, 3)
        self.assertEqual(delays, [0.05, 0.1])


if __name__ == "__main__":
    unittest.main()
