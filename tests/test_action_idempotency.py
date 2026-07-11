import asyncio
import unittest
from types import SimpleNamespace

from api import main as api_main


class ActionIdempotencyTests(unittest.TestCase):
    def test_operation_key_is_stable_for_the_same_business_command(self):
        first = api_main._make_idempotency_key(
            operation="refund_create",
            command_id="refund-click-1",
            user_id="u1001",
            resource_id="ORD20250701001",
        )
        replay = api_main._make_idempotency_key(
            operation="refund_create",
            command_id="refund-click-1",
            user_id="u1001",
            resource_id="ORD20250701001",
        )
        different_command = api_main._make_idempotency_key(
            operation="refund_create",
            command_id="refund-click-2",
            user_id="u1001",
            resource_id="ORD20250701001",
        )

        self.assertEqual(first, replay)
        self.assertNotEqual(first, different_command)
        self.assertTrue(first.startswith("echomind-refund_create-v1-"))

    def test_refund_action_forwards_the_runtime_idempotency_key_to_the_mcp_tool(self):
        calls = []

        class FakeToolManager:
            async def call(self, name, params, context=None):
                calls.append((name, params, context))
                return SimpleNamespace(success=True, data={"status": "submitted"})

        original = api_main._tool_manager
        api_main._tool_manager = FakeToolManager()
        try:
            result = asyncio.run(api_main._maybe_create_refund(
                api_main.ChatRequest(
                    message="我要退款",
                    user_id="u1001",
                    conv_id="c1001",
                ),
                SimpleNamespace(entities={"order_id": ["ORD20250701001"]}),
                force=True,
                idempotency_key="echomind-refund_create-v1-test",
            ))
        finally:
            api_main._tool_manager = original

        self.assertEqual(result["status"], "submitted")
        self.assertEqual(calls[0][0], "refund_create")
        self.assertEqual(calls[0][1]["idempotency_key"], "echomind-refund_create-v1-test")

    def test_workflow_refund_key_includes_the_target_order(self):
        refund_action = SimpleNamespace(id="a1", type=SimpleNamespace(value="create_refund"))
        plan = SimpleNamespace(actions=[refund_action])
        request = api_main.ChatRequest(
            message="我要退款",
            user_id="u1001",
            conv_id="c1001",
            operation_id="refund-click-1",
        )

        first = api_main._workflow_idempotency_keys(
            plan,
            request,
            "c1001",
            SimpleNamespace(entities={"order_id": ["ORD20250701001"]}),
        )
        second = api_main._workflow_idempotency_keys(
            plan,
            request,
            "c1001",
            SimpleNamespace(entities={"order_id": ["ORD20250701003"]}),
        )

        self.assertNotEqual(first["a1"], second["a1"])

    def test_failed_refund_mcp_response_triggers_the_handoff_failure_policy(self):
        from workflow.action_executor import ActionExecutor
        from workflow.action_models import ActionItem, ActionType, FailurePolicy, WorkflowPlan

        class FailedToolManager:
            async def call(self, name, params, context=None):
                return SimpleNamespace(success=False, data=None, error="after-sales unavailable")

        original = api_main._tool_manager
        api_main._tool_manager = FailedToolManager()
        try:
            plan = WorkflowPlan(
                complexity="single_domain",
                primary_intent="billing",
                actions=[ActionItem(
                    id="a1",
                    type=ActionType.CREATE_REFUND,
                    objective="提交退款",
                    domain="billing",
                    retry_limit=0,
                    failure_policy=FailurePolicy.HANDOFF,
                )],
            )
            result = asyncio.run(ActionExecutor(api_main._build_workflow_tool_registry()).execute(
                plan,
                runtime={
                    "req": api_main.ChatRequest(message="我要退款", user_id="u1001", conv_id="c1001"),
                    "intent_result": SimpleNamespace(entities={"order_id": ["ORD20250701001"]}),
                    "idempotency_keys": {"a1": "refund-command-1"},
                },
            ))
        finally:
            api_main._tool_manager = original

        self.assertTrue(result.handoff_required)
        self.assertEqual(result.action_statuses["a1"], "failed")


if __name__ == "__main__":
    unittest.main()
