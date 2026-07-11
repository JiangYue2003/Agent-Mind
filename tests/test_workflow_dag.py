import asyncio
import unittest

from core.intent_recognizer import IntentCategory
from workflow.action_planner import ActionPlanner
from workflow.slot_manager import SlotManager


class WorkflowDagTests(unittest.TestCase):
    def setUp(self):
        self.slot_manager = SlotManager()
        self.planner = ActionPlanner(slot_manager=self.slot_manager)

    def test_planner_builds_multi_source_action_list_for_policy_plus_order_query(self):
        assessment = self.slot_manager.assess(
            message="请告诉我退款规则，并帮我看订单 ORD20250701001 现在退款到哪了",
            intent=IntentCategory.BILLING,
            entities={"order_id": ["ORD20250701001"]},
        )

        plan = self.planner.plan(
            message="请告诉我退款规则，并帮我看订单 ORD20250701001 现在退款到哪了",
            intent=IntentCategory.BILLING,
            entities={"order_id": ["ORD20250701001"]},
            slot_assessment=assessment,
        )

        action_types = [action.type.value for action in plan.actions]
        self.assertEqual(plan.complexity.value, "multi_source")
        self.assertIn("retrieve_policy", action_types)
        self.assertIn("lookup_order", action_types)
        self.assertEqual(action_types[-1], "synthesize_answer")
        self.assertEqual(plan.route_plan.primary_agent.value, "billing")

    def test_planner_assigns_resilience_policies_by_action_semantics(self):
        from workflow.action_models import FailurePolicy

        retrieve = self.planner._retrieve_policy_action("a1", "knowledge")
        lookup = self.planner._lookup_order_action("a2")
        refund = self.planner._create_refund_action("a3")

        self.assertEqual(retrieve.retry_limit, 2)
        self.assertEqual(retrieve.failure_policy, FailurePolicy.CONTINUE)
        self.assertEqual(lookup.timeout_ms, 6_000)
        self.assertEqual(lookup.failure_policy, FailurePolicy.CONTINUE)
        self.assertEqual(refund.retry_limit, 1)
        self.assertEqual(refund.failure_policy, FailurePolicy.HANDOFF)

    def test_action_executor_runs_ready_nodes_and_writes_evidence_store(self):
        from workflow.action_executor import ActionExecutor
        from workflow.action_models import (
            ActionItem,
            ActionType,
            AgentRole,
            EvidenceItem,
            MergeMode,
            RoutePlan,
            StopConditions,
            WorkflowPlan,
        )
        from workflow.tool_registry import ToolRegistry

        call_order = []
        registry = ToolRegistry()

        async def _retrieve_policy(action, runtime, evidence_store):
            call_order.append(action.id)
            return EvidenceItem(
                key=action.output_key,
                source="knowledge_search",
                value={"content": "退款规则说明"},
                prompt_block="[知识库检索结果]\n退款规则说明",
            )

        async def _lookup_order(action, runtime, evidence_store):
            call_order.append(action.id)
            return EvidenceItem(
                key=action.output_key,
                source="order_lookup",
                value={"order_id": "ORD20250701001", "status": "审核中"},
                prompt_block="[order_lookup]\n- 订单号: ORD20250701001\n- 状态: 审核中",
            )

        async def _synthesize(action, runtime, evidence_store):
            call_order.append(action.id)
            self.assertTrue(evidence_store.has_all(["knowledge.policy", "order.snapshot"]))
            return EvidenceItem(
                key=action.output_key,
                source="workflow",
                value={"status": "ready"},
            )

        registry.register(ActionType.RETRIEVE_POLICY, _retrieve_policy)
        registry.register(ActionType.LOOKUP_ORDER, _lookup_order)
        registry.register(ActionType.SYNTHESIZE_ANSWER, _synthesize)

        plan = WorkflowPlan(
            complexity="multi_source",
            primary_intent="billing",
            actions=[
                ActionItem(
                    id="a1",
                    type=ActionType.RETRIEVE_POLICY,
                    objective="检索退款规则",
                    domain="billing",
                    can_parallel=True,
                    output_key="knowledge.policy",
                ),
                ActionItem(
                    id="a2",
                    type=ActionType.LOOKUP_ORDER,
                    objective="查询订单状态",
                    domain="billing",
                    can_parallel=True,
                    output_key="order.snapshot",
                ),
                ActionItem(
                    id="a3",
                    type=ActionType.SYNTHESIZE_ANSWER,
                    objective="组织最终回答",
                    domain="response",
                    depends_on=["a1", "a2"],
                    input_keys=["knowledge.policy", "order.snapshot"],
                    output_key="final.answer",
                ),
            ],
            route_plan=RoutePlan(
                primary_agent=AgentRole.BILLING,
                merge_mode=MergeMode.SINGLE_AGENT,
            ),
            stop_conditions=StopConditions(max_actions=5, max_parallel_groups=2, max_failures=1),
            reason="复合账单问题需要同时查规则和订单",
            confidence=0.93,
        )

        result = asyncio.run(ActionExecutor(registry).execute(plan, runtime={}))

        self.assertTrue(result.evidence_store.has_all(["knowledge.policy", "order.snapshot", "final.answer"]))
        self.assertEqual(result.action_statuses["a3"], "succeeded")
        self.assertEqual(call_order[-1], "a3")
        joined_context = "\n".join(result.context_blocks)
        self.assertIn("[知识库检索结果]", joined_context)
        self.assertIn("[order_lookup]", joined_context)

    def test_action_executor_enforces_the_action_total_timeout_budget(self):
        from workflow.action_executor import ActionExecutor
        from workflow.action_models import ActionItem, ActionType, WorkflowPlan
        from workflow.tool_registry import ToolRegistry

        registry = ToolRegistry()

        async def _slow_lookup(action, runtime, evidence_store):
            await asyncio.sleep(0.03)

        registry.register(ActionType.LOOKUP_ORDER, _slow_lookup)
        plan = WorkflowPlan(
            complexity="single_domain",
            primary_intent="query",
            actions=[ActionItem(
                id="a1",
                type=ActionType.LOOKUP_ORDER,
                objective="查询订单",
                domain="orders",
                timeout_ms=10,
                retry_limit=0,
            )],
        )

        result = asyncio.run(ActionExecutor(registry).execute(plan, runtime={}))

        self.assertEqual(result.action_statuses["a1"], "failed")
        self.assertIn("a1", result.failed_actions)

    def test_action_executor_does_not_retry_a_write_without_an_idempotency_key(self):
        from workflow.action_executor import ActionExecutor
        from workflow.action_models import ActionItem, ActionType, WorkflowPlan
        from workflow.tool_registry import ToolRegistry

        registry = ToolRegistry()
        calls = 0

        async def _refund_create(action, runtime, evidence_store):
            nonlocal calls
            calls += 1
            raise OSError("after-sales service unavailable")

        registry.register(ActionType.CREATE_REFUND, _refund_create)
        plan = WorkflowPlan(
            complexity="single_domain",
            primary_intent="billing",
            actions=[ActionItem(
                id="a1",
                type=ActionType.CREATE_REFUND,
                objective="提交退款",
                domain="billing",
                timeout_ms=1_000,
                retry_limit=2,
            )],
        )

        result = asyncio.run(ActionExecutor(registry).execute(plan, runtime={}))

        self.assertEqual(result.action_statuses["a1"], "failed")
        self.assertEqual(calls, 1)

    def test_action_executor_fail_fast_policy_blocks_remaining_actions(self):
        from workflow.action_executor import ActionExecutor
        from workflow.action_models import (
            ActionItem,
            ActionType,
            FailurePolicy,
            StopConditions,
            WorkflowPlan,
        )
        from workflow.tool_registry import ToolRegistry

        registry = ToolRegistry()
        follow_up_calls = 0

        async def _failed_lookup(action, runtime, evidence_store):
            raise RuntimeError("order service rejected the request")

        async def _follow_up(action, runtime, evidence_store):
            nonlocal follow_up_calls
            follow_up_calls += 1

        registry.register(ActionType.LOOKUP_ORDER, _failed_lookup)
        registry.register(ActionType.SYNTHESIZE_ANSWER, _follow_up)
        plan = WorkflowPlan(
            complexity="single_domain",
            primary_intent="query",
            actions=[
                ActionItem(
                    id="a1",
                    type=ActionType.LOOKUP_ORDER,
                    objective="查询订单",
                    domain="orders",
                    retry_limit=0,
                    failure_policy=FailurePolicy.FAIL_FAST,
                ),
                ActionItem(
                    id="a2",
                    type=ActionType.SYNTHESIZE_ANSWER,
                    objective="生成答复",
                    domain="response",
                ),
            ],
            stop_conditions=StopConditions(max_failures=2),
        )

        result = asyncio.run(ActionExecutor(registry).execute(plan, runtime={}))

        self.assertTrue(result.degraded)
        self.assertEqual(result.action_statuses["a1"], "failed")
        self.assertEqual(result.action_statuses["a2"], "blocked")
        self.assertEqual(follow_up_calls, 0)

    def test_action_executor_handoff_policy_requests_escalation_and_blocks_remaining_actions(self):
        from workflow.action_executor import ActionExecutor
        from workflow.action_models import ActionItem, ActionType, FailurePolicy, WorkflowPlan
        from workflow.tool_registry import ToolRegistry

        registry = ToolRegistry()

        async def _failed_refund(action, runtime, evidence_store):
            raise OSError("after-sales result is unknown")

        registry.register(ActionType.CREATE_REFUND, _failed_refund)
        plan = WorkflowPlan(
            complexity="single_domain",
            primary_intent="billing",
            actions=[
                ActionItem(
                    id="a1",
                    type=ActionType.CREATE_REFUND,
                    objective="提交退款",
                    domain="billing",
                    retry_limit=0,
                    failure_policy=FailurePolicy.HANDOFF,
                ),
                ActionItem(
                    id="a2",
                    type=ActionType.SYNTHESIZE_ANSWER,
                    objective="生成答复",
                    domain="response",
                ),
            ],
        )

        result = asyncio.run(ActionExecutor(registry).execute(plan, runtime={}))

        self.assertTrue(result.handoff_required)
        self.assertEqual(result.action_statuses["a2"], "blocked")


if __name__ == "__main__":
    unittest.main()
