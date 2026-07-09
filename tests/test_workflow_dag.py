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


if __name__ == "__main__":
    unittest.main()
