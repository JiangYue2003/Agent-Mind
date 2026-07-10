import unittest

from core.intent_recognizer import IntentCategory
from workflow.action_planner import ActionPlanner
from workflow.intent_decider import DecisionMode, WorkflowDecision
from workflow.slot_manager import SlotManager
from workflow.state_machine import WorkflowState, WorkflowStateMachine


class WorkflowPhase1Tests(unittest.TestCase):
    def setUp(self):
        self.slot_manager = SlotManager()
        self.planner = ActionPlanner(slot_manager=self.slot_manager)
        self.state_machine = WorkflowStateMachine()

    def test_slot_manager_marks_missing_order_id_for_refund_request(self):
        assessment = self.slot_manager.assess(
            message="我要退款",
            intent=IntentCategory.BILLING,
            entities={"order_id": []},
        )

        self.assertEqual(assessment.primary_goal, "refund_consult")
        self.assertEqual(assessment.missing_required_slots, ["order_id"])
        self.assertIn("订单号", assessment.clarify_question)

    def test_action_planner_prefers_knowledge_for_refund_eta_faq(self):
        assessment = self.slot_manager.assess(
            message="退款多久到账",
            intent=IntentCategory.BILLING,
            entities={"order_id": []},
        )

        plan = self.planner.plan(
            message="退款多久到账",
            intent=IntentCategory.BILLING,
            entities={"order_id": []},
            slot_assessment=assessment,
        )

        self.assertEqual(plan.primary_goal, "refund_policy")
        self.assertEqual(plan.next_action, "retrieve")
        self.assertTrue(plan.need_knowledge)
        self.assertFalse(plan.need_order_lookup)

    def test_semantic_knowledge_decision_does_not_require_order_id_for_delivery_rules(self):
        decision = WorkflowDecision(
            mode=DecisionMode.KNOWLEDGE,
            tools=["knowledge_search"],
            specific_order=False,
            confidence=0.94,
            reason="用户在询问物流异常是否属于正常现象",
        )
        assessment = self.slot_manager.assess(
            message="有单号但没有物流记录正常吗？",
            intent=IntentCategory.QUERY,
            entities={"order_id": []},
            decision=decision,
        )

        plan = self.planner.plan(
            message="有单号但没有物流记录正常吗？",
            intent=IntentCategory.QUERY,
            entities={"order_id": []},
            slot_assessment=assessment,
            decision=decision,
        )

        self.assertFalse(assessment.requires_clarification)
        self.assertTrue(plan.need_knowledge)
        self.assertFalse(plan.need_clarify)
        self.assertFalse(plan.need_order_lookup)

    def test_semantic_live_query_without_order_id_still_requires_clarification(self):
        decision = WorkflowDecision(
            mode=DecisionMode.LIVE_RECORD,
            tools=["shipment_track"],
            specific_order=True,
            confidence=0.95,
            reason="用户在查询自己订单的实时物流",
        )
        assessment = self.slot_manager.assess(
            message="我的订单怎么还没发货？",
            intent=IntentCategory.QUERY,
            entities={"order_id": []},
            decision=decision,
        )

        plan = self.planner.plan(
            message="我的订单怎么还没发货？",
            intent=IntentCategory.QUERY,
            entities={"order_id": []},
            slot_assessment=assessment,
            decision=decision,
        )

        self.assertTrue(assessment.requires_clarification)
        self.assertTrue(plan.need_clarify)
        self.assertIn("订单号", plan.clarify_prompt)

    def test_mixed_knowledge_and_live_query_keeps_knowledge_when_order_id_is_missing(self):
        decision = WorkflowDecision(
            mode=DecisionMode.LIVE_RECORD,
            tools=["knowledge_search", "shipment_track"],
            specific_order=True,
            confidence=0.95,
            reason="用户同时询问物流规则和自己的订单状态",
        )
        assessment = self.slot_manager.assess(
            message="物流多久不动要找客服？顺便帮我看我的订单",
            intent=IntentCategory.QUERY,
            entities={"order_id": []},
            decision=decision,
        )

        plan = self.planner.plan(
            message="物流多久不动要找客服？顺便帮我看我的订单",
            intent=IntentCategory.QUERY,
            entities={"order_id": []},
            slot_assessment=assessment,
            decision=decision,
        )

        self.assertFalse(plan.need_clarify)
        self.assertTrue(plan.need_knowledge)
        self.assertFalse(plan.need_action_tool)
        self.assertIn("订单号", plan.follow_up_prompt)

    def test_state_machine_builds_bounded_act_path(self):
        path = self.state_machine.build_path("act")

        self.assertEqual(path.states[0], WorkflowState.INTAKE)
        self.assertEqual(path.states[1], WorkflowState.ACT)
        self.assertEqual(path.states[-1], WorkflowState.CLOSE)
        self.assertLessEqual(path.action_steps, 3)


if __name__ == "__main__":
    unittest.main()
