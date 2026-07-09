import unittest

from core.intent_recognizer import IntentCategory
from workflow.action_planner import ActionPlanner
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

    def test_state_machine_builds_bounded_act_path(self):
        path = self.state_machine.build_path("act")

        self.assertEqual(path.states[0], WorkflowState.INTAKE)
        self.assertEqual(path.states[1], WorkflowState.ACT)
        self.assertEqual(path.states[-1], WorkflowState.CLOSE)
        self.assertLessEqual(path.action_steps, 3)


if __name__ == "__main__":
    unittest.main()
