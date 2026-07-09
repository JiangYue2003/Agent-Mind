import asyncio
import unittest

from agents.agent_orchestrator import AgentResponse, AgentType, OrchestratorResult, Request
from api import main as api_main
from workflow.action_models import AgentRole, MergeMode, RoutePlan, WorkflowPlan


class _FakePrimarySummarizeOrchestrator:
    def __init__(self):
        self.execute_calls = []
        self.run_calls = []

    async def _execute(self, req, agent_type, context=None):
        self.execute_calls.append({
            "agent_type": agent_type,
            "message": req.message,
            "context": req.context,
        })
        if agent_type == AgentType.TECHNICAL:
            return AgentResponse(
                agent_type=agent_type,
                content="[问题判断]\n疑似服务端接口异常\n[关键事实]\n用户反馈退款失败且出现 500\n[风险与限制]\n需要后台排查日志\n[建议动作]\n先检查退款接口与订单状态一致性",
                success=True,
            )
        return AgentResponse(
            agent_type=agent_type,
            content="已结合账单和技术信息整理最终答复。",
            success=True,
        )

    async def run(self, req, context=None):
        self.run_calls.append({"message": req.message, "context": req.context})
        return OrchestratorResult(
            request_id=req.request_id,
            response="fallback run",
            agent_type=AgentType.GENERAL,
            intent=req.intent,
            latency_ms=1.0,
        )


class PrimarySummarizeTests(unittest.TestCase):
    def setUp(self):
        self._original_orchestrator = api_main._orchestrator

    def tearDown(self):
        api_main._orchestrator = self._original_orchestrator

    def test_primary_summarize_runs_supporting_agent_then_primary_writer(self):
        fake = _FakePrimarySummarizeOrchestrator()
        api_main._orchestrator = fake
        req = Request(
            message="我的订单退款一直失败，而且 App 还报 500",
            user_id="u123",
            conv_id="conv-ps-1",
            context="[知识库检索结果]\n退款规则说明",
        )
        plan = WorkflowPlan(
            complexity="cross_domain",
            primary_intent="billing",
            route_plan=RoutePlan(
                primary_agent=AgentRole.BILLING,
                supporting_agents=[AgentRole.TECHNICAL],
                merge_mode=MergeMode.PRIMARY_SUMMARIZE,
                final_writer=AgentRole.BILLING,
                reason="账单主答，技术辅助",
            ),
        )

        result = asyncio.run(api_main._run_planned_orchestrator(req, plan, trace=None))

        self.assertEqual(len(fake.execute_calls), 2)
        self.assertEqual(fake.execute_calls[0]["agent_type"], AgentType.TECHNICAL)
        self.assertEqual(fake.execute_calls[1]["agent_type"], AgentType.BILLING)
        self.assertEqual(len(fake.run_calls), 0)
        self.assertIn("[辅助专家结构化意见]", fake.execute_calls[1]["context"])
        self.assertIn("[technical 结构化意见]", fake.execute_calls[1]["context"])
        self.assertEqual(result.agent_type, AgentType.BILLING)
        self.assertEqual(result.response, "已结合账单和技术信息整理最终答复。")


if __name__ == "__main__":
    unittest.main()
