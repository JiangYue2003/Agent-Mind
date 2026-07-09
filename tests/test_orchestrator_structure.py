import inspect
import unittest

from agents.agent_orchestrator import AgentOrchestrator, AgentResponse, AgentType, Request


class _FakeAgent:
    def __init__(self, response: AgentResponse):
        self._response = response

    async def handle(self, req, context=None):
        return self._response


class OrchestratorStructureTests(unittest.TestCase):
    def test_orchestrator_exposes_monitoring_methods(self):
        self.assertTrue(hasattr(AgentOrchestrator, "get_stats"))
        self.assertTrue(hasattr(AgentOrchestrator, "update_routing_penalties"))

    def test_execute_accepts_optional_context(self):
        sig = inspect.signature(AgentOrchestrator._execute)
        self.assertIn("context", sig.parameters)

    def test_execute_general_success_does_not_require_fallback(self):
        orchestrator = AgentOrchestrator.__new__(AgentOrchestrator)
        response = AgentResponse(
            agent_type=AgentType.GENERAL,
            content="你好，我是 EchoMind 智能客服。",
            success=True,
        )
        general_agent = _FakeAgent(response)

        def _best_agent(agent_type):
            if agent_type == AgentType.GENERAL:
                return general_agent
            return None

        orchestrator._best_agent = _best_agent
        req = Request(message="你是谁", user_id="u1", conv_id="c1")

        import asyncio

        result = asyncio.run(orchestrator._execute(req, AgentType.GENERAL))
        self.assertTrue(result.success)
        self.assertEqual(result.content, "你好，我是 EchoMind 智能客服。")


if __name__ == "__main__":
    unittest.main()
