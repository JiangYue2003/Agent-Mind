import inspect
import unittest
import asyncio

from agents.agent_orchestrator import AgentOrchestrator, AgentResponse, AgentType, GeneralAgent, Request
from core.intent_recognizer import IntentCategory
from telemetry.runtime import TraceContext


class _FakeAgent:
    def __init__(self, response: AgentResponse):
        self._response = response

    async def handle(self, req, context=None):
        return self._response


class _CapturingAgent(_FakeAgent):
    def __init__(self, response):
        super().__init__(response)
        self.context = None

    async def handle(self, req, context=None):
        self.context = context
        return self._response


class _FakeMessages:
    def __init__(self):
        self.calls = []

    async def create(self, **kwargs):
        self.calls.append(kwargs)
        return type("Response", (), {
            "content": [type("Content", (), {"text": "ok"})()],
        })()


class _FakeClient:
    def __init__(self):
        self.messages = _FakeMessages()


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

        result = asyncio.run(orchestrator._execute(req, AgentType.GENERAL))
        self.assertTrue(result.success)
        self.assertEqual(result.content, "你好，我是 EchoMind 智能客服。")

    def test_run_passes_trace_context_into_execute(self):
        orchestrator = AgentOrchestrator.__new__(AgentOrchestrator)
        captured = {}

        def _collaboration_targets(req):
            return []

        def _route(intent, urgency):
            return AgentType.GENERAL

        async def _execute(req, agent_type, context=None):
            captured["context"] = context
            return AgentResponse(
                agent_type=agent_type,
                content="你好，我是 EchoMind 智能客服。",
                success=True,
            )

        orchestrator._collaboration_targets = _collaboration_targets
        orchestrator._route = _route
        orchestrator._execute = _execute

        req = Request(
            message="你是谁",
            user_id="u1",
            conv_id="c1",
            intent=IntentCategory.GREETING,
        )
        trace = TraceContext(user_id="u1", conv_id="c1", message="你是谁")

        result = asyncio.run(orchestrator.run(req, context={"trace": trace}))

        self.assertEqual(result.agent_type, AgentType.GENERAL)
        self.assertIsNotNone(captured.get("context"))
        self.assertIs(captured["context"]["trace"], trace)

    def test_agent_adds_request_scoped_skill_prompt_without_mutating_base_prompt(self):
        client = _FakeClient()
        agent = GeneralAgent(client, "test-model")
        req = Request(message="退款多久到账", user_id="u1", conv_id="c1")
        base_prompt = agent.system_prompt

        asyncio.run(agent._call_llm(req, context={
            "agent_skills": [{
                "id": "refund-policy",
                "prompt": "仅依据退款知识库作答。",
            }],
        }))

        system_prompt = client.messages.calls[0]["system"]
        self.assertIn("[Skill: refund-policy]", system_prompt)
        self.assertIn("仅依据退款知识库作答。", system_prompt)
        self.assertEqual(agent.system_prompt, base_prompt)

    def test_execute_passes_only_the_target_agents_skills(self):
        orchestrator = AgentOrchestrator.__new__(AgentOrchestrator)
        response = AgentResponse(
            agent_type=AgentType.BILLING,
            content="退款规则如下。",
            success=True,
        )
        billing_agent = _CapturingAgent(response)
        orchestrator._best_agent = lambda agent_type: billing_agent if agent_type == AgentType.BILLING else None

        asyncio.run(orchestrator._execute(
            Request(message="退款规则", user_id="u1", conv_id="c1"),
            AgentType.BILLING,
            context={
                "agent_skills_by_role": {
                    "billing": [{"id": "refund-policy", "prompt": "退款规则约束"}],
                    "technical": [{"id": "incident-playbook", "prompt": "排障约束"}],
                },
            },
        ))

        self.assertEqual(billing_agent.context["agent_skills"], [{
            "id": "refund-policy",
            "prompt": "退款规则约束",
        }])


if __name__ == "__main__":
    unittest.main()
