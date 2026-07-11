import asyncio
import json
import os
import tempfile
import unittest
import zipfile
from io import BytesIO
from pathlib import Path
from unittest import mock

from fastapi.testclient import TestClient

from api import main as api_main
from agents.agent_orchestrator import AgentResponse, AgentType, Request
from core.intent_recognizer import IntentCategory
from skills.runtime import SkillRuntime, SkillStore
from telemetry.runtime import TraceContext
from workflow.action_models import ActionItem, ActionType, AgentRole, RoutePlan, WorkflowPlan


def _skill_archive() -> bytes:
    archive = BytesIO()
    with zipfile.ZipFile(archive, "w") as package:
        package.writestr("skill.json", json.dumps({
            "id": "refund-policy",
            "name": "Refund Policy",
            "version": "1.0.0",
            "description": "回答退款规则问题",
            "targets": {
                "agent_roles": ["billing"],
                "intent_categories": ["billing"],
                "goals": ["knowledge"],
            },
            "allowed_tools": ["knowledge_search"],
            "prompt_file": "prompt.md",
        }))
        package.writestr("prompt.md", "仅依据退款知识库作答。")
    return archive.getvalue()


class SkillAdminApiTests(unittest.TestCase):
    def setUp(self):
        self._temp_dir = tempfile.TemporaryDirectory()
        self._env_patch = mock.patch.dict(os.environ, {
            "SKILL_ADMIN_TOKEN": "local-skill-token",
            "APP_ENV": "development",
        })
        self._env_patch.start()
        root = Path(self._temp_dir.name)
        self._original_runtime = getattr(api_main, "_skill_runtime", None)
        self._original_store = getattr(api_main, "_skill_store", None)
        self._original_orchestrator = api_main._orchestrator
        api_main._skill_runtime = SkillRuntime(
            catalog_dir=root / "catalog",
            published_dir=root / "published",
            drafts_dir=root / "drafts",
            known_tools={"knowledge_search"},
        )
        api_main._skill_store = SkillStore(runtime=api_main._skill_runtime, review_required=False)

    def tearDown(self):
        api_main._skill_runtime = self._original_runtime
        api_main._skill_store = self._original_store
        api_main._orchestrator = self._original_orchestrator
        self._env_patch.stop()
        self._temp_dir.cleanup()

    def test_upload_endpoint_auto_publishes_development_skill_with_valid_token(self):
        client = TestClient(api_main.app)

        response = client.post(
            "/admin/skills/upload",
            files={"file": ("refund-policy.zip", _skill_archive(), "application/zip")},
            headers={"X-Skill-Admin-Token": "local-skill-token"},
        )

        self.assertEqual(response.status_code, 201)
        self.assertEqual(response.json()["status"], "published")
        self.assertTrue(response.json()["auto_published"])
        self.assertIn("refund-policy", api_main._skill_runtime.snapshot.skills)

    def test_development_defaults_to_auto_publish_unless_review_is_explicitly_enabled(self):
        with mock.patch.dict(os.environ, {"APP_ENV": "development"}, clear=True):
            self.assertFalse(api_main._skill_review_required())
        with mock.patch.dict(os.environ, {"APP_ENV": "development", "SKILL_REVIEW_REQUIRED": "true"}, clear=True):
            self.assertTrue(api_main._skill_review_required())
        with mock.patch.dict(os.environ, {"APP_ENV": "production"}, clear=True):
            self.assertTrue(api_main._skill_review_required())

    def test_skill_runtime_factory_uses_every_registered_mcp_tool(self):
        with mock.patch.dict(os.environ, {
            "SKILL_CATALOG_DIR": str(Path(self._temp_dir.name) / "catalog"),
            "SKILL_DATA_DIR": str(Path(self._temp_dir.name) / "data"),
        }, clear=True):
            tool_manager = type("ToolManager", (), {"_tools": {"human_handoff": object()}})()
            runtime = api_main._create_skill_runtime(tool_manager)
        package_dir = runtime.catalog_dir / "handoff-guide" / "1.0.0"
        package_dir.mkdir(parents=True)
        (package_dir / "skill.json").write_text(json.dumps({
            "id": "handoff-guide",
            "name": "Handoff Guide",
            "version": "1.0.0",
            "description": "人工转接规范",
            "targets": {
                "agent_roles": ["escalation"],
                "intent_categories": ["escalation"],
                "goals": ["action"],
                "tools": ["human_handoff"],
            },
            "allowed_tools": ["human_handoff"],
            "prompt_file": "prompt.md",
        }), encoding="utf-8")
        (package_dir / "prompt.md").write_text("保留完整会话上下文。", encoding="utf-8")

        runtime.refresh()

        self.assertIn("handoff-guide", runtime.snapshot.skills)

    def test_catalog_endpoint_exposes_debuggable_skill_state_without_prompt_content(self):
        response = TestClient(api_main.app).get(
            "/admin/skills",
            headers={"X-Skill-Admin-Token": "local-skill-token"},
        )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["generation"], 0)
        self.assertEqual(response.json()["active"], [])
        self.assertEqual(response.json()["drafts"], [])
        self.assertNotIn("prompt", response.text)

    def test_production_upload_remains_a_draft_until_the_publish_endpoint_is_called(self):
        api_main._skill_store = SkillStore(runtime=api_main._skill_runtime, review_required=True)
        client = TestClient(api_main.app)
        upload = client.post(
            "/admin/skills/upload",
            files={"file": ("refund-policy.zip", _skill_archive(), "application/zip")},
            headers={"X-Skill-Admin-Token": "local-skill-token"},
        )

        self.assertEqual(upload.status_code, 201)
        self.assertEqual(upload.json()["status"], "draft")
        self.assertNotIn("refund-policy", api_main._skill_runtime.snapshot.skills)

        publish = client.post(
            f"/admin/skills/{upload.json()['draft_id']}/publish",
            headers={"X-Skill-Admin-Token": "local-skill-token"},
        )

        self.assertEqual(publish.status_code, 200)
        self.assertEqual(publish.json()["status"], "published")
        self.assertIn("refund-policy", api_main._skill_runtime.snapshot.skills)

    def test_plan_skill_context_uses_the_request_snapshot_and_agent_role(self):
        package_dir = api_main._skill_runtime.catalog_dir / "refund-policy" / "1.0.0"
        package_dir.mkdir(parents=True)
        (package_dir / "skill.json").write_text(json.dumps({
            "id": "refund-policy",
            "name": "Refund Policy",
            "version": "1.0.0",
            "description": "回答退款规则问题",
            "targets": {
                "agent_roles": ["billing"],
                "intent_categories": ["billing"],
                "goals": ["knowledge"],
                "tools": ["knowledge_search"],
            },
            "allowed_tools": ["knowledge_search"],
            "prompt_file": "prompt.md",
        }), encoding="utf-8")
        (package_dir / "prompt.md").write_text("仅依据退款知识库作答。", encoding="utf-8")
        api_main._skill_runtime.refresh()
        snapshot = api_main._skill_runtime.snapshot
        plan = WorkflowPlan(
            complexity="single_domain",
            primary_intent="billing",
            primary_goal="knowledge",
            actions=[ActionItem(
                id="a1",
                type=ActionType.RETRIEVE_POLICY,
                objective="检索知识库",
                domain="knowledge",
                tool_name="knowledge_search",
            )],
            route_plan=RoutePlan(
                primary_agent=AgentRole.BILLING,
                supporting_agents=[AgentRole.TECHNICAL],
            ),
        )

        context = api_main._build_agent_skills_context(snapshot, plan, IntentCategory.BILLING)

        self.assertEqual([skill["id"] for skill in context["billing"]], ["refund-policy"])
        self.assertEqual(context["billing"][0]["allowed_tools"], ["knowledge_search"])
        self.assertEqual(context["technical"], [])

    def test_plan_skill_context_includes_a_distinct_final_writer_role(self):
        package_dir = api_main._skill_runtime.catalog_dir / "final-answer-style" / "1.0.0"
        package_dir.mkdir(parents=True)
        (package_dir / "skill.json").write_text(json.dumps({
            "id": "final-answer-style",
            "name": "Final Answer Style",
            "version": "1.0.0",
            "description": "最终答复规范",
            "targets": {
                "agent_roles": ["general"],
                "intent_categories": ["billing"],
                "goals": ["knowledge"],
            },
            "allowed_tools": [],
            "prompt_file": "prompt.md",
        }), encoding="utf-8")
        (package_dir / "prompt.md").write_text("输出简短、明确的最终答复。", encoding="utf-8")
        api_main._skill_runtime.refresh()
        plan = WorkflowPlan(
            complexity="single_domain",
            primary_intent="billing",
            primary_goal="knowledge",
            route_plan=RoutePlan(
                primary_agent=AgentRole.BILLING,
                final_writer=AgentRole.GENERAL,
            ),
        )

        context = api_main._build_agent_skills_context(
            api_main._skill_runtime.snapshot,
            plan,
            IntentCategory.BILLING,
        )

        self.assertEqual([skill["id"] for skill in context["general"]], ["final-answer-style"])

    def test_planned_orchestrator_passes_selected_skills_to_the_agent_execution_context(self):
        package_dir = api_main._skill_runtime.catalog_dir / "refund-policy" / "1.0.0"
        package_dir.mkdir(parents=True)
        (package_dir / "skill.json").write_text(json.dumps({
            "id": "refund-policy",
            "name": "Refund Policy",
            "version": "1.0.0",
            "description": "回答退款规则问题",
            "targets": {
                "agent_roles": ["billing"],
                "intent_categories": ["billing"],
                "goals": ["knowledge"],
                "tools": ["knowledge_search"],
            },
            "allowed_tools": ["knowledge_search"],
            "prompt_file": "prompt.md",
        }), encoding="utf-8")
        (package_dir / "prompt.md").write_text("仅依据退款知识库作答。", encoding="utf-8")
        api_main._skill_runtime.refresh()
        captured = {}

        class _FakeOrchestrator:
            async def _execute(self, request, agent_type, context=None):
                captured["context"] = context
                return AgentResponse(agent_type=agent_type, content="退款规则", success=True)

        api_main._orchestrator = _FakeOrchestrator()
        plan = WorkflowPlan(
            complexity="single_domain",
            primary_intent="billing",
            primary_goal="knowledge",
            actions=[ActionItem(
                id="a1",
                type=ActionType.RETRIEVE_POLICY,
                objective="检索知识库",
                domain="knowledge",
                tool_name="knowledge_search",
            )],
            route_plan=RoutePlan(primary_agent=AgentRole.BILLING),
        )
        request = Request(
            message="退款规则",
            user_id="u1",
            conv_id="c1",
            intent=IntentCategory.BILLING,
        )

        trace = TraceContext(user_id="u1", conv_id="c1", message="退款规则")
        with trace.stage("orchestrator.run"):
            asyncio.run(api_main._run_planned_orchestrator(
                request,
                plan,
                trace=trace,
                skill_snapshot=api_main._skill_runtime.snapshot,
        ))

        self.assertEqual(captured["context"]["agent_skills_by_role"]["billing"][0]["id"], "refund-policy")
        run_stage = next(stage for stage in trace._stages if stage.name == "orchestrator.run")
        self.assertEqual(run_stage.meta["skill_snapshot_generation"], 1)
        self.assertEqual(run_stage.meta["skill_ids_by_role"], {"billing": ["refund-policy"]})


if __name__ == "__main__":
    unittest.main()
