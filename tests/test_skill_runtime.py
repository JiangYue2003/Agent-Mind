import json
import tempfile
import unittest
import zipfile
from io import BytesIO
from pathlib import Path
from unittest import mock

from skills.runtime import SkillRuntime, SkillStore, SkillValidationError


class SkillRuntimeTests(unittest.TestCase):
    def test_general_service_style_catalog_skill_matches_general_agent_without_tools(self):
        repo_root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            runtime = SkillRuntime(
                catalog_dir=repo_root / "skills" / "catalog",
                published_dir=root / "published",
                drafts_dir=root / "drafts",
                known_tools=set(),
            )

            runtime.refresh()

            selected = runtime.select(
                agent_role="general",
                intent_category="greeting",
                goal="knowledge",
                planned_tools=[],
            )

            self.assertEqual([skill.id for skill in selected], ["general-service-style"])
            self.assertEqual(selected[0].allowed_tools, ())

    def test_refresh_loads_a_skill_selected_by_agent_intent_goal_and_tool(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            package_dir = root / "catalog" / "refund-policy" / "1.0.0"
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
                "priority": 20,
            }), encoding="utf-8")
            (package_dir / "prompt.md").write_text("仅依据退款知识库作答。", encoding="utf-8")

            runtime = SkillRuntime(
                catalog_dir=root / "catalog",
                published_dir=root / "published",
                drafts_dir=root / "drafts",
                known_tools={"knowledge_search", "order_lookup"},
            )
            runtime.refresh()

            selected = runtime.select(
                agent_role="billing",
                intent_category="billing",
                goal="knowledge",
                planned_tools=["knowledge_search"],
            )

            self.assertEqual([skill.id for skill in selected], ["refund-policy"])
            self.assertEqual(selected[0].prompt, "仅依据退款知识库作答。")
            self.assertEqual(runtime.snapshot.generation, 1)

    def test_development_upload_publishes_and_activates_a_valid_skill_immediately(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            runtime = SkillRuntime(
                catalog_dir=root / "catalog",
                published_dir=root / "published",
                drafts_dir=root / "drafts",
                known_tools={"knowledge_search"},
            )
            store = SkillStore(runtime=runtime, review_required=False)

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
                        "tools": ["knowledge_search"],
                    },
                    "allowed_tools": ["knowledge_search"],
                    "prompt_file": "prompt.md",
                }))
                package.writestr("prompt.md", "仅依据退款知识库作答。")

            result = store.upload_zip(archive.getvalue())

            self.assertEqual(result.status, "published")
            self.assertTrue(result.auto_published)
            self.assertIn("refund-policy", runtime.snapshot.skills)

    def test_catalog_file_change_creates_a_new_snapshot(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            package_dir = root / "catalog" / "refund-policy" / "1.0.0"
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
                },
                "allowed_tools": [],
                "prompt_file": "prompt.md",
            }), encoding="utf-8")
            prompt_path = package_dir / "prompt.md"
            prompt_path.write_text("旧提示词。", encoding="utf-8")
            runtime = SkillRuntime(
                catalog_dir=root / "catalog",
                published_dir=root / "published",
                drafts_dir=root / "drafts",
                known_tools=set(),
            )
            runtime.refresh()

            prompt_path.write_text("新的、更长的提示词。", encoding="utf-8")

            self.assertTrue(runtime.refresh_if_changed())
            self.assertEqual(runtime.snapshot.generation, 2)
            self.assertEqual(runtime.snapshot.skills["refund-policy"].prompt, "新的、更长的提示词。")
            self.assertFalse(runtime.refresh_if_changed())

    def test_selection_from_a_captured_snapshot_does_not_change_after_reload(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            package_dir = root / "catalog" / "refund-policy" / "1.0.0"
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
                },
                "allowed_tools": [],
                "prompt_file": "prompt.md",
            }), encoding="utf-8")
            prompt_path = package_dir / "prompt.md"
            prompt_path.write_text("旧提示词。", encoding="utf-8")
            runtime = SkillRuntime(
                catalog_dir=root / "catalog",
                published_dir=root / "published",
                drafts_dir=root / "drafts",
                known_tools=set(),
            )
            runtime.refresh()
            captured = runtime.snapshot
            prompt_path.write_text("新的、更长的提示词。", encoding="utf-8")
            runtime.refresh_if_changed()

            selected = runtime.select_from_snapshot(
                captured,
                agent_role="billing",
                intent_category="billing",
                goal="knowledge",
                planned_tools=[],
            )

            self.assertEqual(selected[0].prompt, "旧提示词。")

    def test_upload_rejects_a_non_utf8_prompt_as_a_validation_error(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            runtime = SkillRuntime(
                catalog_dir=root / "catalog",
                published_dir=root / "published",
                drafts_dir=root / "drafts",
                known_tools=set(),
            )
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
                    "allowed_tools": [],
                    "prompt_file": "prompt.md",
                }))
                package.writestr("prompt.md", b"\xff")

            with self.assertRaises(SkillValidationError):
                SkillStore(runtime=runtime, review_required=False).upload_zip(archive.getvalue())

    def test_auto_publish_fails_when_an_invalid_catalog_package_blocks_activation(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            invalid_dir = root / "catalog" / "invalid-skill" / "1.0.0"
            invalid_dir.mkdir(parents=True)
            (invalid_dir / "skill.json").write_text(json.dumps({
                "id": "invalid-skill",
                "name": "Invalid Skill",
                "version": "1.0.0",
                "description": "invalid",
                "targets": {
                    "agent_roles": ["billing"],
                    "intent_categories": ["billing"],
                    "goals": ["knowledge"],
                },
                "allowed_tools": ["missing_tool"],
                "prompt_file": "prompt.md",
            }), encoding="utf-8")
            (invalid_dir / "prompt.md").write_text("invalid", encoding="utf-8")
            runtime = SkillRuntime(
                catalog_dir=root / "catalog",
                published_dir=root / "published",
                drafts_dir=root / "drafts",
                known_tools={"knowledge_search"},
            )
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

            with self.assertRaises(SkillValidationError):
                SkillStore(runtime=runtime, review_required=False).upload_zip(archive.getvalue())

            self.assertNotIn("refund-policy", runtime.snapshot.skills)
            self.assertFalse((root / "published" / "refund-policy" / "1.0.0").exists())

    def test_refresh_retries_when_catalog_changes_while_it_is_being_read(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            package_dir = root / "catalog" / "refund-policy" / "1.0.0"
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
                },
                "allowed_tools": [],
                "prompt_file": "prompt.md",
            }), encoding="utf-8")
            prompt_path = package_dir / "prompt.md"
            prompt_path.write_text("初始提示词。", encoding="utf-8")
            runtime = SkillRuntime(
                catalog_dir=root / "catalog",
                published_dir=root / "published",
                drafts_dir=root / "drafts",
                known_tools=set(),
            )
            runtime.refresh()
            prompt_path.write_text("读取期间的提示词。", encoding="utf-8")
            original_load = runtime._load_active_skills
            changed = False

            def load_then_change():
                nonlocal changed
                loaded = original_load()
                if not changed:
                    changed = True
                    prompt_path.write_text("最终稳定的提示词。", encoding="utf-8")
                return loaded

            with mock.patch.object(runtime, "_load_active_skills", side_effect=load_then_change):
                runtime.refresh()

            self.assertEqual(runtime.snapshot.skills["refund-policy"].prompt, "最终稳定的提示词。")

    def test_auto_publish_rejects_a_duplicate_id_and_version_from_the_git_catalog(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            package_dir = root / "catalog" / "refund-policy" / "1.0.0"
            package_dir.mkdir(parents=True)
            manifest = {
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
            }
            (package_dir / "skill.json").write_text(json.dumps(manifest), encoding="utf-8")
            (package_dir / "prompt.md").write_text("Git 版本提示词。", encoding="utf-8")
            runtime = SkillRuntime(
                catalog_dir=root / "catalog",
                published_dir=root / "published",
                drafts_dir=root / "drafts",
                known_tools={"knowledge_search"},
            )
            runtime.refresh()
            archive = BytesIO()
            with zipfile.ZipFile(archive, "w") as package:
                package.writestr("skill.json", json.dumps(manifest))
                package.writestr("prompt.md", "上传版本提示词。")

            with self.assertRaises(SkillValidationError):
                SkillStore(runtime=runtime, review_required=False).upload_zip(archive.getvalue())

            self.assertEqual(runtime.snapshot.skills["refund-policy"].source, "catalog")
            self.assertFalse((root / "published" / "refund-policy" / "1.0.0").exists())

    def test_allowed_tools_restrict_skill_activation_when_no_matching_tool_is_planned(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            package_dir = root / "catalog" / "refund-policy" / "1.0.0"
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
                },
                "allowed_tools": ["knowledge_search"],
                "prompt_file": "prompt.md",
            }), encoding="utf-8")
            (package_dir / "prompt.md").write_text("仅依据退款知识库作答。", encoding="utf-8")
            runtime = SkillRuntime(
                catalog_dir=root / "catalog",
                published_dir=root / "published",
                drafts_dir=root / "drafts",
                known_tools={"knowledge_search"},
            )
            runtime.refresh()

            selected = runtime.select(
                agent_role="billing",
                intent_category="billing",
                goal="knowledge",
                planned_tools=[],
            )

            self.assertEqual(selected, [])

    def test_auto_publish_rolls_back_when_catalog_changes_during_fingerprinting(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            runtime = SkillRuntime(
                catalog_dir=root / "catalog",
                published_dir=root / "published",
                drafts_dir=root / "drafts",
                known_tools={"knowledge_search"},
            )
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

            with mock.patch.object(runtime, "_source_fingerprint", side_effect=FileNotFoundError("catalog changed")):
                with self.assertRaises(SkillValidationError):
                    SkillStore(runtime=runtime, review_required=False).upload_zip(archive.getvalue())

            self.assertFalse((root / "published" / "refund-policy" / "1.0.0").exists())

    def test_catalog_status_reports_the_last_successful_refresh_time(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            runtime = SkillRuntime(
                catalog_dir=root / "catalog",
                published_dir=root / "published",
                drafts_dir=root / "drafts",
                known_tools=set(),
            )

            runtime.refresh()

            self.assertIsNotNone(runtime.describe()["last_refresh_at"])


if __name__ == "__main__":
    unittest.main()
