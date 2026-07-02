import json
import pathlib
import tempfile
import unittest

from tools.prepare_eval_annotation_input import build_annotation_input


class PrepareEvalAnnotationInputTests(unittest.TestCase):
    def test_build_annotation_input_groups_by_policy_and_scope(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            base = pathlib.Path(temp_dir)
            metadata = {
                "metadata": [
                    {
                        "filename": "请假制度-v1.md",
                        "title": "请假制度",
                        "policy_id": "policy_leave",
                        "version_no": "V1",
                        "effective_at": "2023-09-01",
                        "issue_code": "人资〔2023〕18号",
                        "scope_key": "",
                    },
                    {
                        "filename": "请假制度-v2.md",
                        "title": "请假制度",
                        "policy_id": "policy_leave",
                        "version_no": "V2",
                        "effective_at": "2025-01-01",
                        "issue_code": "人资〔2025〕01号",
                        "scope_key": "",
                    },
                ]
            }
            active = {
                "scope": "active",
                "items": [
                    {
                        "chunk_key": "v2-parent-0:0",
                        "policy_id": "policy_leave",
                        "title": "请假制度",
                        "version_no": "V2",
                        "effective_at": "2025-01-01",
                        "issue_code": "人资〔2025〕01号",
                        "parent_id": "v2-parent-0",
                        "heading_path": "请假制度 > 审批时限",
                        "section_title": "审批时限",
                        "content": "直属主管应在请假单提交后 6 个工作小时内完成审批。",
                    }
                ],
            }
            archive = {
                "scope": "archive",
                "items": [
                    {
                        "chunk_key": "v1-parent-0:0",
                        "policy_id": "policy_leave",
                        "title": "请假制度",
                        "version_no": "V1",
                        "effective_at": "2023-09-01",
                        "issue_code": "人资〔2023〕18号",
                        "parent_id": "v1-parent-0",
                        "heading_path": "请假制度 > 审批时限",
                        "section_title": "审批时限",
                        "content": "直属主管应在请假单提交后 8 个工作小时内完成审批。",
                    }
                ],
            }

            (base / "metadata.json").write_text(json.dumps(metadata, ensure_ascii=False), encoding="utf-8")
            (base / "active.json").write_text(json.dumps(active, ensure_ascii=False), encoding="utf-8")
            (base / "archive.json").write_text(json.dumps(archive, ensure_ascii=False), encoding="utf-8")

            result = build_annotation_input(
                base / "metadata.json",
                base / "active.json",
                base / "archive.json",
            )

            self.assertEqual(result["total_policies"], 1)
            self.assertEqual(result["policies"][0]["policy_id"], "policy_leave")
            self.assertEqual(len(result["policies"][0]["active_chunks"]), 1)
            self.assertEqual(len(result["policies"][0]["archive_chunks"]), 1)
            self.assertEqual(result["policies"][0]["versions"][0]["version_no"], "V1")
            self.assertEqual(result["policies"][0]["versions"][1]["version_no"], "V2")


if __name__ == "__main__":
    unittest.main()
