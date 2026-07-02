import json
import pathlib
import tempfile
import unittest

from tools.split_annotation_batches import build_annotation_batches


class SplitAnnotationBatchesTests(unittest.TestCase):
    def test_build_annotation_batches_creates_one_batch_per_policy(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            base = pathlib.Path(temp_dir)
            annotation_input = {
                "total_policies": 3,
                "policies": [
                    {
                        "policy_id": "policy_reimbursement",
                        "title": "报销制度",
                        "versions": [{"version_no": "V1"}, {"version_no": "V2"}],
                        "active_chunks": [{"chunk_key": "a1", "content": "报销到账 3 个工作日"}],
                        "archive_chunks": [{"chunk_key": "a0", "content": "报销到账 5 个工作日"}],
                    },
                    {
                        "policy_id": "policy_refund_compensation",
                        "title": "退款与对外赔付审批制度",
                        "versions": [{"version_no": "V1"}, {"version_no": "V2"}],
                        "active_chunks": [{"chunk_key": "b1", "content": "退款到账 3-5 个工作日"}],
                        "archive_chunks": [{"chunk_key": "b0", "content": "退款到账 5-7 个工作日"}],
                    },
                    {
                        "policy_id": "policy_leave",
                        "title": "请假制度",
                        "versions": [{"version_no": "V1"}, {"version_no": "V2"}],
                        "active_chunks": [{"chunk_key": "c1", "content": "请假审批 6 个工作小时"}],
                        "archive_chunks": [{"chunk_key": "c0", "content": "请假审批 8 个工作小时"}],
                    },
                ],
            }
            input_path = base / "annotation_input.json"
            input_path.write_text(json.dumps(annotation_input, ensure_ascii=False), encoding="utf-8")

            batches = build_annotation_batches(input_path, distractor_count=2)

            self.assertEqual(len(batches), 3)
            self.assertEqual(batches[0]["target_policy"]["policy_id"], "policy_leave")
            self.assertIn("distractor_policies", batches[0])

    def test_build_annotation_batches_prioritizes_preferred_distractors(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            base = pathlib.Path(temp_dir)
            annotation_input = {
                "total_policies": 4,
                "policies": [
                    {
                        "policy_id": "policy_reimbursement",
                        "title": "报销制度",
                        "versions": [{"version_no": "V1"}, {"version_no": "V2"}],
                        "active_chunks": [{"chunk_key": "a1", "content": "报销到账 3 个工作日"}],
                        "archive_chunks": [{"chunk_key": "a0", "content": "报销到账 5 个工作日"}],
                    },
                    {
                        "policy_id": "policy_refund_compensation",
                        "title": "退款与对外赔付审批制度",
                        "versions": [{"version_no": "V1"}, {"version_no": "V2"}],
                        "active_chunks": [{"chunk_key": "b1", "content": "退款到账 3-5 个工作日"}],
                        "archive_chunks": [{"chunk_key": "b0", "content": "退款到账 5-7 个工作日"}],
                    },
                    {
                        "policy_id": "policy_travel",
                        "title": "差旅制度",
                        "versions": [{"version_no": "V1"}, {"version_no": "V2"}],
                        "active_chunks": [{"chunk_key": "c1", "content": "差旅补贴 150 元"}],
                        "archive_chunks": [{"chunk_key": "c0", "content": "差旅补贴 120 元"}],
                    },
                    {
                        "policy_id": "policy_leave",
                        "title": "请假制度",
                        "versions": [{"version_no": "V1"}, {"version_no": "V2"}],
                        "active_chunks": [{"chunk_key": "d1", "content": "请假审批 6 个工作小时"}],
                        "archive_chunks": [{"chunk_key": "d0", "content": "请假审批 8 个工作小时"}],
                    },
                ],
            }
            input_path = base / "annotation_input.json"
            input_path.write_text(json.dumps(annotation_input, ensure_ascii=False), encoding="utf-8")

            batches = build_annotation_batches(input_path, distractor_count=2)
            reimbursement_batch = next(
                batch for batch in batches if batch["target_policy"]["policy_id"] == "policy_reimbursement"
            )

            distractor_ids = [item["policy_id"] for item in reimbursement_batch["distractor_policies"]]
            self.assertIn("policy_refund_compensation", distractor_ids)
            self.assertEqual(len(distractor_ids), 2)


if __name__ == "__main__":
    unittest.main()
