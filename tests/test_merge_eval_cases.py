import json
import pathlib
import tempfile
import unittest

from tools.merge_eval_cases import (
    build_balanced_case_set,
    merge_case_files,
)


class MergeEvalCasesTests(unittest.TestCase):
    def test_merge_case_files_combines_multiple_case_arrays(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            base = pathlib.Path(temp_dir)
            case_a = [{
                "case_id": "leave-001",
                "query": "请假最多能请几天",
                "query_type": "current_policy",
                "gold_answer": "全年累计不得超过 12 个工作日。",
                "gold_policy_ids": ["policy_leave"],
                "gold_version_no": "V2",
                "gold_effective_at": "2025-01-01",
                "gold_parent_ids": ["leave-v2:parent:1"],
                "gold_chunk_keys": ["leave-v2:parent:1:0"],
                "must_include_facts": ["12个工作日"],
                "must_not_include_facts": ["10个工作日"],
                "unanswerable": False,
                "difficulty": "easy",
                "evidence_summary": "V2 事假条款明确给出了全年累计上限。",
            }]
            case_b = [{
                "case_id": "refund-001",
                "query": "退款多久到账",
                "query_type": "current_policy",
                "gold_answer": "审批完成后，原路退款通常在 3 至 5 个工作日内到账。",
                "gold_policy_ids": ["policy_refund_compensation"],
                "gold_version_no": "V2",
                "gold_effective_at": "2025-03-20",
                "gold_parent_ids": ["refund-v2:parent:2"],
                "gold_chunk_keys": ["refund-v2:parent:2:0"],
                "must_include_facts": ["3至5个工作日"],
                "must_not_include_facts": ["5至7个工作日"],
                "unanswerable": False,
                "difficulty": "easy",
                "evidence_summary": "V2 时效条款明确给出了当前退款到账时限。",
            }]
            (base / "leave.cases.json").write_text(json.dumps(case_a, ensure_ascii=False), encoding="utf-8")
            (base / "refund.cases.json").write_text(json.dumps(case_b, ensure_ascii=False), encoding="utf-8")

            merged = merge_case_files(sorted(base.glob("*.cases.json")))

            self.assertEqual(len(merged), 2)
            self.assertEqual([item["case_id"] for item in merged], ["leave-001", "refund-001"])

    def test_merge_case_files_rejects_duplicate_case_id(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            base = pathlib.Path(temp_dir)
            duplicate = [{
                "case_id": "dup-001",
                "query": "问题",
                "query_type": "current_policy",
                "gold_answer": "答案",
                "gold_policy_ids": ["policy_leave"],
                "gold_version_no": "V2",
                "gold_effective_at": "2025-01-01",
                "gold_parent_ids": ["p"],
                "gold_chunk_keys": ["c"],
                "must_include_facts": ["a"],
                "must_not_include_facts": ["b"],
                "unanswerable": False,
                "difficulty": "easy",
                "evidence_summary": "证据",
            }]
            (base / "a.cases.json").write_text(json.dumps(duplicate, ensure_ascii=False), encoding="utf-8")
            (base / "b.cases.json").write_text(json.dumps(duplicate, ensure_ascii=False), encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "dup-001"):
                merge_case_files(sorted(base.glob("*.cases.json")))

    def test_build_balanced_case_set_limits_version_sensitive_cases(self):
        cases = []
        for idx in range(8):
            cases.append({
                "case_id": f"current-{idx}",
                "query": "q",
                "query_type": "current_policy",
                "gold_answer": "a",
                "gold_policy_ids": ["policy_x"],
                "gold_version_no": "V2",
                "gold_effective_at": "2025-01-01",
                "gold_parent_ids": ["p"],
                "gold_chunk_keys": ["c"],
                "must_include_facts": ["f"],
                "must_not_include_facts": ["g"],
                "unanswerable": False,
                "difficulty": "easy",
                "evidence_summary": "e",
            })
        for idx in range(4):
            cases.append({
                "case_id": f"history-{idx}",
                "query": "q",
                "query_type": "history_version",
                "gold_answer": "a",
                "gold_policy_ids": ["policy_x"],
                "gold_version_no": "V1",
                "gold_effective_at": "2024-01-01",
                "gold_parent_ids": ["p"],
                "gold_chunk_keys": ["c"],
                "must_include_facts": ["f"],
                "must_not_include_facts": ["g"],
                "unanswerable": False,
                "difficulty": "easy",
                "evidence_summary": "e",
            })
        for idx in range(2):
            cases.append({
                "case_id": f"time-{idx}",
                "query": "q",
                "query_type": "time_based_policy",
                "gold_answer": "a",
                "gold_policy_ids": ["policy_x"],
                "gold_version_no": "V1",
                "gold_effective_at": "2024-01-01",
                "gold_parent_ids": ["p"],
                "gold_chunk_keys": ["c"],
                "must_include_facts": ["f"],
                "must_not_include_facts": ["g"],
                "unanswerable": False,
                "difficulty": "easy",
                "evidence_summary": "e",
            })

        balanced = build_balanced_case_set(cases, max_version_sensitive_ratio=0.125)

        version_sensitive = [
            item for item in balanced
            if item["query_type"] in {"history_version", "time_based_policy"}
        ]
        self.assertLessEqual(len(version_sensitive), 2)
        self.assertEqual(len(balanced), 10)

    def test_build_balanced_case_set_prefers_non_version_cases_when_trimming(self):
        cases = [
            {
                "case_id": "history-001",
                "query": "q",
                "query_type": "history_version",
                "gold_answer": "a",
                "gold_policy_ids": ["policy_x"],
                "gold_version_no": "V1",
                "gold_effective_at": "2024-01-01",
                "gold_parent_ids": ["p"],
                "gold_chunk_keys": ["c"],
                "must_include_facts": ["f"],
                "must_not_include_facts": ["g"],
                "unanswerable": False,
                "difficulty": "easy",
                "evidence_summary": "e",
            },
            {
                "case_id": "time-001",
                "query": "q",
                "query_type": "time_based_policy",
                "gold_answer": "a",
                "gold_policy_ids": ["policy_x"],
                "gold_version_no": "V1",
                "gold_effective_at": "2024-01-01",
                "gold_parent_ids": ["p"],
                "gold_chunk_keys": ["c"],
                "must_include_facts": ["f"],
                "must_not_include_facts": ["g"],
                "unanswerable": False,
                "difficulty": "easy",
                "evidence_summary": "e",
            },
            {
                "case_id": "current-001",
                "query": "q",
                "query_type": "current_policy",
                "gold_answer": "a",
                "gold_policy_ids": ["policy_x"],
                "gold_version_no": "V2",
                "gold_effective_at": "2025-01-01",
                "gold_parent_ids": ["p"],
                "gold_chunk_keys": ["c"],
                "must_include_facts": ["f"],
                "must_not_include_facts": ["g"],
                "unanswerable": False,
                "difficulty": "easy",
                "evidence_summary": "e",
            },
            {
                "case_id": "threshold-001",
                "query": "q",
                "query_type": "threshold_rule",
                "gold_answer": "a",
                "gold_policy_ids": ["policy_x"],
                "gold_version_no": "V2",
                "gold_effective_at": "2025-01-01",
                "gold_parent_ids": ["p"],
                "gold_chunk_keys": ["c"],
                "must_include_facts": ["f"],
                "must_not_include_facts": ["g"],
                "unanswerable": False,
                "difficulty": "easy",
                "evidence_summary": "e",
            },
        ]

        balanced = build_balanced_case_set(cases, max_version_sensitive_ratio=0.25)

        self.assertEqual(
            [item["case_id"] for item in balanced],
            ["current-001", "history-001", "threshold-001"],
        )


if __name__ == "__main__":
    unittest.main()
