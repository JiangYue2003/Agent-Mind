import json
import pathlib
import tempfile
import unittest

from tools.filter_eval_cases import filter_cases, load_cases


class FilterEvalCasesTests(unittest.TestCase):
    def test_load_cases_reads_json_array(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = pathlib.Path(temp_dir) / "cases.json"
            payload = [{"case_id": "a", "unanswerable": False}]
            path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")

            loaded = load_cases(path)

            self.assertEqual(loaded, payload)

    def test_filter_cases_excludes_unanswerable_by_default(self):
        cases = [
            {"case_id": "a", "unanswerable": False, "query_type": "current_policy"},
            {"case_id": "b", "unanswerable": True, "query_type": "unanswerable"},
            {"case_id": "c", "unanswerable": False, "query_type": "history_version"},
        ]

        filtered = filter_cases(cases)

        self.assertEqual([item["case_id"] for item in filtered], ["a", "c"])

    def test_filter_cases_can_keep_only_unanswerable(self):
        cases = [
            {"case_id": "a", "unanswerable": False, "query_type": "current_policy"},
            {"case_id": "b", "unanswerable": True, "query_type": "unanswerable"},
            {"case_id": "c", "unanswerable": True, "query_type": "unanswerable"},
        ]

        filtered = filter_cases(cases, keep_unanswerable_only=True)

        self.assertEqual([item["case_id"] for item in filtered], ["b", "c"])

    def test_filter_cases_can_exclude_version_sensitive_query_types(self):
        cases = [
            {"case_id": "a", "unanswerable": False, "query_type": "current_policy"},
            {"case_id": "b", "unanswerable": False, "query_type": "history_version"},
            {"case_id": "c", "unanswerable": False, "query_type": "time_based_policy"},
            {"case_id": "d", "unanswerable": False, "query_type": "exception_rule"},
        ]

        filtered = filter_cases(
            cases,
            exclude_query_types={"history_version", "time_based_policy"},
        )

        self.assertEqual([item["case_id"] for item in filtered], ["a", "d"])


if __name__ == "__main__":
    unittest.main()
