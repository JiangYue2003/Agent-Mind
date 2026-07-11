import hashlib
import json
import pathlib
import tempfile
import unittest

from evaluation.retrieval_metrics import load_gold_chunk_hashes, load_gold_chunk_keys
from evaluation.retrieval_runner import (
    collect_retrieval_records,
    load_retrieval_cases,
    run_retrieval_evaluation,
)


class _FakeResponse:
    def __init__(self, payload):
        self._payload = payload

    def raise_for_status(self):
        return None

    def json(self):
        return self._payload


class _FakeClient:
    def __init__(self):
        self.calls = []

    def post(self, path, params=None, json=None):
        self.calls.append({"method": "POST", "path": path, "params": params, "json": json})
        return _FakeResponse({
            "rerank_applied": True,
            "rerank_providers": ["tei"],
            "retrieval_debug": {
                "raw_candidates": [
                    {"chunk_key": "doc:parent:0:0", "rank": 1, "score": 0.9},
                    {"chunk_key": "noise:parent:0:0", "rank": 2, "score": 0.1},
                ],
                "reranked_candidates": [
                    {"chunk_key": "doc:parent:0:0", "rank": 1, "score": 0.99},
                ],
            },
        })

    def get(self, path, params=None):
        self.calls.append({"method": "GET", "path": path, "params": params, "json": None})
        return _FakeResponse({
            "items": [{
                "chunk_key": "doc:parent:0:0",
                "content": "明确的知识库证据。",
                "title": "规则说明",
            }],
        })


class RetrievalRunnerTests(unittest.TestCase):
    def test_load_retrieval_cases_requires_evidence_only_for_answerable_cases(self):
        answerable = {
            "case_id": "search-001",
            "query": "规则是什么？",
            "expected_answerable": True,
            "case_type": "single_source",
            "category": "direct",
            "difficulty": "easy",
            "source_file": "规则说明.md",
            "target_fact": "规则是明确的。",
            "evidence_text": "明确的知识库证据。",
        }
        invalid = dict(answerable)
        invalid.pop("evidence_text")
        ood = {
            "case_id": "search-ood-001",
            "query": "知识库没有的规则是什么？",
            "expected_answerable": False,
            "case_type": "ood",
            "category": "ood",
            "difficulty": "medium",
        }

        with tempfile.TemporaryDirectory() as temp_dir:
            path = pathlib.Path(temp_dir) / "cases.jsonl"
            path.write_text(
                "\n".join(json.dumps(item, ensure_ascii=False) for item in [answerable, invalid, ood]),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "evidence_text"):
                load_retrieval_cases(path)

            path.write_text(
                "\n".join(json.dumps(item, ensure_ascii=False) for item in [answerable, ood]),
                encoding="utf-8",
            )
            self.assertEqual(load_retrieval_cases(path), [answerable, ood])

    def test_collect_retrieval_records_calls_search_only(self):
        cases = [{
            "case_id": "search-001",
            "query": "规则是什么？",
            "expected_answerable": True,
            "case_type": "single_source",
            "category": "direct",
            "difficulty": "easy",
            "source_file": "规则说明.md",
            "target_fact": "规则是明确的。",
            "evidence_text": "明确的知识库证据。",
        }]
        client = _FakeClient()

        records = collect_retrieval_records(
            cases,
            client=client,
            top_k=5,
            recall_k=20,
            require_rerank=True,
            required_rerank_provider="tei",
        )

        self.assertEqual(len(records), 1)
        self.assertEqual(records[0]["metadata"]["case_id"], "search-001")
        self.assertEqual(records[0]["search"]["raw_candidates"][0]["chunk_key"], "doc:parent:0:0")
        self.assertEqual(client.calls, [{
            "method": "POST",
            "path": "/search",
            "params": {"query": "规则是什么？", "top_k": 5, "recall_k": 20, "include_debug": True},
            "json": None,
        }])

    def test_run_retrieval_evaluation_persists_deterministic_metrics(self):
        case = {
            "case_id": "search-001",
            "query": "规则是什么？",
            "expected_answerable": True,
            "case_type": "single_source",
            "category": "direct",
            "difficulty": "easy",
            "source_file": "规则说明.md",
            "target_fact": "规则是明确的。",
            "evidence_text": "明确的知识库证据。",
        }
        key = "doc:parent:0:0"
        client = _FakeClient()

        with tempfile.TemporaryDirectory() as temp_dir:
            root = pathlib.Path(temp_dir)
            dataset_path = root / "cases.jsonl"
            gold_path = root / "gold.json"
            dataset_path.write_text(json.dumps(case, ensure_ascii=False) + "\n", encoding="utf-8")
            gold_path.write_text(json.dumps({
                "case_gold_chunk_keys": {case["case_id"]: [key]},
                "chunk_content_sha256": {key: hashlib.sha256("明确的知识库证据。".encode("utf-8")).hexdigest()},
            }), encoding="utf-8")

            report = run_retrieval_evaluation(
                dataset_path=dataset_path,
                gold_path=gold_path,
                output_dir=root / "output",
                client=client,
                run_id="search-run",
                top_k=5,
                recall_k=20,
                require_rerank=True,
                required_rerank_provider="tei",
            )
            saved = json.loads((root / "output" / "search-run.json").read_text(encoding="utf-8"))

        self.assertEqual(report["status"], "completed")
        self.assertEqual(saved["retrieval"]["raw_k"], 20)
        self.assertEqual(saved["retrieval"]["final_k"], 5)
        self.assertEqual(saved["retrieval"]["summary"]["raw"]["hit_at_k"], 1.0)
        self.assertEqual(saved["summary"], {"total_cases": 1, "answerable_cases": 1, "unanswerable_cases": 0})

    def test_search_dataset_has_independent_70_answerable_and_10_ood_cases(self):
        root = pathlib.Path(__file__).resolve().parents[1]
        cases = load_retrieval_cases(root / "evaluation" / "datasets" / "customer_service_search_v1.jsonl")
        answerable = [case for case in cases if case["expected_answerable"]]
        ood = [case for case in cases if not case["expected_answerable"]]
        old_cases = [json.loads(line) for line in (root / "evaluation" / "datasets" / "customer_service_v1.jsonl").read_text(encoding="utf-8").splitlines() if line]

        self.assertEqual(len(cases), 80)
        self.assertEqual(len(answerable), 70)
        self.assertEqual(len(ood), 10)
        self.assertEqual(sum(case["case_type"] == "single_source" for case in answerable), 60)
        self.assertEqual(sum(case["case_type"] == "cross_topic_confusion" for case in answerable), 10)
        self.assertTrue(all(case["case_type"] == "ood" for case in ood))
        self.assertFalse({case["query"] for case in cases} & {case["user_input"] for case in old_cases})

    def test_search_gold_covers_every_answerable_case_and_hashes_every_key(self):
        root = pathlib.Path(__file__).resolve().parents[1]
        cases = load_retrieval_cases(root / "evaluation" / "datasets" / "customer_service_search_v1.jsonl")
        gold_path = root / "evaluation" / "datasets" / "customer_service_search_gold_v2.json"
        gold_chunk_keys = load_gold_chunk_keys(gold_path)
        gold_chunk_hashes = load_gold_chunk_hashes(gold_path)
        answerable_ids = {case["case_id"] for case in cases if case["expected_answerable"]}
        required_keys = {key for keys in gold_chunk_keys.values() for key in keys}

        self.assertEqual(set(gold_chunk_keys), answerable_ids)
        self.assertEqual(set(gold_chunk_hashes), required_keys)


if __name__ == "__main__":
    unittest.main()
