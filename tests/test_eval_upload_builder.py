import json
import pathlib
import tempfile
import unittest

from tools.upload_eval_knowledge import build_documents_from_metadata


class EvalUploadBuilderTests(unittest.TestCase):
    def test_build_documents_from_metadata_merges_md_content_and_metadata(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            base = pathlib.Path(temp_dir)
            (base / "请假制度-v1.md").write_text("# 请假制度\nV1 内容", encoding="utf-8")
            metadata = {
                "files": [],
                "metadata": [
                    {
                        "filename": "请假制度-v1.md",
                        "title": "请假制度",
                        "policy_id": "policy_leave",
                        "version_no": "V1",
                        "effective_at": "2023-09-01",
                        "issue_code": "人资〔2023〕18号",
                        "scope_key": "",
                    }
                ],
            }
            (base / "metadata.json").write_text(json.dumps(metadata, ensure_ascii=False), encoding="utf-8")

            documents = build_documents_from_metadata(base / "metadata.json")

            self.assertEqual(len(documents), 1)
            self.assertEqual(documents[0]["title"], "请假制度")
            self.assertEqual(documents[0]["policy_id"], "policy_leave")
            self.assertEqual(documents[0]["version_no"], "V1")
            self.assertEqual(documents[0]["effective_at"], "2023-09-01")
            self.assertIn("V1 内容", documents[0]["content"])

    def test_build_documents_from_metadata_raises_when_md_missing(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            base = pathlib.Path(temp_dir)
            metadata = {
                "files": [],
                "metadata": [
                    {
                        "filename": "缺失文件-v1.md",
                        "title": "缺失文件",
                        "policy_id": "policy_missing",
                        "version_no": "V1",
                        "effective_at": "2024-01-01",
                        "issue_code": "综合〔2024〕01号",
                        "scope_key": "",
                    }
                ],
            }
            (base / "metadata.json").write_text(json.dumps(metadata, ensure_ascii=False), encoding="utf-8")

            with self.assertRaisesRegex(FileNotFoundError, "缺失文件-v1.md"):
                build_documents_from_metadata(base / "metadata.json")


if __name__ == "__main__":
    unittest.main()
