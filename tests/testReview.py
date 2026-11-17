import contextlib
import io
import json
import os
import subprocess
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import httpx
from openai import OpenAI

import review


SAMPLE = Path(__file__).resolve().parents[1] / "examples" / "buggy.patch"
ROOT = Path(__file__).resolve().parents[1]


class DiffTests(unittest.TestCase):
    def test_env_setting_reads_only_requested_key(self):
        with mock.patch.dict(os.environ, {}, clear=True):
            with mock.patch.object(review.Path, "read_text", return_value="OPENAI_API_KEY=sk-test\nGITHUB_TOKEN=gh-test\n"):
                self.assertEqual(review.get_setting("GITHUB_TOKEN"), "gh-test")
                self.assertEqual(review.get_setting("OPENAI_API_KEY"), "sk-test")

    def test_sample_changed_lines(self):
        info = review.parse_diff(SAMPLE.read_text())
        self.assertEqual(info.changed_lines["examples/sampleProject/orders.py"], {1, 3, 4, 6, 7})

    def test_deletion_has_new_file_anchor(self):
        patch = "--- a/a.py\n+++ b/a.py\n@@ -1,2 +1 @@\n-old\n keep\n"
        self.assertEqual(review.parse_diff(patch).changed_lines["a.py"], {1})

    def test_added_text_that_looks_like_patch_header(self):
        patch = "--- a/a.txt\n+++ b/a.txt\n@@ -0,0 +1 @@\n+++ b/other.txt\n"
        self.assertEqual(review.parse_diff(patch).changed_lines, {"a.txt": {1}})

    def test_read_file_rejects_escape_and_long_range(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp) / "project"
            root.mkdir()
            (Path(temp) / "secret.txt").write_text("secret")
            info = review.DiffInfo({"../secret.txt": {1}, "ok.txt": {1}})
            (root / "ok.txt").write_text("safe")
            self.assertIn("outside", review.read_file(root, info, "../secret.txt", 1, 1)["error"])
            self.assertIn("at most", review.read_file(root, info, "ok.txt", 1, 100)["error"])

    def test_base_reads_a_real_git_diff(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            subprocess.run(["git", "init", "-q"], cwd=root, check=True)
            source = root / "app.py"
            source.write_text("value = 1\n")
            subprocess.run(["git", "add", "app.py"], cwd=root, check=True)
            subprocess.run(
                ["git", "-c", "user.name=Test", "-c", "user.email=test@example.com", "commit", "-qm", "initial"],
                cwd=root, check=True,
            )
            source.write_text("value = 2\n")
            args = SimpleNamespace(diff=None, base="HEAD")
            diff = review.load_diff(args, root)
            self.assertEqual(review.parse_diff(diff).changed_lines["app.py"], {1})

    def test_pr_url_fetches_diff_without_a_clone(self):
        seen = []

        def handler(request):
            seen.append(request)
            return httpx.Response(200, headers={"Content-Type": "text/plain"}, content=SAMPLE.read_bytes())

        fake_client = httpx.Client(transport=httpx.MockTransport(handler))
        with mock.patch.object(review.httpx, "Client", return_value=fake_client):
            with mock.patch.dict(os.environ, {"GITHUB_TOKEN": "test-token"}):
                diff = review.fetch_pr_diff("https://github.com/example/project/pull/12")
        self.assertEqual(review.parse_diff(diff).files, ["examples/sampleProject/orders.py"])
        self.assertEqual(str(seen[0].url), "https://api.github.com/repos/example/project/pulls/12")
        self.assertEqual(seen[0].headers["accept"], "application/vnd.github.diff")
        self.assertEqual(seen[0].headers["authorization"], "Bearer test-token")

    def test_pr_url_rejects_other_hosts(self):
        with self.assertRaises(review.ReviewError):
            review.fetch_pr_diff("https://github.com.evil.example/a/b/pull/1")


class FakeResponses:
    def __init__(self, replies):
        self.replies = iter(replies)
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return next(self.replies)


class AgentTests(unittest.TestCase):
    def test_post_pr_review_sends_one_comment_review(self):
        requests = []

        def handler(request):
            requests.append(request)
            return httpx.Response(200, json={"html_url": "https://github.com/example/project/pull/12#review-1"})

        fake_client = httpx.Client(transport=httpx.MockTransport(handler))
        finding = {
            "category": "bug", "severity": "high", "file": "app.py", "line": 3,
            "explanation": "A failure condition.", "suggestion": "Fix it.",
        }
        with mock.patch.object(review, "get_setting", return_value="gh-test"):
            with mock.patch.object(review.httpx, "Client", return_value=fake_client):
                url = review.post_pr_review(
                    "https://github.com/example/project/pull/12",
                    {"summary": "Review summary", "findings": [finding]},
                )
        self.assertEqual(url, "https://github.com/example/project/pull/12#review-1")
        self.assertEqual(requests[0].method, "POST")
        self.assertEqual(str(requests[0].url), "https://api.github.com/repos/example/project/pulls/12/reviews")
        body = json.loads(requests[0].content)
        self.assertEqual(body["event"], "COMMENT")
        self.assertIn("app.py:3", body["body"])

    def test_post_requires_pr(self):
        error = io.StringIO()
        with contextlib.redirect_stderr(error):
            code = review.main(["--post", "--diff", str(SAMPLE)])
        self.assertEqual(code, 1)
        self.assertIn("--post requires --pr", error.getvalue())

    def test_cli_prints_findings_before_posting(self):
        output = io.StringIO()
        result = {"summary": "Review summary", "findings": [{
            "category": "bug", "severity": "high", "file": "app.py", "line": 3,
            "explanation": "A failure condition.", "suggestion": "Fix it.",
        }]}

        def post_after_print(url, review_result):
            self.assertIn("Review summary", output.getvalue())
            self.assertIn("app.py:3", output.getvalue())
            return "https://github.com/example/project/pull/12#review-1"

        with mock.patch.object(review, "load_diff", return_value=SAMPLE.read_text()):
            with mock.patch.object(review, "get_api_key", return_value="test-key"):
                with mock.patch.object(review, "review_diff", return_value=result):
                    with mock.patch.object(review, "post_pr_review", side_effect=post_after_print):
                        with contextlib.redirect_stdout(output):
                            code = review.main(["--pr", "https://github.com/example/project/pull/12", "--post"])
        self.assertEqual(code, 0)
        self.assertIn("Posted review:", output.getvalue())

    def test_agent_reads_context_then_returns_review(self):
        path = "examples/sampleProject/orders.py"
        tool_call = SimpleNamespace(
            type="function_call",
            name="read_file",
            arguments=json.dumps({"path": path, "start_line": 1, "end_line": 7}),
            call_id="call_1",
        )
        finding = {
            "category": "bug",
            "severity": "high",
            "file": path,
            "line": 3,
            "explanation": "Page 1 skips the first ten orders.",
            "suggestion": "Use (page - 1) * 10.",
        }
        fake = FakeResponses([
            SimpleNamespace(status="completed", output=[tool_call], output_text=""),
            SimpleNamespace(status="completed", output=[], output_text=json.dumps({"summary": "Pagination changed.", "findings": [finding]})),
        ])
        client = SimpleNamespace(responses=fake)
        diff = SAMPLE.read_text()
        result = review.review_diff(client, diff, review.parse_diff(diff), ROOT, "test-model")
        self.assertEqual(result["findings"], [finding])
        self.assertEqual(len(fake.calls), 2)
        tool_result = fake.calls[1]["input"][-1]
        self.assertEqual(tool_result["type"], "function_call_output")
        self.assertIn("3:     start = page * 10", json.loads(tool_result["output"])["content"])

    def test_unrelated_location_is_removed(self):
        info = review.DiffInfo({"file.py": {3}})
        finding = {
            "category": "bug", "severity": "high", "file": "file.py", "line": 99,
            "explanation": "Bad location", "suggestion": "Fix it",
        }
        with contextlib.redirect_stderr(io.StringIO()):
            result = review.validate_review({"summary": "Review", "findings": [finding]}, info)
        self.assertEqual(result["findings"], [])

    def test_empty_diff_does_not_need_api_key(self):
        with tempfile.TemporaryDirectory() as temp:
            patch = Path(temp) / "empty.patch"
            patch.write_text("")
            output = io.StringIO()
            with contextlib.redirect_stdout(output):
                code = review.main(["--diff", str(patch), "--json"])
            self.assertEqual(code, 0)
            self.assertEqual(json.loads(output.getvalue())["findings"], [])

    def test_bad_patch_reports_an_error(self):
        with tempfile.TemporaryDirectory() as temp:
            patch = Path(temp) / "bad.patch"
            patch.write_text("not a diff\n")
            error = io.StringIO()
            with contextlib.redirect_stderr(error):
                code = review.main(["--diff", str(patch)])
            self.assertEqual(code, 1)
            self.assertIn("No reviewable text changes", error.getvalue())

    def test_real_sdk_serializes_tool_loop(self):
        path = "examples/sampleProject/orders.py"
        requests = []

        def handler(request):
            body = json.loads(request.content)
            requests.append(body)
            if len(requests) == 1:
                output = [{
                    "id": "fc_1", "type": "function_call", "status": "completed",
                    "call_id": "call_1", "name": "read_file",
                    "arguments": json.dumps({"path": path, "start_line": 1, "end_line": 7}),
                }]
            else:
                result = {"summary": "Order pagination changed.", "findings": []}
                output = [{
                    "id": "msg_1", "type": "message", "status": "completed", "role": "assistant",
                    "content": [{"type": "output_text", "text": json.dumps(result), "annotations": []}],
                }]
            return httpx.Response(200, json={
                "id": f"resp_{len(requests)}", "object": "response", "created_at": 0,
                "model": "gpt-6.1-sol", "status": "completed", "output": output,
            })

        http_client = httpx.Client(transport=httpx.MockTransport(handler))
        client = OpenAI(api_key="test-key", http_client=http_client)
        try:
            diff = SAMPLE.read_text()
            result = review.review_diff(client, diff, review.parse_diff(diff), ROOT, "gpt-6.1-sol")
        finally:
            client.close()
        self.assertEqual(result["summary"], "Order pagination changed.")
        self.assertEqual(len(requests), 2)
        self.assertEqual(requests[0]["text"]["format"]["type"], "json_schema")
        self.assertEqual(requests[1]["input"][-1]["type"], "function_call_output")

    def test_remote_review_does_not_offer_local_file_tool(self):
        fake = FakeResponses([
            SimpleNamespace(status="completed", output=[], output_text=json.dumps({"summary": "Done", "findings": []})),
        ])
        diff = SAMPLE.read_text()
        review.review_diff(SimpleNamespace(responses=fake), diff, review.parse_diff(diff), ROOT, "test-model", allow_file_reads=False)
        self.assertEqual(fake.calls[0]["tools"], [])
        self.assertEqual(fake.calls[0]["tool_choice"], "none")


if __name__ == "__main__":
    unittest.main()
