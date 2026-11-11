import tempfile
import unittest
from pathlib import Path
import ssl
import urllib.parse
from unittest.mock import patch

from coder79.agent import WorkspaceTools, _request_public_url, _text_tool_calls, chat


class WorkspaceToolsTests(unittest.TestCase):
    def setUp(self):
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary_directory.name)
        self.tools = WorkspaceTools(self.root)

    def tearDown(self):
        self.temporary_directory.cleanup()

    def test_list_tree_shows_structure_and_skips_generated_folders(self):
        (self.root / "src").mkdir()
        (self.root / "src" / "app.py").write_text("print('hello')", encoding="utf-8")
        (self.root / "node_modules").mkdir()
        (self.root / "node_modules" / "package.js").write_text("", encoding="utf-8")

        tree = self.tools.list_tree(max_depth=3)

        self.assertIn("src/", tree)
        self.assertIn("app.py", tree)
        self.assertNotIn("node_modules", tree)

    def test_read_file_rejects_paths_outside_workspace(self):
        with self.assertRaisesRegex(ValueError, "outside"):
            self.tools.read_file("../outside.txt")

    def test_read_file_blocks_secrets_and_limits_output(self):
        (self.root / ".env").write_text("TOKEN=secret", encoding="utf-8")
        (self.root / "large.txt").write_text("abcdefghij", encoding="utf-8")

        with self.assertRaisesRegex(ValueError, "blocked"):
            self.tools.read_file(".env")
        self.assertTrue(self.tools.read_file("large.txt", max_chars=4).endswith("truncated)"))

    def test_search_files_finds_text_and_skips_secret_files(self):
        (self.root / "app.py").write_text("def greet(): pass", encoding="utf-8")
        (self.root / ".env").write_text("greet secret", encoding="utf-8")

        result = self.tools.search_files("GREET", "*.py")

        self.assertIn("app.py:1:", result)
        self.assertNotIn(".env", result)

    def test_fetch_url_extracts_static_page_structure_and_visible_content(self):
        parsed_url = urllib.parse.urlsplit("https://example.com/")
        markup = (
            "<html><head><title>Example site</title>"
            '<meta name="description" content="A sample page"></head>'
            '<body><nav><a href="/about">About us</a></nav><main><h1>Welcome</h1>'
            "<p>Public page content.</p><script>ignore this</script></main></body></html>"
        ).encode()
        with (
            patch.object(
                self.tools,
                "_validate_public_url",
                return_value=(parsed_url, "93.184.216.34"),
            ),
            patch(
                "coder79.agent._request_public_url",
                return_value=(200, {"content-type": "text/html; charset=utf-8"}, markup),
            ),
        ):
            result = self.tools.fetch_url("https://example.com/")

        self.assertIn("Title: Example site", result)
        self.assertIn("Description: A sample page", result)
        self.assertIn("Page structure: nav, main", result)
        self.assertIn("Headings: Welcome", result)
        self.assertIn("About us: https://example.com/about", result)
        self.assertIn("Public page content.", result)
        self.assertNotIn("ignore this", result)

    def test_fetch_url_validates_each_public_redirect(self):
        start = urllib.parse.urlsplit("https://example.com/")
        destination = urllib.parse.urlsplit("https://example.com/new")
        with (
            patch.object(
                self.tools,
                "_validate_public_url",
                side_effect=[(start, "93.184.216.34"), (destination, "93.184.216.34")],
            ) as validate,
            patch(
                "coder79.agent._request_public_url",
                side_effect=[
                    (302, {"location": "/new"}, b""),
                    (200, {"content-type": "text/plain; charset=utf-8"}, b"Loaded page"),
                ],
            ),
        ):
            result = self.tools.fetch_url("https://example.com/")

        self.assertIn("URL: https://example.com/new", result)
        self.assertIn("Loaded page", result)
        self.assertEqual(validate.call_count, 2)

    def test_fetch_url_rejects_non_public_or_non_web_targets(self):
        for url in (
            "file:///etc/passwd",
            "http://127.0.0.1/",
            "http://user:password@example.com/",
            "https://example.com:8443/",
        ):
            with self.subTest(url=url), self.assertRaises(ValueError):
                self.tools.fetch_url(url)

    def test_public_fetch_reports_certificate_validation_errors_without_disabling_tls(self):
        with patch("coder79.agent.http.client.HTTPSConnection") as https_connection:
            connection = https_connection.return_value
            connection.request.side_effect = ssl.SSLCertVerificationError(
                "certificate verify failed"
            )

            with self.assertRaisesRegex(ValueError, "TLS certificate validation failed"):
                _request_public_url(
                    urllib.parse.urlsplit("https://example.com/"),
                    "93.184.216.34",
                )

        connection.close.assert_called_once()

    def test_text_tool_call_parser_accepts_only_known_calls(self):
        response = (
            '<tools>{"type":"function","function":{"name":"list_tree"}}</tools>\n'
            '```json\n{"name":"list_tree","arguments":{"path":".","max_depth":2}}\n```\n'
            "A fabricated tree should not be returned as the answer."
        )

        calls = _text_tool_calls(response)

        self.assertEqual(calls[0]["function"]["name"], "list_tree")
        self.assertEqual(calls[0]["function"]["arguments"]["max_depth"], 2)
        json_only_call = _text_tool_calls(
            '{"name":"read_file","arguments":{"path":"README.md"}}'
        )
        self.assertEqual(json_only_call[0]["function"]["name"], "read_file")
        web_call = _text_tool_calls(
            '{"name":"fetch_url","arguments":{"url":"https://example.com/"}}'
        )
        self.assertEqual(web_call[0]["function"]["name"], "fetch_url")
        self.assertEqual(_text_tool_calls('{"name":"run_command","arguments":{}}'), [])
        self.assertEqual(_text_tool_calls('{"name":[],"arguments":{}}'), [])

    def test_text_tool_call_parser_accepts_continue_tool_schemas_and_unwraps_arguments(self):
        response = (
            '{"name":"greet","arguments":{"name":'
            '{"type":"string","value":"Ada"}}}'
        )

        calls = _text_tool_calls(response, {"greet"})

        self.assertEqual(
            calls,
            [{"function": {"name": "greet", "arguments": {"name": "Ada"}}}],
        )

    def test_chat_uses_actual_tool_result_instead_of_textual_fake_result(self):
        (self.root / "actual.txt").write_text("workspace evidence", encoding="utf-8")
        responses = [
            {
                "message": {
                    "role": "assistant",
                    "content": (
                        '</tools>\n```json\n{"name":"list_tree","arguments":{"path":".",'
                        '"max_depth":1}}\n```\n fake folder tree'
                    ),
                }
            },
            {"message": {"role": "assistant", "content": "The project has actual.txt."}},
        ]
        with         patch("coder79.agent._request_ollama", side_effect=responses):
            messages = [{"role": "user", "content": "Inspect the project."}]
            answer = chat(self.tools, "coder79", "http://localhost:11434", messages)

        self.assertEqual(answer, "The project has actual.txt.")
        self.assertIn("actual.txt", messages[-1]["content"])
        self.assertNotIn("fake folder tree", str(messages))

    def test_chat_retries_once_when_model_returns_empty_after_tool_use(self):
        responses = [
            {
                "message": {
                    "role": "assistant",
                    "content": '{"name":"list_tree","arguments":{"path":".","max_depth":1}}',
                }
            },
            {"message": {"role": "assistant", "content": ""}},
            {"message": {"role": "assistant", "content": "The workspace is ready."}},
        ]
        with patch("coder79.agent._request_ollama", side_effect=responses):
            messages = [{"role": "user", "content": "Inspect the project."}]
            answer = chat(self.tools, "coder79", "http://localhost:11434", messages)

        self.assertEqual(answer, "The workspace is ready.")
        self.assertIn("tool results", messages[-2]["content"])

    def test_chat_uses_actual_web_tool_result(self):
        responses = [
            {
                "message": {
                    "role": "assistant",
                    "content": (
                        '{"name":"fetch_url","arguments":'
                        '{"url":"https://example.com/"}}'
                    ),
                }
            },
            {"message": {"role": "assistant", "content": "The page is titled Example."}},
        ]
        with (
            patch("coder79.agent._request_ollama", side_effect=responses),
            patch.object(self.tools, "fetch_url", return_value="Title: Example"),
        ):
            messages = [
                {"role": "user", "content": "Analyze https://example.com/"}
            ]
            answer = chat(self.tools, "coder79", "http://localhost:11434", messages)

        self.assertEqual(answer, "The page is titled Example.")
        self.assertIn("Web tool result from fetch_url:", str(messages))
        self.assertIn("Title: Example", str(messages))


if __name__ == "__main__":
    unittest.main()
