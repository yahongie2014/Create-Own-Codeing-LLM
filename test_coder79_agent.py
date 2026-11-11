import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from coder79_agent import WorkspaceTools, _text_tool_calls, chat


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
        self.assertEqual(_text_tool_calls('{"name":"run_command","arguments":{}}'), [])
        self.assertEqual(_text_tool_calls('{"name":[],"arguments":{}}'), [])

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
        with patch("coder79_agent._request_ollama", side_effect=responses):
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
        with patch("coder79_agent._request_ollama", side_effect=responses):
            messages = [{"role": "user", "content": "Inspect the project."}]
            answer = chat(self.tools, "coder79", "http://localhost:11434", messages)

        self.assertEqual(answer, "The workspace is ready.")
        self.assertIn("tool results", messages[-2]["content"])


if __name__ == "__main__":
    unittest.main()
