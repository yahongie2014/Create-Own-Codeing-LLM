import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from coder79_agent import WorkspaceTools
from coder79_ide_server import (
    create_chat_completion,
    handle_mcp_message,
    openai_stream_chunks,
)


class IdeServerTests(unittest.TestCase):
    def setUp(self):
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary_directory.name)
        (self.root / "src").mkdir()
        (self.root / "src" / "app.py").write_text("print('hello')", encoding="utf-8")
        self.workspace = WorkspaceTools(self.root)

    def tearDown(self):
        self.temporary_directory.cleanup()

    def test_mcp_initialize_advertises_tools_and_workspace_instructions(self):
        response = handle_mcp_message(
            {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "initialize",
                "params": {"protocolVersion": "2025-03-26"},
            },
            self.workspace,
            "coder79",
            "http://localhost:11434",
        )

        self.assertEqual(response["result"]["protocolVersion"], "2025-03-26")
        self.assertEqual(response["result"]["serverInfo"]["name"], "coder79")
        self.assertIn(str(self.workspace.root), response["result"]["instructions"])

    def test_mcp_lists_tools_and_restricts_tool_calls_to_workspace(self):
        tools_response = handle_mcp_message(
            {"jsonrpc": "2.0", "id": 2, "method": "tools/list"},
            self.workspace,
            "coder79",
            "http://localhost:11434",
        )
        call_response = handle_mcp_message(
            {
                "jsonrpc": "2.0",
                "id": 3,
                "method": "tools/call",
                "params": {"name": "read_file", "arguments": {"path": "src/app.py"}},
            },
            self.workspace,
            "coder79",
            "http://localhost:11434",
        )

        self.assertEqual(
            [tool["name"] for tool in tools_response["result"]["tools"]],
            ["list_tree", "read_file", "search_files", "fetch_url"],
        )
        self.assertIn("print('hello')", call_response["result"]["content"][0]["text"])
        self.assertFalse(call_response["result"]["isError"])

    def test_mcp_fetch_url_tool_is_available(self):
        with patch.object(self.workspace, "fetch_url", return_value="Title: Example"):
            response = handle_mcp_message(
                {
                    "jsonrpc": "2.0",
                    "id": 6,
                    "method": "tools/call",
                    "params": {
                        "name": "fetch_url",
                        "arguments": {"url": "https://example.com/"},
                    },
                },
                self.workspace,
                "coder79",
                "http://localhost:11434",
            )

        self.assertIn("Title: Example", response["result"]["content"][0]["text"])
        self.assertFalse(response["result"]["isError"])

    def test_mcp_notifications_have_no_response(self):
        response = handle_mcp_message(
            {"jsonrpc": "2.0", "method": "notifications/initialized"},
            self.workspace,
            "coder79",
            "http://localhost:11434",
        )
        self.assertIsNone(response)

    def test_mcp_rejects_malformed_requests_and_negotiates_supported_versions(self):
        malformed = handle_mcp_message(
            {"jsonrpc": "2.0", "id": 4, "method": "tools/call", "params": {"name": "read_file", "arguments": []}},
            self.workspace,
            "coder79",
            "http://localhost:11434",
        )
        unsupported_version = handle_mcp_message(
            {
                "jsonrpc": "2.0",
                "id": 5,
                "method": "initialize",
                "params": {"protocolVersion": "future-version"},
            },
            self.workspace,
            "coder79",
            "http://localhost:11434",
        )

        self.assertEqual(malformed["error"]["code"], -32602)
        self.assertEqual(
            unsupported_version["result"]["protocolVersion"], "2025-03-26"
        )

    def test_openai_chat_completion_uses_workspace_context(self):
        with patch("coder79_ide_server.chat", return_value="Inspected the project.") as model_chat:
            response = create_chat_completion(
                {
                    "model": "coder79",
                    "messages": [{"role": "user", "content": "Explain the project."}],
                },
                self.workspace,
                "coder79",
                "http://localhost:11434",
            )

        self.assertEqual(response["choices"][0]["message"]["content"], "Inspected the project.")
        prompt = model_chat.call_args.args[3]
        self.assertIn("src/", prompt[0]["content"])
        self.assertEqual(prompt[-1], {"role": "user", "content": "Explain the project."})

    def test_openai_chat_completion_runs_the_ollama_agent_loop(self):
        with patch(
            "coder79_agent._request_ollama",
            return_value={"message": {"role": "assistant", "content": "Found the entry point."}},
        ) as ollama_request:
            response = create_chat_completion(
                {
                    "model": "coder79",
                    "messages": [{"role": "user", "content": "Find the entry point."}],
                },
                self.workspace,
                "coder79",
                "http://localhost:11434",
            )

        self.assertEqual(
            response["choices"][0]["message"]["content"], "Found the entry point."
        )
        self.assertEqual(ollama_request.call_args.args[1], "coder79")
        self.assertIn("workspace", ollama_request.call_args.args[2][0]["content"])

    def test_openai_endpoint_rejects_non_text_and_unsupported_tools(self):
        with self.assertRaisesRegex(ValueError, "text messages"):
            create_chat_completion(
                {
                    "messages": [
                        {
                            "role": "user",
                            "content": [{"type": "image_url", "image_url": {"url": "data:"}}],
                        }
                    ]
                },
                self.workspace,
                "coder79",
                "http://localhost:11434",
            )
        with self.assertRaisesRegex(ValueError, "unsupported role"):
            create_chat_completion(
                {"messages": [{"role": [], "content": "Hi"}]},
                self.workspace,
                "coder79",
                "http://localhost:11434",
            )
        with self.assertRaisesRegex(ValueError, "function calling"):
            create_chat_completion(
                {"messages": [{"role": "user", "content": "Hi"}], "tools": [{"type": "function"}]},
                self.workspace,
                "coder79",
                "http://localhost:11434",
            )
        with self.assertRaisesRegex(LookupError, "Model not found"):
            create_chat_completion(
                {"model": [], "messages": [{"role": "user", "content": "Hi"}]},
                self.workspace,
                "coder79",
                "http://localhost:11434",
            )

    def test_openai_stream_chunks_follow_chat_completion_format(self):
        chunks = openai_stream_chunks(
            {
                "id": "chatcmpl-coder79",
                "object": "chat.completion",
                "created": 0,
                "model": "coder79",
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": "A concise answer."},
                        "finish_reason": "stop",
                    }
                ],
            }
        )

        self.assertEqual(chunks[0]["choices"][0]["delta"], {"role": "assistant"})
        self.assertEqual(
            "".join(chunk["choices"][0]["delta"].get("content", "") for chunk in chunks),
            "A concise answer.",
        )
        self.assertEqual(chunks[-1]["choices"][0]["finish_reason"], "stop")


if __name__ == "__main__":
    unittest.main()
