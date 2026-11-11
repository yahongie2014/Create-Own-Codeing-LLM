import json
import tempfile
import threading
import unittest
from pathlib import Path
import urllib.request
from unittest.mock import patch
from http.server import ThreadingHTTPServer

from coder79.agent import WorkspaceTools
from coder79.ide_server import (
    create_chat_completion,
    handle_mcp_message,
    openai_stream_chunks,
    serve_openai,
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
        with patch("coder79.ide_server.chat", return_value="Inspected the project.") as model_chat:
            response = create_chat_completion(
                {
                    "model": "coder79",
                    "messages": [{"role": "user", "content": "Explain the project."}],
                    "tools": [],
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
            "coder79.agent._request_ollama",
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

    def test_openai_endpoint_rejects_invalid_messages_and_preserves_continue_tools(self):
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
        tool_schema = {
            "type": "function",
            "function": {
                "name": "write_file",
                "description": "Write a file.",
                "parameters": {
                    "type": "object",
                    "properties": {"path": {"type": "string"}},
                },
            },
        }
        with patch(
            "coder79.ide_server._request_ollama",
            return_value={
                "message": {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {
                            "function": {
                                "name": "write_file",
                                "arguments": {"path": "src/app.py"},
                            }
                        }
                    ],
                }
            },
        ) as ollama_request:
            response = create_chat_completion(
                {
                    "messages": [
                        {"role": "user", "content": "Update this file."},
                        {
                            "role": "assistant",
                            "content": None,
                            "tool_calls": [
                                {
                                    "id": "call_previous",
                                    "type": "function",
                                    "function": {
                                        "name": "read_file",
                                        "arguments": '{"path":"src/app.py"}',
                                    },
                                }
                            ],
                        },
                        {
                            "role": "tool",
                            "tool_call_id": "call_previous",
                            "name": "read_file",
                            "content": "print('hello')",
                        },
                    ],
                    "tools": [tool_schema],
                },
                self.workspace,
                "coder79",
                "http://localhost:11434",
            )
        completion = response["choices"][0]
        self.assertEqual(completion["finish_reason"], "tool_calls")
        self.assertEqual(completion["message"]["tool_calls"][0]["function"]["name"], "write_file")
        self.assertEqual(
            json.loads(completion["message"]["tool_calls"][0]["function"]["arguments"]),
            {"path": "src/app.py"},
        )
        self.assertEqual(ollama_request.call_args.args[1], "coder79")
        self.assertEqual(ollama_request.call_args.kwargs["tools"], [tool_schema])
        messages = ollama_request.call_args.args[2]
        self.assertIn("currently open in the IDE", messages[0]["content"])
        self.assertIn("do not", messages[0]["content"])
        self.assertNotIn(str(self.workspace.root), messages[0]["content"])
        self.assertNotIn("Initial workspace tree", messages[0]["content"])
        self.assertEqual(
            messages[-2]["tool_calls"][0]["function"]["arguments"],
            {"path": "src/app.py"},
        )
        self.assertEqual(messages[-1], {"role": "tool", "content": "print('hello')", "tool_name": "read_file"})
        streamed = openai_stream_chunks(response)
        self.assertEqual(
            streamed[-1]["choices"][0]["finish_reason"], "tool_calls"
        )
        self.assertEqual(
            streamed[1]["choices"][0]["delta"]["tool_calls"][0]["function"]["name"],
            "write_file",
        )
        with self.assertRaisesRegex(ValueError, "function tool"):
            create_chat_completion(
                {
                    "messages": [{"role": "user", "content": "Hi"}],
                    "tools": [{"type": "computer"}],
                },
                self.workspace,
                "coder79",
                "http://localhost:11434",
            )
        with patch(
            "coder79.ide_server._request_ollama",
            return_value={
                "message": {
                    "role": "assistant",
                    "content": (
                        '{"name":"read_file","arguments":{"path":'
                        '{"type":"string","value":"src/app.py"}}}'
                    ),
                }
            },
        ):
            text_call_response = create_chat_completion(
                {
                    "messages": [{"role": "user", "content": "Read the file."}],
                    "tools": [
                        {
                            "type": "function",
                            "function": {
                                "name": "read_file",
                                "parameters": {
                                    "type": "object",
                                    "properties": {"path": {"type": "string"}},
                                },
                            },
                        }
                    ],
                },
                self.workspace,
                "coder79",
                "http://localhost:11434",
            )
        text_call = text_call_response["choices"][0]
        self.assertEqual(text_call["finish_reason"], "tool_calls")
        self.assertIsNone(text_call["message"]["content"])
        self.assertEqual(
            json.loads(
                text_call["message"]["tool_calls"][0]["function"]["arguments"]
            ),
            {"path": "src/app.py"},
        )
        with self.assertRaisesRegex(LookupError, "Model not found"):
            create_chat_completion(
                {"model": [], "messages": [{"role": "user", "content": "Hi"}]},
                self.workspace,
                "coder79",
                "http://localhost:11434",
            )

    def test_openai_health_route_responds_while_chat_request_is_running(self):
        server_ready = threading.Event()
        chat_started = threading.Event()
        release_chat = threading.Event()
        server_holder = []
        post_errors = []

        def create_server(address, handler):
            server = ThreadingHTTPServer(address, handler)
            server_holder.append(server)
            server_ready.set()
            return server

        def wait_for_chat(*_args):
            chat_started.set()
            if not release_chat.wait(timeout=5):
                raise TimeoutError("Test chat request was not released.")
            return "Completed."

        with (
            patch("coder79.ide_server.ThreadingHTTPServer", side_effect=create_server),
            patch("coder79.ide_server.chat", side_effect=wait_for_chat),
        ):
            server_thread = threading.Thread(
                target=serve_openai,
                args=(self.workspace, "coder79", "http://localhost:11434", 0, "127.0.0.1"),
                daemon=True,
            )
            server_thread.start()
            self.assertTrue(server_ready.wait(timeout=5))
            server = server_holder[0]
            base_url = f"http://127.0.0.1:{server.server_address[1]}"
            request = urllib.request.Request(
                f"{base_url}/v1/chat/completions",
                data=json.dumps(
                    {
                        "model": "coder79",
                        "messages": [{"role": "user", "content": "Hi"}],
                    }
                ).encode(),
                headers={"Content-Type": "application/json"},
            )

            def submit_chat():
                try:
                    with urllib.request.urlopen(request, timeout=5) as response:
                        response.read()
                except Exception as exc:
                    post_errors.append(exc)

            post_thread = threading.Thread(target=submit_chat)
            post_thread.start()
            try:
                self.assertTrue(chat_started.wait(timeout=5))
                with urllib.request.urlopen(f"{base_url}/v1/models", timeout=2) as response:
                    self.assertEqual(response.status, 200)
            finally:
                release_chat.set()
                post_thread.join(timeout=5)
                server.shutdown()
                server_thread.join(timeout=5)
                server.server_close()

        self.assertFalse(post_thread.is_alive())
        self.assertFalse(server_thread.is_alive())
        self.assertEqual(post_errors, [])

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
