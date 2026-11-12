import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from coder79.train import (
    default_llama_cpp_dir,
    export_to_gguf,
    load_training_conversations,
    tokenize_conversations,
    train_tokenizer,
)


class TrainingDataTests(unittest.TestCase):
    def test_export_to_gguf_uses_local_llama_cpp_converter(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            llama_cpp_dir = root / "llama.cpp"
            llama_cpp_dir.mkdir()
            (llama_cpp_dir / "convert_hf_to_gguf.py").write_text(
                "import pathlib, sys\n"
                "args = sys.argv\n"
                "output = pathlib.Path(args[args.index('--outfile') + 1])\n"
                "output.write_bytes(b'GGUF')\n",
                encoding="utf-8",
            )
            output_path = root / "models" / "coder79-scratch-f16.gguf"

            exported_path = export_to_gguf(
                root / "trained-model", output_path, llama_cpp_dir
            )

            self.assertEqual(exported_path, output_path.resolve())
            self.assertEqual(output_path.read_bytes(), b"GGUF")

    def test_export_to_gguf_reports_missing_converter(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            with self.assertRaisesRegex(FileNotFoundError, "converter not found"):
                export_to_gguf(
                    Path(temporary_directory) / "trained-model",
                    Path(temporary_directory) / "model.gguf",
                    Path(temporary_directory) / "missing-llama.cpp",
                )

    def test_default_llama_cpp_dir_uses_environment_override(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            with patch.dict("os.environ", {"LLAMA_CPP_DIR": temporary_directory}):
                self.assertEqual(
                    default_llama_cpp_dir(), Path(temporary_directory)
                )

    def test_train_tokenizer_learns_vocabulary_and_formats_chat(self):
        conversations = [
            {
                "messages": [
                    {"role": "user", "content": "Write a Python function."},
                    {"role": "assistant", "content": "def greet(): return 'hello'"},
                ]
            }
        ]

        with tempfile.TemporaryDirectory() as temporary_directory:
            tokenizer = train_tokenizer(
                conversations, Path(temporary_directory), vocab_size=300
            )
        rendered = tokenizer.apply_chat_template(
            conversations[0]["messages"],
            tokenize=False,
            add_generation_prompt=False,
        )

        self.assertIn("<|user|>", rendered)
        self.assertIn("<|assistant|>", rendered)
        self.assertGreater(len(tokenizer), 7)
        self.assertGreater(len(tokenizer(rendered)["input_ids"]), 0)
        unseen_text_ids = tokenizer("Καλημέρα 🚀")["input_ids"]
        self.assertNotIn(tokenizer.unk_token_id, unseen_text_ids)

    def test_load_training_conversations_includes_coding_profile_and_skill_guides(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            project_dir = Path(temporary_directory)
            data_dir = project_dir / "data"
            data_dir.mkdir()
            for filename in (
                "coding_dataset.jsonl",
                "coding_dataset_php.jsonl",
                "coding_dataset_advanced.jsonl",
            ):
                (data_dir / filename).write_text(
                    json.dumps({"messages": [{"role": "user", "content": filename}]})
                    + "\n",
                    encoding="utf-8",
                )
            (data_dir / "coding_skills.json").write_text(
                json.dumps(
                    {
                        "training_conversations": [
                            {"messages": [{"role": "user", "content": "profile"}]}
                        ]
                    }
                ),
                encoding="utf-8",
            )
            skill_path = (
                data_dir / "AI-Research-SKILLs" / "11-evaluation" / "benchmarking"
            )
            skill_path.mkdir(parents=True)
            (skill_path / "SKILL.md").write_text(
                "---\n"
                "name: benchmark-evaluation\n"
                "description: Evaluate models with reproducible benchmarks.\n"
                "---\n\n"
                "# Benchmark evaluation\n"
                "Keep a held-out test set.\n",
                encoding="utf-8",
            )

            conversations = load_training_conversations(project_dir)

        user_content = [
            message["content"]
            for conversation in conversations
            for message in conversation["messages"]
            if message["role"] == "user"
        ]
        assistant_content = [
            message["content"]
            for conversation in conversations
            for message in conversation["messages"]
            if message["role"] == "assistant"
        ]
        self.assertIn("coding_dataset.jsonl", user_content)
        self.assertIn("coding_dataset_php.jsonl", user_content)
        self.assertIn("coding_dataset_advanced.jsonl", user_content)
        self.assertIn("profile", user_content)
        self.assertTrue(
            any("benchmark-evaluation" in content for content in user_content)
        )
        self.assertTrue(
            any("# Benchmark evaluation" in content for content in assistant_content)
        )

    def test_tokenize_conversations_splits_long_sequences_without_dropping_tokens(self):
        class FakeTokenizer:
            def apply_chat_template(self, messages, tokenize, add_generation_prompt):
                return messages[0]["content"]

            def __call__(self, text, add_special_tokens, truncation):
                return {"input_ids": list(range(int(text)))}

        examples, split_count = tokenize_conversations(
            [{"messages": [{"role": "user", "content": "10"}]}],
            FakeTokenizer(),
            max_seq_length=4,
        )

        self.assertEqual(split_count, 1)
        self.assertEqual(
            [token for example in examples for token in example["input_ids"]],
            list(range(10)),
        )
        self.assertTrue(all(len(example["input_ids"]) <= 4 for example in examples))

    def test_tokenize_conversations_rejects_nonpositive_sequence_length(self):
        with self.assertRaisesRegex(ValueError, "must be positive"):
            tokenize_conversations([], tokenizer=object(), max_seq_length=0)


if __name__ == "__main__":
    unittest.main()
