import json
import tempfile
import unittest
from pathlib import Path

from coder79.train import load_training_conversations, tokenize_conversations


class TrainingDataTests(unittest.TestCase):
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
