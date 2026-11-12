import tempfile
import unittest
from pathlib import Path

from coder79.skills import MAX_SKILL_CHARS, SkillLibrary


class SkillLibraryTests(unittest.TestCase):
    def setUp(self):
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary_directory.name)
        self.skill_path = self.root / "11-evaluation" / "benchmarking"
        self.skill_path.mkdir(parents=True)
        (self.skill_path / "SKILL.md").write_text(
            "---\n"
            "name: benchmark-evaluation\n"
            "description: Evaluate language models with reproducible benchmarks.\n"
            "tags: [evaluation, benchmark]\n"
            "---\n\n"
            "# Benchmark evaluation\n"
            "Use held-out data.\n",
            encoding="utf-8",
        )
        self.library = SkillLibrary(self.root)

    def tearDown(self):
        self.temporary_directory.cleanup()

    def test_search_returns_matching_skill_metadata_and_id(self):
        result = self.library.search("reproducible model benchmarks")

        self.assertIn("Skill ID: 11-evaluation/benchmarking", result)
        self.assertIn("Name: benchmark-evaluation", result)
        self.assertIn("reproducible benchmarks", result)

    def test_read_loads_only_the_requested_skill(self):
        result = self.library.read("11-evaluation/benchmarking")

        self.assertIn("# Benchmark evaluation", result)
        self.assertIn("Use held-out data.", result)

    def test_read_rejects_unknown_skill_ids(self):
        with self.assertRaisesRegex(ValueError, "Unknown skill ID"):
            self.library.read("../outside")

    def test_best_match_returns_only_an_indexed_skill_id(self):
        self.assertEqual(
            self.library.best_match_id("reproducible model benchmarks"),
            "11-evaluation/benchmarking",
        )
        self.assertIsNone(self.library.best_match_id("unrelated culinary topic"))

    def test_training_conversations_use_concise_metadata_not_full_skill_documents(self):
        conversations = self.library.training_conversations()

        self.assertEqual(len(conversations), 1)
        messages = conversations[0]["messages"]
        self.assertEqual([message["role"] for message in messages], ["user", "assistant"])
        self.assertIn("benchmark-evaluation", messages[0]["content"])
        self.assertEqual(
            messages[1]["content"],
            "Evaluate language models with reproducible benchmarks.",
        )
        self.assertNotIn("# Benchmark evaluation", messages[1]["content"])

    def test_read_truncates_oversized_skill_content(self):
        (self.skill_path / "SKILL.md").write_text(
            "---\nname: benchmark-evaluation\ndescription: benchmark\n---\n"
            + ("x" * (MAX_SKILL_CHARS + 1)),
            encoding="utf-8",
        )
        library = SkillLibrary(self.root)

        result = library.read("11-evaluation/benchmarking")

        self.assertIn("skill content truncated", result)
        self.assertLessEqual(len(result), MAX_SKILL_CHARS + 100)


if __name__ == "__main__":
    unittest.main()
