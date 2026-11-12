"""On-demand search and loading for local Markdown skills."""

from __future__ import annotations

from dataclasses import dataclass
import os
import re
from pathlib import Path


MAX_SKILL_RESULTS = 8
MAX_SKILL_CHARS = 20_000
MAX_SKILL_TRAINING_CHARS = 3_000
_WORD_PATTERN = re.compile(r"[a-z0-9][a-z0-9+#.-]*", re.IGNORECASE)


@dataclass(frozen=True)
class Skill:
    identifier: str
    name: str
    description: str
    path: Path


class SkillLibrary:
    """Index SKILL.md metadata and load only the matching guide on request."""

    def __init__(self, root: Path | None = None) -> None:
        default_root = Path(__file__).resolve().parents[1] / "data" / "AI-Research-SKILLs"
        self.root = (root or default_root).expanduser().resolve()
        if not self.root.is_dir():
            raise NotADirectoryError(f"Skills directory does not exist: {self.root}")

        self._skills = self._load_index()

    def search(self, query: str, max_results: int = MAX_SKILL_RESULTS) -> str:
        result_limit = max(1, min(int(max_results), MAX_SKILL_RESULTS))
        matches = self._ranked_matches(query)
        if not matches:
            return f"No skills matched {query!r}."
        return "\n\n".join(
            f"Skill ID: {skill.identifier}\n"
            f"Name: {skill.name}\n"
            f"Description: {skill.description}"
            for _, skill in matches[:result_limit]
        )

    def best_match_id(self, query: str) -> str | None:
        matches = self._ranked_matches(query)
        return matches[0][1].identifier if matches else None

    def training_conversations(self) -> list[dict[str, list[dict[str, str]]]]:
        return [
            {
                "messages": [
                    {
                        "role": "user",
                        "content": (
                            f"In one concise paragraph, explain what {skill.name} "
                            "covers and when it is useful. Use only the reference "
                            "description."
                        ),
                    },
                    {"role": "assistant", "content": skill.description},
                ]
            }
            for skill in self._skills
            if skill.description
        ]

    def training_document_conversations(
        self, max_chars: int = MAX_SKILL_TRAINING_CHARS
    ) -> list[dict[str, list[dict[str, str]]]]:
        if max_chars < 1:
            raise ValueError("Training document chunk size must be positive.")

        conversations = []
        for skill in self._skills:
            content = skill.path.read_text(encoding="utf-8")
            for index, chunk in enumerate(
                self._split_training_content(content, max_chars), start=1
            ):
                conversations.append(
                    {
                        "messages": [
                            {
                                "role": "user",
                                "content": (
                                    f"Teach me the technical guidance in section "
                                    f"{index} of the {skill.name} skill."
                                ),
                            },
                            {"role": "assistant", "content": chunk},
                        ]
                    }
                )
        return conversations

    def _ranked_matches(self, query: str) -> list[tuple[int, Skill]]:
        terms = set(_WORD_PATTERN.findall(query.casefold()))
        if not terms:
            raise ValueError("Skill search query must include at least one word.")
        matches: list[tuple[int, Skill]] = []
        phrase = " ".join(query.casefold().split())
        for skill in self._skills:
            name = skill.name.casefold()
            identifier = skill.identifier.casefold().replace("-", " ").replace("/", " ")
            description = skill.description.casefold()
            searchable = f"{name} {identifier} {description}"
            score = sum(
                (4 if term in name else 0)
                + (2 if term in description else 0)
                + (1 if term in identifier else 0)
                for term in terms
            )
            if phrase and phrase in searchable:
                score += 4
            if score:
                matches.append((score, skill))
        matches.sort(key=lambda item: (-item[0], item[1].identifier.casefold()))
        return matches

    def read(self, identifier: str) -> str:
        skill = next(
            (item for item in self._skills if item.identifier == identifier),
            None,
        )
        if skill is None:
            raise ValueError(f"Unknown skill ID: {identifier!r}. Search skills first.")

        content = skill.path.read_text(encoding="utf-8")
        if len(content) > MAX_SKILL_CHARS:
            content = content[:MAX_SKILL_CHARS] + "\n\n... (skill content truncated)"
        return f"Skill reference: {skill.name} ({skill.identifier})\n\n{content}"

    def _load_index(self) -> list[Skill]:
        skills = []
        for current, directories, filenames in os.walk(
            self.root, followlinks=False
        ):
            current_path = Path(current)
            directories[:] = sorted(
                name
                for name in directories
                if not name.startswith(".")
                and not (current_path / name).is_symlink()
            )
            if "SKILL.md" in filenames:
                path = current_path / "SKILL.md"
                if path.is_symlink():
                    continue
                resolved_path = path.resolve(strict=True)
                if not resolved_path.is_relative_to(self.root):
                    continue
                relative_path = resolved_path.relative_to(self.root)
                name, description = self._metadata(resolved_path)
                skills.append(
                    Skill(
                        identifier=relative_path.parent.as_posix(),
                        name=name or relative_path.parent.name,
                        description=description,
                        path=resolved_path,
                    )
                )
        return sorted(skills, key=lambda skill: skill.identifier.casefold())

    @staticmethod
    def _split_training_content(content: str, max_chars: int) -> list[str]:
        chunks = []
        current = ""
        for line in content.splitlines(keepends=True):
            while line:
                available = max_chars - len(current)
                if available == 0:
                    chunks.append(current)
                    current = ""
                    available = max_chars
                current += line[:available]
                line = line[available:]
                if len(current) == max_chars:
                    chunks.append(current)
                    current = ""
        if current:
            chunks.append(current)
        return chunks

    @staticmethod
    def _metadata(path: Path) -> tuple[str, str]:
        content = path.read_text(encoding="utf-8")
        sections = content.split("---", 2)
        if len(sections) < 3 or sections[0].strip():
            return "", ""

        name = ""
        description = ""
        lines = sections[1].splitlines()
        index = 0
        while index < len(lines):
            key, separator, value = lines[index].partition(":")
            if not separator:
                index += 1
                continue
            key = key.strip()
            value = value.strip().strip("\"'")
            if key == "name":
                name = value
            elif key == "description":
                description_parts = [value] if value else []
                index += 1
                while index < len(lines) and (
                    not lines[index].strip() or lines[index][0].isspace()
                ):
                    continuation = lines[index].strip()
                    if continuation:
                        description_parts.append(continuation)
                    index += 1
                description = " ".join(description_parts)
                continue
            index += 1
        return name, description
