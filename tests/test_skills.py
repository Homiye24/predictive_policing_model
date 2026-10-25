from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SKILLS_ROOT = PROJECT_ROOT / ".github" / "skills"


def test_copilot_skills_have_valid_metadata_and_bodies() -> None:
    skill_files = sorted(SKILLS_ROOT.glob("*/SKILL.md"))

    assert skill_files, "No Copilot Agent Skills were found."

    names: set[str] = set()
    for skill_file in skill_files:
        content = skill_file.read_text(encoding="utf-8")
        sections = content.split("---", maxsplit=2)

        assert len(sections) == 3, f"Missing YAML frontmatter in {skill_file}"
        metadata = yaml.safe_load(sections[1])
        assert isinstance(metadata, dict), f"Invalid frontmatter in {skill_file}"
        assert metadata.get("name") == skill_file.parent.name
        assert isinstance(metadata.get("description"), str)
        assert metadata["description"].strip()
        assert sections[2].strip(), f"Skill instructions are empty in {skill_file}"
        assert metadata["name"] not in names, f"Duplicate skill name: {metadata['name']}"
        names.add(metadata["name"])
