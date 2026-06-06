import importlib.util
from pathlib import Path
from typing import Any

from knowledge_base.registry import CATEGORIES

_PROJECT_ROOT = Path(__file__).resolve().parent.parent


# Return category ids and titles for UI selection widgets.
def list_categories() -> list[tuple[str, str]]:
    return [(category_id, meta["title"]) for category_id, meta in CATEGORIES.items()]


# Load a category module from the registry entry for the given id.
def _load_category_module(category_id: str) -> Any:
    entry = CATEGORIES.get(category_id)
    if entry is None:
        raise ValueError(f"Unknown knowledge category: {category_id}")

    file_path = Path(entry["file"])
    if not file_path.is_absolute():
        file_path = _PROJECT_ROOT / file_path
    if not file_path.is_file():
        raise ValueError(f"Knowledge category file not found: {file_path}")

    spec = importlib.util.spec_from_file_location(f"kb_{category_id}", file_path)
    if spec is None or spec.loader is None:
        raise ValueError(f"Could not load knowledge category: {category_id}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# Format one disorder entry into markdown-style prompt text.
def _format_disorder(disorder: dict[str, Any]) -> str:
    lines = [f"### {disorder['label']}", "", "Criteria:"]
    for criterion in disorder["criteria"]:
        lines.append(f"- {criterion}")

    notes = disorder.get("notes", [])
    if notes:
        lines.append("")
        lines.append("Notes:")
        for note in notes:
            lines.append(f"- {note}")

    return "\n".join(lines)


# Format all disorders for one category into prompt text.
def _format_category(title: str, disorders: list[dict[str, Any]]) -> str:
    sections = [f"## {title}", ""]
    for disorder in disorders:
        sections.append(_format_disorder(disorder))
        sections.append("")
    return "\n".join(sections).strip()


# Build the DSM-5 reference block appended to agent prompts for selected categories.
def build_knowledge_context(category_ids: list[str]) -> str:
    if not category_ids:
        return ""

    sections: list[str] = []
    for category_id in category_ids:
        module = _load_category_module(category_id)
        title = getattr(module, "CATEGORY_TITLE", CATEGORIES[category_id]["title"])
        disorders = module.DISORDERS
        sections.append(_format_category(title, disorders))

    return "\n\n".join(sections)
