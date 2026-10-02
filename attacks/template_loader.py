"""External policy templates: loading, hashing, and `{{slot}}` rendering.

Templates are data supplied outside the repository (attacks/templates/external/, git-ignored).
The repository ships neutral placeholders only. Rendering is plain slot substitution — no
format strings or code execution — and unknown slots fail closed.
"""

import re
from pathlib import Path
from typing import Any

from pydantic import Field

from configs.loader import load_yaml
from storage.hashing import sha256_bytes
from storage.versioning import Versioned

SLOT_RE = re.compile(r"\{\{([a-z_]+)\}\}")


class TemplateError(Exception):
    pass


def render(template: str, values: dict[str, Any]) -> str:
    missing = sorted(set(SLOT_RE.findall(template)) - set(values))
    if missing:
        raise TemplateError(f"template slots without values: {', '.join(missing)}")
    return SLOT_RE.sub(lambda m: str(values[m.group(1)]), template)


class LadderTemplate(Versioned):
    policy: str = "fixed_escalation"
    version: str
    steps: list[str] = Field(min_length=1, max_length=64)


class AttackerPromptTemplate(Versioned):
    policy: str = "attacker_llm"
    version: str
    system: str = Field(min_length=1)
    user: str = Field(min_length=1)
    history_turn: str = Field(min_length=1)
    max_message_chars: int = Field(default=4000, gt=0, le=20000)


def load_template(path: str | Path, model: type[Versioned]) -> tuple[Any, str]:
    """Returns (validated template, sha256 of the file bytes)."""
    path = Path(path)
    if not path.is_file():
        raise TemplateError(f"template not found: {path}")
    raw = load_yaml(path)
    if not isinstance(raw, dict):
        raise TemplateError(f"{path.name}: template must be a mapping")
    try:
        return model.model_validate(raw), sha256_bytes(path.read_bytes())
    except ValueError as exc:
        raise TemplateError(f"{path.name}: invalid template\n{exc}") from exc
