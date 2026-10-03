"""Build attack policies from config. Each episode gets a fresh policy instance."""

from collections.abc import Callable
from pathlib import Path

from attacks.attacker_llm import AttackerLLMPolicy
from attacks.base import AttackPolicy
from attacks.crescendo import CrescendoPolicy
from attacks.fixed_escalation import FixedEscalationPolicy
from attacks.template_loader import AttackerPromptTemplate, CrescendoTemplate, LadderTemplate, load_template
from configs.schema import PolicyConfig
from storage.hashing import hash_obj


class PolicyConfigError(Exception):
    pass


class PolicyFactory:
    def __init__(self, config: PolicyConfig, make: Callable[[], AttackPolicy], template_hashes: dict[str, str]):
        self.config = config
        self._make = make
        self.template_hashes = template_hashes
        self.config_hash = hash_obj(config)

    @property
    def key(self) -> str:
        return f"{self.config.name}@{self.config.version}"

    def __call__(self) -> AttackPolicy:
        return self._make()


# name -> (required template slot, template model, constructor)
_POLICIES = {
    "fixed_escalation": ("ladder", LadderTemplate, FixedEscalationPolicy),
    "attacker_llm": ("prompt", AttackerPromptTemplate, AttackerLLMPolicy),
    "crescendo": ("prompt", CrescendoTemplate, CrescendoPolicy),
}


def build_policy(config: PolicyConfig, resolve: Callable[[str], Path]) -> PolicyFactory:
    if config.name not in _POLICIES:
        raise PolicyConfigError(
            f"unknown policy {config.name!r}; available: {sorted(_POLICIES)} "
            "(planner-attacker-judge and evolutionary policies are extension points, not implemented)"
        )
    slot, model, ctor = _POLICIES[config.name]
    extra = set(config.template_paths) - {slot}
    if slot not in config.template_paths or extra:
        raise PolicyConfigError(f"{config.name} needs exactly one template path named {slot!r}")
    if config.params:
        raise PolicyConfigError(f"{config.name} takes no params")
    template, digest = load_template(resolve(config.template_paths[slot]), model)
    if template.version != config.version:
        raise PolicyConfigError(f"{config.name}: template version {template.version} != config {config.version}")
    return PolicyFactory(config, lambda: ctor(config.version, template), {slot: digest})
