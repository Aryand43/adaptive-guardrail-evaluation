"""Experiment configuration schema.

File paths are kept exactly as written (relative to the config file) so the config
hash is machine-independent. The loader resolves them, checks they exist, and hashes
their contents separately.
"""

import re
from typing import Any, Literal

from pydantic import Field, field_validator, model_validator

from models.budget import BudgetLimits
from models.types import ModelSpec
from storage.versioning import Frozen, Versioned

_SLUG = re.compile(r"^[a-z0-9][a-z0-9_.-]{0,63}$")

RunMode = Literal["dev", "final"]
Split = Literal["dev", "test"]


class DatasetConfig(Frozen):
    manifest_path: str  # external dataset manifest; objectives are never inlined in configs
    split: Split


class PricingConfig(Frozen):
    path: str


class PolicyConfig(Frozen):
    name: str
    version: str
    params: dict[str, Any] = Field(default_factory=dict)
    # External prompt-template files (e.g. escalation ladder, attacker system prompt).
    # The repository ships neutral placeholders only.
    template_paths: dict[str, str] = Field(default_factory=dict)


class EvaluatorConfig(Frozen):
    impl: str  # registry key, e.g. "mock", "llm_judge"
    version: str
    model: str | None = None  # key into ExperimentConfig.models; None for non-LLM impls
    rubric_id: str | None = None
    params: dict[str, Any] = Field(default_factory=dict)


class EvaluatorsConfig(Frozen):
    """`harm` is ground truth and hidden from the attacker; `refusal` and `progress` are feedback."""

    harm: EvaluatorConfig
    refusal: EvaluatorConfig
    progress: EvaluatorConfig
    success_threshold: float = Field(gt=0.0, le=1.0)


class ObjectiveWeights(Frozen):
    lambda_query: float = Field(ge=0.0)
    lambda_token: float = Field(ge=0.0)
    lambda_latency: float = Field(ge=0.0)
    lambda_money: float = Field(ge=0.0)


class StopRulesConfig(Frozen):
    stagnation_enabled: bool = False
    stagnation_window: int = Field(default=3, gt=0)
    stagnation_min_gain: float = Field(default=0.0, ge=0.0)


class StorageConfig(Frozen):
    runs_dir: str = "runs"


class ExperimentConfig(Versioned):
    experiment_id: str
    mode: RunMode
    models: dict[str, ModelSpec]
    attacker: str
    targets: list[str] = Field(min_length=1)
    dataset: DatasetConfig
    pricing: PricingConfig
    policies: list[PolicyConfig] = Field(min_length=1)
    evaluators: EvaluatorsConfig
    budgets: list[BudgetLimits] = Field(min_length=1)
    objective_weights: ObjectiveWeights
    stop_rules: StopRulesConfig = StopRulesConfig()
    seeds: list[int] = Field(min_length=1)
    n_trials: int = Field(default=1, gt=0)
    storage: StorageConfig = StorageConfig()

    @field_validator("experiment_id")
    @classmethod
    def _slug(cls, v: str) -> str:
        if not _SLUG.match(v):
            raise ValueError("experiment_id must be a lowercase slug")
        return v

    @model_validator(mode="after")
    def _cross_checks(self) -> "ExperimentConfig":
        refs = {"attacker": self.attacker} | {f"targets[{i}]": t for i, t in enumerate(self.targets)}
        for role in ("harm", "refusal", "progress"):
            model = getattr(self.evaluators, role).model
            if model is not None:
                refs[f"evaluators.{role}.model"] = model
        missing = [f"{where}={name!r}" for where, name in refs.items() if name not in self.models]
        if missing:
            raise ValueError(f"unknown model references: {', '.join(missing)}")
        if len(set(self.targets)) != len(self.targets):
            raise ValueError("duplicate target models")
        if self.attacker in self.targets:
            raise ValueError("attacker and target must be separate model entries")
        keys = [(p.name, p.version) for p in self.policies]
        if len(set(keys)) != len(keys):
            raise ValueError("duplicate policy (name, version)")
        if len(set(self.seeds)) != len(self.seeds):
            raise ValueError("duplicate seeds")
        # Sealed-test protection at the config level: dev runs never touch the test split,
        # final runs only ever use it.
        if self.mode == "dev" and self.dataset.split != "dev":
            raise ValueError("dev mode may only use the dev split")
        if self.mode == "final" and self.dataset.split != "test":
            raise ValueError("final mode must use the held-out test split")
        return self

    def referenced_paths(self) -> dict[str, str]:
        """Logical name → path (as written) of every external file this config depends on."""
        paths = {"dataset.manifest": self.dataset.manifest_path, "pricing": self.pricing.path}
        for p in self.policies:
            for tname, tpath in p.template_paths.items():
                paths[f"policy.{p.name}@{p.version}.{tname}"] = tpath
        return paths
