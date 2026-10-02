"""Provider-agnostic model identity and call roles."""

from enum import StrEnum

from pydantic import Field

from storage.versioning import Frozen


class Role(StrEnum):
    """Who a model call is made on behalf of. Every call is attributed to exactly one role."""

    ATTACKER = "attacker"
    ATTACKER_AUX = "attacker_aux"  # planner / attacker-side judge (extension policies)
    TARGET = "target"
    FILTER = "filter"
    EVALUATOR = "evaluator"


class SamplingParams(Frozen):
    temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    top_p: float = Field(default=1.0, gt=0.0, le=1.0)
    max_tokens: int = Field(gt=0)
    seed: int | None = None


class ModelSpec(Frozen):
    """Exact model identity. Changing any field is a different model for accounting and results."""

    provider: str = Field(min_length=1)  # e.g. "mock", "foundry"
    model_id: str = Field(min_length=1)
    version: str = Field(min_length=1)
    deployment: str | None = None  # Foundry deployment name, if different from model_id
    endpoint_ref: str | None = None  # name of an endpoint entry / env var; never a secret
    params: SamplingParams

    @property
    def pricing_key(self) -> str:
        return f"{self.provider}:{self.model_id}:{self.version}"
