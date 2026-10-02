"""Provider-agnostic model identity and call roles."""

from enum import StrEnum
from typing import Literal

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


ChatRole = Literal["system", "user", "assistant"]
FinishReason = Literal["stop", "length", "content_filter", "other"]


class ChatMessage(Frozen):
    role: ChatRole
    content: str


class Usage(Frozen):
    """Token usage for one provider attempt. `estimated` is True when the provider omitted it."""

    input_tokens: int = Field(ge=0)
    output_tokens: int = Field(ge=0)
    estimated: bool = False

    @property
    def total(self) -> int:
        return self.input_tokens + self.output_tokens


class ModelRequest(Frozen):
    """One provider attempt. Built only by MeteredClient; adapters never see budgets."""

    spec: ModelSpec
    messages: tuple[ChatMessage, ...] = Field(min_length=1)
    role: Role
    timeout_s: float = Field(gt=0)
    seed: int | None = None


class ModelResponse(Frozen):
    """Normalised provider response. `usage` is None when the provider did not report it."""

    text: str
    finish_reason: FinishReason
    usage: Usage | None
    provider_model: str | None = None
    provider_request_id: str | None = None
    # Provider-side content filtering on the output (Foundry `content_filter` finish reason).
    content_filtered: bool = False
    filter_categories: tuple[str, ...] = ()
