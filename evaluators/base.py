"""Evaluator interfaces.

Two channels (PROJECT_SOURCE_OF_TRUTH §7):
- ground truth: `HarmScorer`. Its verdicts decide success and are never shown to the attacker;
- feedback: `RefusalClassifier` and `ProgressScorer`, visible to adaptive attack policies.

Evaluators are pure with respect to the episode: they receive an `EvalContext`, may make
model calls through a `ScopedClient` (evaluator role), and return a verdict. They never
mutate episode state.
"""

from abc import ABC, abstractmethod
from enum import StrEnum

from pydantic import Field

from models.metered import ScopedClient
from storage.versioning import Frozen


class EvaluatorError(Exception):
    """An evaluator could not produce a verdict (e.g. unparseable judge output)."""


class BlockSource(StrEnum):
    INPUT_FILTER = "input_filter"
    OUTPUT_FILTER = "output_filter"
    PROVIDER_FILTER = "provider_filter"


class RefusalType(StrEnum):
    NONE = "none"
    PARTIAL = "partial"
    REFUSAL = "refusal"
    BLOCKED_INPUT_FILTER = "blocked_input_filter"
    BLOCKED_OUTPUT_FILTER = "blocked_output_filter"
    BLOCKED_PROVIDER_FILTER = "blocked_provider_filter"

    @classmethod
    def for_block(cls, source: BlockSource) -> "RefusalType":
        return cls(f"blocked_{source.value}")


class EvalContext(Frozen):
    objective_text: str
    attacker_message: str
    target_response: str  # empty when the target system blocked the turn
    turn: int = Field(ge=1)
    blocked_by: BlockSource | None = None


class HarmVerdict(Frozen):
    score: float = Field(ge=0.0, le=1.0)
    evaluator_id: str
    rationale_ref: str | None = None


class RefusalVerdict(Frozen):
    refusal_type: RefusalType
    score: float = Field(ge=0.0, le=1.0)  # confidence that the turn was refused
    evaluator_id: str

    @property
    def refused(self) -> bool:
        return self.refusal_type not in (RefusalType.NONE, RefusalType.PARTIAL)


class ProgressVerdict(Frozen):
    score: float = Field(ge=0.0, le=1.0)
    evaluator_id: str


class _Evaluator(ABC):
    #: Model name (key into ExperimentConfig.models) if this evaluator calls a model.
    model_name: str | None = None

    def __init__(self, version: str) -> None:
        self.version = version

    @property
    def evaluator_id(self) -> str:
        return f"{self.impl}@{self.version}"

    impl: str


class HarmScorer(_Evaluator):
    """Ground truth. The only component whose output can mark an episode successful."""

    @abstractmethod
    def score(self, ctx: EvalContext, client: ScopedClient | None) -> HarmVerdict: ...


class RefusalClassifier(_Evaluator):
    @abstractmethod
    def classify(self, ctx: EvalContext, client: ScopedClient | None) -> RefusalVerdict: ...


class ProgressScorer(_Evaluator):
    @abstractmethod
    def score(self, ctx: EvalContext, client: ScopedClient | None) -> ProgressVerdict: ...
