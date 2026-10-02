"""Attack policy interface and the attacker-visible view (PROJECT_SOURCE_OF_TRUTH §5, §8).

A policy receives only an `AttackerView` and, if it uses a model, a `ScopedClient` bound to
the attacker role and attacker model. It returns an `AttackAction`; it cannot mutate the
episode, reach the ledger, or see ground-truth harm scores — `AttackerView` has no field
that could carry them.
"""

from abc import ABC, abstractmethod

from pydantic import Field

from evaluators.base import RefusalType
from models.metered import ScopedClient
from storage.versioning import Frozen


class FeedbackTurn(Frozen):
    """One completed turn as the attacker may see it: text plus feedback-channel verdicts."""

    turn: int = Field(ge=1)
    attacker_message: str
    target_response: str  # empty if the target system blocked the turn
    refusal_type: RefusalType
    refused: bool
    progress: float = Field(ge=0.0, le=1.0)


class BudgetView(Frozen):
    remaining_turns: int
    remaining_queries: int
    remaining_tokens: int
    remaining_cost_usd: float
    remaining_wall_s: float


class AttackerView(Frozen):
    objective_id: str
    objective_category: str
    objective_text: str
    next_turn: int = Field(ge=1)
    history: tuple[FeedbackTurn, ...]
    budget: BudgetView
    seed: int


class AttackAction(Frozen):
    """`message=None` (or `stop=True`) means the policy has nothing further to send."""

    message: str | None
    stop: bool = False
    note: str | None = Field(default=None, max_length=2000)  # optional rationale; stored as a blob


class AttackPolicy(ABC):
    name: str
    version: str
    #: Upper bound on attacker-role model calls per turn, used to reserve the turn's budget.
    max_model_calls_per_turn: int = 0
    #: Prompt tokens the policy adds on top of the conversation (system prompt, template).
    prompt_overhead_tokens: int = 0

    @abstractmethod
    def next_action(self, view: AttackerView, client: ScopedClient | None) -> AttackAction: ...
