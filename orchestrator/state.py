"""Immutable episode state. Only the orchestrator constructs successor states."""

from enum import StrEnum

from pydantic import Field

from evaluators.base import BlockSource, HarmVerdict, ProgressVerdict, RefusalVerdict
from storage.versioning import Frozen


class EpisodeStatus(StrEnum):
    RUNNING = "running"
    SUCCESS = "success"
    FAILURE = "failure"
    ERROR = "error"


class StopReason(StrEnum):
    """Listed in priority order: when several apply at once, the earliest wins."""

    FATAL_ERROR = "fatal_error"
    SUCCESS = "success"
    BUDGET_EXHAUSTED = "budget_exhausted"  # a reservable dimension fully consumed (incl. overrun)
    WALL_TIME = "wall_time"
    TURN_LIMIT = "turn_limit"
    INSUFFICIENT_BUDGET = "insufficient_budget"  # next turn's reservation does not fit
    POLICY_EXHAUSTED = "policy_exhausted"
    STAGNATION = "stagnation"


STOP_PRIORITY: tuple[StopReason, ...] = tuple(StopReason)


class TurnRecord(Frozen):
    turn: int = Field(ge=1)
    attacker_message_ref: str
    target_response_ref: str | None
    blocked_by: BlockSource | None
    harm: HarmVerdict
    refusal: RefusalVerdict
    progress: ProgressVerdict


class EpisodeState(Frozen):
    episode_id: str
    objective_id: str
    status: EpisodeStatus = EpisodeStatus.RUNNING
    turns: tuple[TurnRecord, ...] = ()
    success_turn: int | None = None
    stop_reason: StopReason | None = None
    stop_detail: str | None = None

    @property
    def turns_completed(self) -> int:
        return len(self.turns)

    @property
    def max_harm(self) -> float:
        return max((t.harm.score for t in self.turns), default=0.0)

    def with_turn(self, record: TurnRecord, success_threshold: float) -> "EpisodeState":
        if self.status != EpisodeStatus.RUNNING:
            raise ValueError("cannot add a turn to a stopped episode")
        if record.turn != self.turns_completed + 1:
            raise ValueError(f"expected turn {self.turns_completed + 1}, got {record.turn}")
        success_turn = self.success_turn
        if success_turn is None and record.harm.score >= success_threshold:
            success_turn = record.turn  # only the ground-truth harm verdict can set this
        return self.model_copy(update={"turns": (*self.turns, record), "success_turn": success_turn})

    def stopped(self, reason: StopReason, detail: str | None = None) -> "EpisodeState":
        if self.status != EpisodeStatus.RUNNING:
            raise ValueError("episode already stopped")
        if reason == StopReason.FATAL_ERROR:
            status = EpisodeStatus.ERROR
        elif self.success_turn is not None:
            status = EpisodeStatus.SUCCESS
        else:
            status = EpisodeStatus.FAILURE
        return self.model_copy(update={"status": status, "stop_reason": reason, "stop_detail": detail})
