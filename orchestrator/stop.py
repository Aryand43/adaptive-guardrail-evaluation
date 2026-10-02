"""Stop controller: evaluates stop rules before and after each turn and resolves priority."""

from configs.schema import StopRulesConfig
from models.budget import Amount, BudgetLedger
from orchestrator.state import STOP_PRIORITY, EpisodeState, StopReason
from storage.versioning import Frozen


class StopDecision(Frozen):
    reason: StopReason
    detail: str | None = None


def resolve(candidates: list[StopDecision]) -> StopDecision | None:
    if not candidates:
        return None
    return min(candidates, key=lambda d: STOP_PRIORITY.index(d.reason))


class StopController:
    def __init__(self, rules: StopRulesConfig, success_threshold: float) -> None:
        self.rules = rules
        self.success_threshold = success_threshold

    def _resource_candidates(self, ledger: BudgetLedger) -> list[StopDecision]:
        out = []
        if exhausted := ledger.exhausted():
            out.append(StopDecision(reason=StopReason.BUDGET_EXHAUSTED, detail=",".join(exhausted)))
        if ledger.remaining_wall_s <= 0:
            out.append(StopDecision(reason=StopReason.WALL_TIME))
        if ledger.remaining_turns == 0:
            out.append(StopDecision(reason=StopReason.TURN_LIMIT))
        return out

    def before_turn(self, state: EpisodeState, ledger: BudgetLedger, turn_estimate: Amount) -> StopDecision | None:
        candidates = self._resource_candidates(ledger)
        if short := ledger.shortfall(turn_estimate):
            candidates.append(StopDecision(reason=StopReason.INSUFFICIENT_BUDGET, detail=",".join(short)))
        return resolve(candidates)

    def after_turn(self, state: EpisodeState, ledger: BudgetLedger) -> StopDecision | None:
        candidates = self._resource_candidates(ledger)
        if state.success_turn is not None:
            candidates.append(StopDecision(reason=StopReason.SUCCESS))
        if self._stagnated(state):
            candidates.append(StopDecision(reason=StopReason.STAGNATION))
        return resolve(candidates)

    def _stagnated(self, state: EpisodeState) -> bool:
        """Feedback-channel progress has not improved by `min_gain` over the last `window` turns."""
        w = self.rules.stagnation_window
        if not self.rules.stagnation_enabled or state.turns_completed <= w:
            return False
        progress = [t.progress.score for t in state.turns]
        before = max(progress[:-w])
        recent = max(progress[-w:])
        return recent - before <= self.rules.stagnation_min_gain
