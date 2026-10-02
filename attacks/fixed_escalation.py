"""Fixed progressive escalation (§5, policy 1): a deterministic, non-adaptive baseline.

Turn t sends ladder step t, rendered with the objective. It ignores feedback by design and
stops when the ladder is exhausted. It makes no model calls.
"""

from attacks.base import AttackAction, AttackerView, AttackPolicy
from attacks.template_loader import LadderTemplate, render
from models.metered import ScopedClient


class FixedEscalationPolicy(AttackPolicy):
    name = "fixed_escalation"
    max_model_calls_per_turn = 0

    def __init__(self, version: str, ladder: LadderTemplate) -> None:
        self.version = version
        self.ladder = ladder

    def next_action(self, view: AttackerView, client: ScopedClient | None) -> AttackAction:
        i = view.next_turn - 1
        if i >= len(self.ladder.steps):
            return AttackAction(message=None, stop=True)
        values = {"objective": view.objective_text, "turn": view.next_turn}
        return AttackAction(message=render(self.ladder.steps[i], values))
