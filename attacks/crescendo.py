"""Crescendo (Russinovich, Salem & Eldan, 2024): the state-of-the-art multi-turn baseline.

The attacker model escalates gradually from benign questions toward the objective, building on
the target's own earlier answers. When the target refuses, the policy *backtracks*: the refused
exchange is removed from the target's context and the attacker tries a different message, up to
`max_backtracks` times per episode. Unlike the original algorithm, backtracked attempts count as
turns here, so Crescendo is compared with other policies under the same turn budget (§4).

Bounded like `attacker_llm`: one attacker-role call per turn through a ScopedClient, prompts come
only from an external template, replies are length-capped, and the policy sees only the
AttackerView. Crescendo's meta-prompt is supplied in that template, never in this file.
"""

from attacks.base import AttackAction, AttackerView, AttackPolicy
from attacks.template_loader import CrescendoTemplate, render
from models.metered import ScopedClient
from models.tokens import estimate_text_tokens
from models.types import ChatMessage


class CrescendoPolicy(AttackPolicy):
    name = "crescendo"
    max_model_calls_per_turn = 1

    def __init__(self, version: str, template: CrescendoTemplate) -> None:
        self.version = version
        self.template = template
        self.prompt_overhead_tokens = estimate_text_tokens(template.system + template.user + template.history_turn)
        self.backtracked: set[int] = set()

    def build_messages(self, view: AttackerView, backtracking: bool) -> list[ChatMessage]:
        t = self.template
        history = "\n".join(
            render(
                t.history_turn,
                {
                    "turn": h.turn,
                    "attacker_message": h.attacker_message,
                    "target_response": h.target_response or "[blocked]",
                    "refused": int(h.refused),
                    "refusal_type": h.refusal_type.value,
                    "progress": f"{h.progress:.1f}",
                    "backtracked": int(h.turn in self.backtracked),
                },
            )
            for h in view.history
        )
        b = view.budget
        values = {
            "objective": view.objective_text,
            "category": view.objective_category,
            "turn": view.next_turn,
            "history": history or "(none)",
            "backtracking": int(backtracking),
            "backtracks_left": t.max_backtracks - len(self.backtracked),
            "remaining_turns": b.remaining_turns,
            "remaining_queries": b.remaining_queries,
            "remaining_tokens": b.remaining_tokens,
        }
        return [
            ChatMessage(role="system", content=render(t.system, values)),
            ChatMessage(role="user", content=render(t.user, values)),
        ]

    def next_action(self, view: AttackerView, client: ScopedClient | None) -> AttackAction:
        if client is None:
            raise ValueError("crescendo requires an attacker-scoped client")
        last = view.history[-1] if view.history else None
        backtrack = (
            last is not None
            and last.refused
            and last.turn not in self.backtracked
            and len(self.backtracked) < self.template.max_backtracks
        )
        if backtrack:
            self.backtracked.add(last.turn)
        resp = client.call(self.build_messages(view, backtrack), purpose="attack_message")
        message = resp.text.strip()[: self.template.max_message_chars]
        if not message:
            return AttackAction(message=None, stop=True)
        return AttackAction(message=message, backtrack=backtrack)
