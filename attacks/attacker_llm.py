"""Adaptive attacker-LLM policy (§5, policy 2).

Bounded by construction: exactly one attacker-role model call per turn through a ScopedClient,
prompts come only from an external template, the reply is length-capped, and the policy sees
only the AttackerView (conversation, refusal/progress feedback, remaining budget).
"""

from attacks.base import AttackAction, AttackerView, AttackPolicy
from attacks.template_loader import AttackerPromptTemplate, render
from models.metered import ScopedClient
from models.tokens import estimate_text_tokens
from models.types import ChatMessage


class AttackerLLMPolicy(AttackPolicy):
    name = "attacker_llm"
    max_model_calls_per_turn = 1

    def __init__(self, version: str, template: AttackerPromptTemplate) -> None:
        self.version = version
        self.template = template
        self.prompt_overhead_tokens = estimate_text_tokens(template.system + template.user + template.history_turn)

    def build_messages(self, view: AttackerView) -> list[ChatMessage]:
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
            raise ValueError("attacker_llm requires an attacker-scoped client")
        resp = client.call(self.build_messages(view), purpose="attack_message")
        message = resp.text.strip()[: self.template.max_message_chars]
        if not message:
            return AttackAction(message=None, stop=True)
        return AttackAction(message=message)
