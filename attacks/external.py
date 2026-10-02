"""Integration interface for approved external red-teaming frameworks (e.g. PyRIT, HarmBench,
JailbreakBench attack implementations).

STATUS: interface only — not registered in `attacks.registry` and not usable from experiment
configs until approved (see PROJECT_SOURCE_OF_TRUTH.md §14, deviation log). No framework is a
dependency of this repository; providers are imported lazily from an allow-list.

Boundary: an external provider is wrapped as an ordinary AttackPolicy, so it inherits every
architectural guarantee. It receives only the AttackerView and a `BoundedGenerate` callable —
never a client, ledger, HTTP session, credentials, or harm scores. Every generation goes
through the episode's MeteredClient (attacker role), the number of calls per turn is capped
and reserved in advance, and the proposed message is length-capped.
"""

import importlib
from abc import ABC, abstractmethod
from collections.abc import Sequence

from attacks.base import AttackAction, AttackerView, AttackPolicy
from models.metered import ScopedClient
from models.types import ChatMessage

#: Allow-list of importable provider modules. Extending it is a reviewed change.
APPROVED_PROVIDERS: dict[str, str] = {}


class ExternalProviderUnavailable(Exception):
    pass


class CallLimitExceeded(Exception):
    pass


class BoundedGenerate:
    """The only model access an external provider gets: N attacker-role calls per turn."""

    def __init__(self, client: ScopedClient, max_calls: int) -> None:
        self._client = client
        self._remaining = max_calls

    @property
    def remaining(self) -> int:
        return self._remaining

    def __call__(self, messages: Sequence[ChatMessage]) -> str:
        if self._remaining <= 0:
            raise CallLimitExceeded("external provider exceeded its per-turn call limit")
        self._remaining -= 1
        return self._client.call(list(messages), purpose="external_attack").text


class ExternalAttackProvider(ABC):
    provider_id: str
    version: str
    #: Upper bound on model calls per proposed message; used for the turn reservation.
    max_calls_per_turn: int
    prompt_overhead_tokens: int = 0

    @abstractmethod
    def propose(self, view: AttackerView, generate: BoundedGenerate | None) -> str | None:
        """Return the next target-bound message, or None to stop."""


class ExternalProviderPolicy(AttackPolicy):
    """Adapts an ExternalAttackProvider to the AttackPolicy contract."""

    def __init__(self, provider: ExternalAttackProvider, *, max_message_chars: int = 4000) -> None:
        if provider.max_calls_per_turn < 0 or provider.max_calls_per_turn > 8:
            raise ValueError("external providers may use between 0 and 8 model calls per turn")
        self.provider = provider
        self.name = f"external:{provider.provider_id}"
        self.version = provider.version
        self.max_model_calls_per_turn = provider.max_calls_per_turn
        self.prompt_overhead_tokens = provider.prompt_overhead_tokens
        self.max_message_chars = max_message_chars

    def next_action(self, view: AttackerView, client: ScopedClient | None) -> AttackAction:
        generate = None
        if self.max_model_calls_per_turn:
            if client is None:
                raise ValueError(f"{self.name} needs an attacker-scoped client")
            generate = BoundedGenerate(client, self.max_model_calls_per_turn)
        message = self.provider.propose(view, generate)
        if message is None or not message.strip():
            return AttackAction(message=None, stop=True)
        return AttackAction(message=message[: self.max_message_chars])


def load_provider(provider_id: str, **kwargs) -> ExternalAttackProvider:
    """Import an approved provider module lazily. Fails closed for anything not allow-listed."""
    module_name = APPROVED_PROVIDERS.get(provider_id)
    if module_name is None:
        raise ExternalProviderUnavailable(f"external provider {provider_id!r} is not approved")
    try:
        module = importlib.import_module(module_name)
    except ImportError as exc:
        raise ExternalProviderUnavailable(f"{provider_id}: optional dependency not installed") from exc
    provider = module.make_provider(**kwargs)
    if not isinstance(provider, ExternalAttackProvider):
        raise ExternalProviderUnavailable(f"{provider_id}: make_provider() returned {type(provider).__name__}")
    return provider
