"""The provider adapter interface.

An adapter performs exactly one provider attempt per `complete()` call. It does not retry,
meter, log, or budget; MeteredClient does all of that. Adapters raise only `ModelError`
subclasses and never log secrets or raw text.
"""

from abc import ABC, abstractmethod

from models.types import ModelRequest, ModelResponse, ModelSpec


class ModelAdapter(ABC):
    #: Stable identifier recorded in manifests (e.g. "mock", "openrouter").
    adapter_id: str

    @abstractmethod
    def complete(self, request: ModelRequest) -> ModelResponse:
        """Perform one attempt. Must honour `request.timeout_s`."""

    def check(self, spec: ModelSpec) -> str | None:
        """Return a reason this adapter cannot serve `spec`, or None. Called before any call."""
        return None
