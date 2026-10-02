"""Target system: optional input filter -> target model -> optional output filter (§6).

All model calls go through ScopedClients created by the orchestrator. A provider-side
content-filter rejection is a *guardrail outcome*, not an error.
"""

import re
from dataclasses import dataclass
from abc import ABC, abstractmethod
from collections.abc import Sequence

from evaluators.base import BlockSource
from models.metered import MeteredCallError, ScopedClient
from models.types import ChatMessage, FinishReason
from storage.versioning import Frozen


class FilterDecision(Frozen):
    blocked: bool
    categories: tuple[str, ...] = ()


class GuardrailFilter(ABC):
    """An input or output guardrail. Model-backed filters call through a FILTER-role client."""

    filter_id: str
    model_name: str | None = None

    @abstractmethod
    def check(self, text: str, client: ScopedClient | None) -> FilterDecision: ...


class MockMarkerFilter(GuardrailFilter):
    """Offline test filter: blocks text whose mock pressure/compliance marker crosses a threshold."""

    _MARK = re.compile(r"\[\[mock:(?:pressure|compliance)=([0-9.]+)\]\]")

    def __init__(self, threshold: float, filter_id: str = "mock_marker@0.1") -> None:
        self.threshold = threshold
        self.filter_id = filter_id

    def check(self, text: str, client: ScopedClient | None) -> FilterDecision:
        values = [float(v) for v in self._MARK.findall(text)]
        blocked = bool(values) and max(values) >= self.threshold
        return FilterDecision(blocked=blocked, categories=("mock_marker",) if blocked else ())


class TargetReply(Frozen):
    text: str  # empty when blocked before a response reached the user
    response_ref: str | None
    blocked_by: BlockSource | None
    finish_reason: FinishReason | None
    filter_categories: tuple[str, ...] = ()


@dataclass(frozen=True)
class FilterClients:
    input: ScopedClient | None = None
    output: ScopedClient | None = None


class TargetSystem:
    def __init__(
        self,
        target_model: str,
        *,
        input_filter: GuardrailFilter | None = None,
        output_filter: GuardrailFilter | None = None,
        system_prompt: str | None = None,
    ) -> None:
        self.target_model = target_model
        self.input_filter = input_filter
        self.output_filter = output_filter
        self.system_prompt = system_prompt

    @property
    def model_calls_per_turn(self) -> dict[str, int]:
        """Model name -> max calls per turn, for turn reservations."""
        calls = {self.target_model: 1}
        for f in (self.input_filter, self.output_filter):
            if f is not None and f.model_name is not None:
                calls[f.model_name] = calls.get(f.model_name, 0) + 1
        return calls

    def respond(
        self,
        conversation: Sequence[ChatMessage],
        target_client: ScopedClient,
        filters: FilterClients | None = None,
    ) -> tuple[TargetReply, list[tuple[str, FilterDecision]]]:
        """Returns the reply and the filter decisions taken (stage, decision) for logging."""
        filters = filters or FilterClients()
        decisions: list[tuple[str, FilterDecision]] = []
        latest = conversation[-1].content
        if self.input_filter is not None:
            d = self.input_filter.check(latest, filters.input)
            decisions.append(("input", d))
            if d.blocked:
                return self._blocked(BlockSource.INPUT_FILTER, d.categories), decisions

        messages = list(conversation)
        if self.system_prompt:
            messages.insert(0, ChatMessage(role="system", content=self.system_prompt))
        try:
            resp = target_client.call(messages, purpose="target_response")
        except MeteredCallError as exc:
            if exc.content_filtered:
                cats = tuple(getattr(exc.error, "categories", ()))
                return self._blocked(BlockSource.PROVIDER_FILTER, cats), decisions
            raise
        if resp.content_filtered:
            return (
                TargetReply(
                    text="",
                    response_ref=resp.response_ref,
                    blocked_by=BlockSource.PROVIDER_FILTER,
                    finish_reason=resp.finish_reason,
                    filter_categories=resp.filter_categories,
                ),
                decisions,
            )

        if self.output_filter is not None:
            d = self.output_filter.check(resp.text, filters.output)
            decisions.append(("output", d))
            if d.blocked:
                return (
                    TargetReply(
                        text="",
                        response_ref=resp.response_ref,  # what the model said is kept for audit
                        blocked_by=BlockSource.OUTPUT_FILTER,
                        finish_reason=resp.finish_reason,
                        filter_categories=d.categories,
                    ),
                    decisions,
                )
        return (
            TargetReply(
                text=resp.text,
                response_ref=resp.response_ref,
                blocked_by=None,
                finish_reason=resp.finish_reason,
            ),
            decisions,
        )

    @staticmethod
    def _blocked(source: BlockSource, categories: tuple[str, ...]) -> TargetReply:
        return TargetReply(
            text="", response_ref=None, blocked_by=source, finish_reason=None, filter_categories=categories
        )
