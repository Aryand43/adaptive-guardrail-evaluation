"""MeteredClient: the only path from the harness to a model adapter.

One `call()` is one logical *query*. Within it, each provider attempt is reserved before it is
sent and reconciled afterwards:

    reserve(estimate) -> adapter.complete() -> commit(actual) -> [backoff -> retry]

- The query is counted once, on the first attempt, whether or not the call succeeds.
- Tokens and cost are charged for every attempt. When the provider omits usage, or an attempt
  fails after the prompt may have been processed (timeout, 5xx), usage is estimated and flagged.
- Retries need their own reservation; if it does not fit, retrying stops and the call fails.
- Each attempt's timeout is capped by the episode's remaining wall-clock budget.
- Backoff is deterministic (no jitter) so mock runs replay exactly.

Raw prompts and responses go to the blob store; events carry only their hashes.
"""

import time
from collections.abc import Callable, Mapping, Sequence

from pydantic import Field

from models.adapters.base import ModelAdapter
from models.budget import Amount, BudgetDenied, BudgetLedger, Reservation
from models.errors import ModelContentFilterError, ModelError, ModelRateLimitError, UnknownModelError
from models.pricing import PricingTable
from models.tokens import estimate_prompt_tokens, estimate_text_tokens
from models.types import ChatMessage, FinishReason, ModelRequest, ModelResponse, ModelSpec, Role, Usage
from storage.blobs import BlobStore
from storage.clock import Clock
from storage.events import EventSink, EventType
from storage.hashing import canonical_json
from storage.versioning import Frozen


class RetryPolicy(Frozen):
    max_attempts: int = Field(default=3, ge=1)
    timeout_s: float = Field(default=60.0, gt=0)
    backoff_base_s: float = Field(default=1.0, ge=0)
    backoff_max_s: float = Field(default=20.0, ge=0)

    def backoff(self, attempt: int, error: ModelError) -> float:
        if isinstance(error, ModelRateLimitError) and error.retry_after_s is not None:
            return min(error.retry_after_s, self.backoff_max_s)
        return min(self.backoff_base_s * 2 ** (attempt - 1), self.backoff_max_s)


class AttemptRecord(Frozen):
    attempt: int
    outcome: str  # "ok" or an error kind
    latency_s: float
    usage: Usage
    cost_nano: int


class MeteredResponse(Frozen):
    call_id: str
    role: Role
    model_name: str
    text: str
    finish_reason: FinishReason
    content_filtered: bool
    filter_categories: tuple[str, ...]
    usage: Usage  # summed over attempts; `estimated` if any attempt was estimated
    cost_nano: int
    latency_s: float  # wall time including backoff
    attempts: tuple[AttemptRecord, ...]
    request_ref: str
    response_ref: str


class MeteredCallError(Exception):
    """A logical call that produced no usable response. Carries the typed provider error."""

    def __init__(self, call_id: str, error: ModelError | BudgetDenied, attempts: Sequence[AttemptRecord]):
        self.call_id = call_id
        self.error = error
        self.attempts = tuple(attempts)
        super().__init__(f"{call_id}: {error}")

    @property
    def budget_denied(self) -> bool:
        return isinstance(self.error, BudgetDenied)

    @property
    def content_filtered(self) -> bool:
        return isinstance(self.error, ModelContentFilterError)


def _sum_usage(attempts: Sequence[AttemptRecord]) -> Usage:
    return Usage(
        input_tokens=sum(a.usage.input_tokens for a in attempts),
        output_tokens=sum(a.usage.output_tokens for a in attempts),
        estimated=any(a.usage.estimated for a in attempts),
    )


def _amount_json(a: Amount) -> dict:
    return a.model_dump(mode="json")


class MeteredClient:
    def __init__(
        self,
        *,
        models: Mapping[str, ModelSpec],
        adapters: Mapping[str, ModelAdapter],
        pricing: PricingTable,
        ledger: BudgetLedger,
        clock: Clock,
        blobs: BlobStore,
        sink: EventSink,
        retry: RetryPolicy | None = None,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self._models = dict(models)
        self._adapters = dict(adapters)
        self._pricing = pricing
        self.ledger = ledger
        self._clock = clock
        self._blobs = blobs
        self._sink = sink
        self.retry = retry or RetryPolicy()
        self._sleep = sleep
        self._n_calls = 0
        problems = [p for name in self._models if (p := self._unservable(name))]
        if problems:
            raise UnknownModelError("; ".join(problems))

    def _unservable(self, name: str) -> str | None:
        spec = self._models.get(name)
        if spec is None:
            return f"unknown model {name!r}"
        if spec.provider not in self._adapters:
            return f"no adapter for provider {spec.provider!r} (model {name})"
        if not self._pricing.has(spec.pricing_key):
            return f"no pricing for {spec.pricing_key} (model {name})"
        if problem := self._adapters[spec.provider].check(spec):
            return f"model {name}: {problem}"
        return None

    # -- estimation ------------------------------------------------------------------

    def estimate(self, model_name: str, prompt_tokens: int, *, count_query: bool = True) -> Amount:
        """Worst-case cost of one attempt: estimated prompt plus the full max_tokens."""
        spec = self._spec(model_name)
        out = spec.params.max_tokens
        return Amount(
            queries=1 if count_query else 0,
            tokens=prompt_tokens + out,
            cost_nano=self._pricing.cost_nano(spec.pricing_key, prompt_tokens, out),
        )

    def estimate_messages(self, model_name: str, messages: Sequence[ChatMessage]) -> Amount:
        return self.estimate(model_name, estimate_prompt_tokens(messages))

    def _spec(self, model_name: str) -> ModelSpec:
        problem = self._unservable(model_name)
        if problem:
            raise UnknownModelError(problem)
        return self._models[model_name]

    # -- calling ---------------------------------------------------------------------

    def call(
        self,
        role: Role,
        model_name: str,
        messages: Sequence[ChatMessage],
        *,
        purpose: str,
        seed: int | None = None,
        within: Reservation | None = None,
    ) -> MeteredResponse:
        spec = self._spec(model_name)
        adapter = self._adapters[spec.provider]
        messages = tuple(messages)
        call_id = f"call-{self._n_calls:05d}"
        self._n_calls += 1
        request_ref = self._blobs.put_text(canonical_json([m.model_dump() for m in messages]))
        est_in = estimate_prompt_tokens(messages)
        base = {"call_id": call_id, "role": role.value, "model": model_name, "purpose": purpose}
        identity = {
            "provider": spec.provider,
            "model_id": spec.model_id,
            "model_version": spec.version,
            "deployment": spec.deployment,
            "params": spec.params.model_dump(mode="json") | {"seed": seed},
            "pricing_key": spec.pricing_key,
            "pricing_version": self._pricing.version,
        }

        attempts: list[AttemptRecord] = []
        start_ns = self._clock.monotonic_ns()
        for attempt in range(1, self.retry.max_attempts + 1):
            first = attempt == 1
            est = self.estimate(model_name, est_in, count_query=first)
            label = f"{call_id}#{attempt}"
            try:
                if self.ledger.remaining_wall_s <= 0:
                    raise BudgetDenied(label, ["wall_s"])
                res = self.ledger.reserve(est, label, parent=within)
            except BudgetDenied as denied:
                self._sink.append(
                    EventType.BUDGET_DENIED,
                    base | {"attempt": attempt, "dimensions": denied.dimensions, "requested": _amount_json(est)},
                )
                raise MeteredCallError(call_id, denied, attempts) from None
            self._sink.append(
                EventType.BUDGET_RESERVE,
                base | {"attempt": attempt, "reservation_id": res.id, "amount": _amount_json(est)},
            )

            request = ModelRequest(
                spec=spec,
                messages=messages,
                role=role,
                timeout_s=min(self.retry.timeout_s, self.ledger.remaining_wall_s),
                seed=seed,
            )
            t0 = self._clock.monotonic_ns()
            try:
                response = adapter.complete(request)
                error = None
            except ModelError as exc:
                response, error = None, exc
            latency = (self._clock.monotonic_ns() - t0) / 1e9

            usage = self._actual_usage(messages, response, error, est_in)
            cost = self._pricing.cost_nano(spec.pricing_key, usage.input_tokens, usage.output_tokens)
            actual = Amount(queries=1 if first else 0, tokens=usage.total, cost_nano=cost)
            rec = self.ledger.commit(res, actual)
            attempts.append(
                AttemptRecord(
                    attempt=attempt,
                    outcome="ok" if error is None else error.kind,
                    latency_s=latency,
                    usage=usage,
                    cost_nano=cost,
                )
            )
            response_ref = self._blobs.put_text(response.text) if response is not None else None
            self._sink.append(
                EventType.MODEL_CALL,
                base
                | identity
                | {
                    "attempt": attempt,
                    "outcome": attempts[-1].outcome,
                    # Error kind and HTTP status only: provider messages may echo prompt text.
                    "error_status": getattr(error, "status", None),
                    "latency_s": latency,
                    "input_tokens": usage.input_tokens,
                    "output_tokens": usage.output_tokens,
                    "usage_estimated": usage.estimated,
                    "cost_nano": cost,
                    "finish_reason": response.finish_reason if response else None,
                    "content_filtered": bool(
                        (response and response.content_filtered)
                        or isinstance(error, ModelContentFilterError)
                    ),
                    "filter_categories": list(
                        response.filter_categories
                        if response is not None
                        else getattr(error, "categories", ())
                    ),
                    "provider_model": response.provider_model if response else None,
                    "provider_request_id": response.provider_request_id if response else None,
                    "request_ref": request_ref,
                    "response_ref": response_ref,
                }
                # Which upstream host served a routed call (OpenRouter); absent for direct providers.
                | ({"upstream_provider": response.upstream_provider}
                   if response is not None and response.upstream_provider else {}),
            )
            self._sink.append(
                EventType.BUDGET_RECONCILE,
                base
                | {
                    "attempt": attempt,
                    "reservation_id": res.id,
                    "reserved": _amount_json(rec.reserved),
                    "actual": _amount_json(rec.actual),
                    "overrun": _amount_json(rec.overrun),
                },
            )

            if response is not None:
                return MeteredResponse(
                    call_id=call_id,
                    role=role,
                    model_name=model_name,
                    text=response.text,
                    finish_reason=response.finish_reason,
                    content_filtered=response.content_filtered,
                    filter_categories=response.filter_categories,
                    usage=_sum_usage(attempts),
                    cost_nano=sum(a.cost_nano for a in attempts),
                    latency_s=(self._clock.monotonic_ns() - start_ns) / 1e9,
                    attempts=tuple(attempts),
                    request_ref=request_ref,
                    response_ref=response_ref,
                )
            assert error is not None
            if not error.retryable or attempt == self.retry.max_attempts:
                raise MeteredCallError(call_id, error, attempts)
            delay = self.retry.backoff(attempt, error)
            if delay >= self.ledger.remaining_wall_s:
                raise MeteredCallError(call_id, error, attempts)
            self._sleep(delay)
        raise AssertionError("unreachable")

    @staticmethod
    def _actual_usage(
        messages: tuple[ChatMessage, ...],
        response: ModelResponse | None,
        error: ModelError | None,
        est_in: int,
    ) -> Usage:
        if response is not None:
            if response.usage is not None:
                return response.usage
            return Usage(
                input_tokens=estimate_prompt_tokens(messages),
                output_tokens=estimate_text_tokens(response.text),
                estimated=True,
            )
        assert error is not None
        # Failed attempt: charge the prompt if the provider may have processed it.
        billable = error.input_billable
        return Usage(input_tokens=est_in if billable else 0, output_tokens=0, estimated=billable)


class ScopedClient:
    """A MeteredClient restricted to one role and one model.

    Handed to policies (attacker role) and model-backed evaluators (evaluator role) so they
    can make calls without touching the ledger, other roles, or other models. Calls draw
    from the reservation supplied by the orchestrator.
    """

    def __init__(
        self,
        client: MeteredClient,
        role: Role,
        model_name: str,
        *,
        seed: int | None = None,
        within: Callable[[], Reservation | None] = lambda: None,
    ) -> None:
        self._client = client
        self.role = role
        self.model_name = model_name
        self._seed = seed
        self._within = within

    def call(self, messages: Sequence[ChatMessage], *, purpose: str) -> MeteredResponse:
        return self._client.call(
            self.role, self.model_name, messages, purpose=purpose, seed=self._seed, within=self._within()
        )

    def estimate(self, messages: Sequence[ChatMessage]) -> Amount:
        return self._client.estimate_messages(self.model_name, messages)
