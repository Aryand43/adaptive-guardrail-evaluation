import pytest

from models.adapters.mock import MockAdapter
from models.budget import BudgetLedger, BudgetLimits
from models.errors import (
    ModelBadRequestError,
    ModelContentFilterError,
    ModelRateLimitError,
    ModelServerError,
    ModelTimeoutError,
    UnknownModelError,
)
from models.metered import MeteredCallError, MeteredClient, RetryPolicy
from models.pricing import load_pricing
from models.tokens import estimate_prompt_tokens
from models.types import ChatMessage, ModelSpec, Role, SamplingParams
from storage.blobs import BlobStore
from storage.events import EventSink, EventType, read_events, verify_chain
from tests.conftest import ROOT

PRICING = load_pricing(ROOT / "configs/pricing/mock.yaml")
MSGS = [ChatMessage(role="user", content="[[mock:pressure=5]] PLACEHOLDER request")]
BIG = BudgetLimits(turns=5, queries=50, tokens=100_000, cost_usd=1.0, wall_s=600)


def _spec(model_id, max_tokens=64):
    return ModelSpec(provider="mock", model_id=model_id, version="1", params=SamplingParams(max_tokens=max_tokens))


MODELS = {"target": _spec("mock-target"), "attacker": _spec("mock-attacker"), "judge": _spec("mock-judge")}


@pytest.fixture
def make(tmp_path, clock):
    n = iter(range(100))

    def _make(adapter=None, limits=BIG, models=MODELS, retry=None):
        adapter = adapter or MockAdapter(sleep=clock.advance)
        sink = EventSink(tmp_path / f"ep{next(n)}.jsonl", "run", "ep", clock, durable=False)
        client = MeteredClient(
            models=models,
            adapters={"mock": adapter},
            pricing=PRICING,
            ledger=BudgetLedger(limits, clock),
            clock=clock,
            blobs=BlobStore(tmp_path / "blobs"),
            sink=sink,
            retry=retry,
            sleep=clock.advance,
        )
        client.sink = sink
        return client

    return _make


def events(client):
    return read_events(client.sink.path)


def types(client):
    return [e.event_type for e in events(client)]


def test_successful_call_accounting(make):
    c = make()
    r = c.call(Role.TARGET, "target", MSGS, purpose="t", seed=0)
    assert r.usage.estimated is False and len(r.attempts) == 1
    assert r.cost_nano == PRICING.cost_nano("mock:mock-target:1", r.usage.input_tokens, r.usage.output_tokens)
    assert c.ledger.consumed.queries == 1
    assert c.ledger.consumed.tokens == r.usage.total
    assert c.ledger.consumed.cost_nano == r.cost_nano
    assert c.ledger.outstanding.is_zero
    assert types(c) == [EventType.BUDGET_RESERVE, EventType.MODEL_CALL, EventType.BUDGET_RECONCILE]
    assert r.latency_s == pytest.approx(r.attempts[0].latency_s) and r.latency_s > 0


def test_raw_text_only_in_blob_store(make, tmp_path):
    c = make()
    r = c.call(Role.TARGET, "target", MSGS, purpose="t")
    store = BlobStore(tmp_path / "blobs")
    assert store.get_text(r.response_ref) == r.text
    raw_log = c.sink.path.read_text()
    assert "PLACEHOLDER" not in raw_log and "MOCK-RESPONSE" not in raw_log
    call = events(c)[1].payload
    assert call["response_ref"] == r.response_ref and call["request_ref"] == r.request_ref


def test_unknown_or_unpriced_models_fail_closed_before_any_call(make):
    unpriced = MODELS | {"x": ModelSpec(provider="mock", model_id="unpriced", version="1", params=SamplingParams(max_tokens=8))}
    with pytest.raises(UnknownModelError, match="no pricing"):
        make(models=unpriced)
    no_adapter = MODELS | {"y": ModelSpec(provider="foundry", model_id="mock-target", version="1", params=SamplingParams(max_tokens=8))}
    with pytest.raises(UnknownModelError, match="no adapter"):
        make(models=no_adapter)
    c = make()
    with pytest.raises(UnknownModelError, match="unknown model"):
        c.call(Role.TARGET, "nope", MSGS, purpose="t")
    assert events(c) == []


def test_retry_after_server_error_counts_one_query_and_charges_attempts(make, clock):
    adapter = MockAdapter(sleep=clock.advance, fault_plan=lambda i, r: ModelServerError("503", status=503) if i == 0 else None)
    c = make(adapter)
    r = c.call(Role.TARGET, "target", MSGS, purpose="t")
    assert [a.outcome for a in r.attempts] == ["server_error", "ok"]
    assert c.ledger.consumed.queries == 1
    failed = r.attempts[0]
    assert failed.usage.estimated and failed.usage.input_tokens == estimate_prompt_tokens(MSGS)
    assert c.ledger.consumed.tokens == r.usage.total == failed.usage.total + r.attempts[1].usage.total
    assert r.usage.estimated  # any estimated attempt flags the total
    # deterministic backoff of 1s is included in call latency
    assert r.latency_s == pytest.approx(1.0 + r.attempts[1].latency_s)
    assert types(c).count(EventType.BUDGET_RESERVE) == 2


def test_non_retryable_error_fails_immediately(make):
    c = make(MockAdapter(fault_plan=lambda i, r: ModelBadRequestError("400", status=400)))
    with pytest.raises(MeteredCallError) as exc:
        c.call(Role.TARGET, "target", MSGS, purpose="t")
    assert isinstance(exc.value.error, ModelBadRequestError) and len(exc.value.attempts) == 1
    assert c.ledger.consumed.queries == 1 and c.ledger.consumed.tokens == 0  # not billable
    assert c.ledger.outstanding.is_zero


def test_timeouts_exhaust_retries(make, clock):
    c = make(MockAdapter(sleep=clock.advance, base_latency_s=100.0), retry=RetryPolicy(max_attempts=3, timeout_s=5))
    with pytest.raises(MeteredCallError) as exc:
        c.call(Role.TARGET, "target", MSGS, purpose="t")
    assert isinstance(exc.value.error, ModelTimeoutError)
    assert [a.outcome for a in exc.value.attempts] == ["timeout"] * 3
    assert c.ledger.consumed.queries == 1
    assert c.ledger.consumed.tokens == 3 * estimate_prompt_tokens(MSGS)
    # 3 x 5s timeouts + 1s + 2s backoff
    assert c.ledger.elapsed_s == pytest.approx(18.0)


def test_rate_limit_retry_after_is_respected(make, clock):
    err = ModelRateLimitError("429", retry_after_s=7.0)
    c = make(MockAdapter(fault_plan=lambda i, r: err if i == 0 else None))
    r = c.call(Role.TARGET, "target", MSGS, purpose="t")
    assert r.latency_s == pytest.approx(7.0)
    assert r.attempts[0].usage.input_tokens == 0  # rate-limited prompt is not billed


def test_timeout_is_capped_by_remaining_wall_budget(make, clock):
    limits = BIG.model_copy(update={"wall_s": 3.0})
    c = make(MockAdapter(sleep=clock.advance, base_latency_s=100.0), limits=limits, retry=RetryPolicy(timeout_s=60))
    with pytest.raises(MeteredCallError) as exc:
        c.call(Role.TARGET, "target", MSGS, purpose="t")
    assert len(exc.value.attempts) == 1  # backoff would not fit the remaining wall budget
    assert c.ledger.elapsed_s == pytest.approx(3.0)
    with pytest.raises(MeteredCallError) as exc2:
        c.call(Role.TARGET, "target", MSGS, purpose="t")
    assert exc2.value.budget_denied and exc2.value.error.dimensions == ["wall_s"]


def test_missing_usage_is_estimated_and_flagged(make):
    c = make(MockAdapter(omit_usage=lambda i, r: True))
    r = c.call(Role.TARGET, "target", MSGS, purpose="t")
    assert r.usage.estimated
    assert r.usage.input_tokens == estimate_prompt_tokens(MSGS)
    assert events(c)[1].payload["usage_estimated"] is True


def test_overrun_is_reconciled_and_reported(make):
    models = MODELS | {"target": _spec("mock-target", max_tokens=3)}
    c = make(MockAdapter(ignore_max_tokens=True), models=models)
    r = c.call(Role.TARGET, "target", MSGS, purpose="t")
    reconcile = events(c)[2].payload
    assert reconcile["overrun"]["tokens"] > 0
    assert c.ledger.overrun.tokens == reconcile["overrun"]["tokens"]
    assert c.ledger.consumed.tokens == r.usage.total  # recorded in full


def test_budget_denied_before_any_provider_call(make):
    adapter = MockAdapter()
    tiny = BIG.model_copy(update={"tokens": 10})
    c = make(adapter, limits=tiny)
    with pytest.raises(MeteredCallError) as exc:
        c.call(Role.TARGET, "target", MSGS, purpose="t")
    assert exc.value.budget_denied and exc.value.error.dimensions == ["tokens"]
    assert adapter.n_attempts == 0
    assert types(c) == [EventType.BUDGET_DENIED]
    assert c.ledger.consumed.is_zero


def test_retry_stops_when_its_reservation_does_not_fit(make):
    est = estimate_prompt_tokens(MSGS) + 64
    # Room for the first attempt's reservation and charge, but not a second reservation.
    limits = BIG.model_copy(update={"tokens": est + 10})
    c = make(MockAdapter(fault_plan=lambda i, r: ModelServerError("503")), limits=limits)
    with pytest.raises(MeteredCallError) as exc:
        c.call(Role.TARGET, "target", MSGS, purpose="t")
    assert exc.value.budget_denied and len(exc.value.attempts) == 1
    assert types(c)[-1] == EventType.BUDGET_DENIED


def test_content_filter_is_typed_and_not_retried(make):
    c = make(MockAdapter(filter_at_pressure=1))
    with pytest.raises(MeteredCallError) as exc:
        c.call(Role.TARGET, "target", MSGS, purpose="t")
    assert exc.value.content_filtered and len(exc.value.attempts) == 1
    assert isinstance(exc.value.error, ModelContentFilterError)
    assert events(c)[1].payload["content_filtered"] is True


def test_calls_draw_from_turn_reservation(make):
    c = make()
    turn = c.ledger.reserve(c.estimate_messages("target", MSGS) + c.estimate_messages("judge", MSGS), "turn")
    c.call(Role.TARGET, "target", MSGS, purpose="t", within=turn)
    c.call(Role.EVALUATOR, "judge", MSGS, purpose="j", within=turn)
    c.ledger.release(turn)
    assert c.ledger.outstanding.is_zero and c.ledger.consumed.queries == 2


def test_per_role_latency_is_logged(make):
    c = make()
    c.call(Role.ATTACKER, "attacker", MSGS, purpose="a")
    c.call(Role.TARGET, "target", MSGS, purpose="t")
    calls = [e.payload for e in events(c) if e.event_type == EventType.MODEL_CALL]
    assert [p["role"] for p in calls] == ["attacker", "target"]
    assert all(p["latency_s"] > 0 for p in calls)
    assert verify_chain(c.sink.path).n_events == 6
