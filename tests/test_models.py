import pytest
from pydantic import ValidationError

from models.adapters.mock import MockAdapter
from models.errors import ModelContentFilterError, ModelServerError, ModelTimeoutError
from models.pricing import (
    PricingTable,
    UnknownModelPricingError,
    load_pricing,
    nano_to_usd,
    usd_to_nano,
)
from models.tokens import estimate_prompt_tokens
from models.types import ChatMessage, ModelRequest, ModelSpec, Role, SamplingParams
from tests.conftest import ROOT

MOCK_PRICING = ROOT / "configs/pricing/mock.yaml"


def spec(model_id="mock-target", max_tokens=64):
    return ModelSpec(
        provider="mock", model_id=model_id, version="1", params=SamplingParams(max_tokens=max_tokens)
    )


def req(text, role=Role.TARGET, seed=0, model_id="mock-target", max_tokens=64, timeout_s=30.0):
    return ModelRequest(
        spec=spec(model_id, max_tokens),
        messages=(ChatMessage(role="user", content=text),),
        role=role,
        timeout_s=timeout_s,
        seed=seed,
    )


# -- mock adapter ------------------------------------------------------------------------


def test_mock_is_deterministic_across_instances():
    r = req("[[mock:pressure=4]] PLACEHOLDER")
    a, b = MockAdapter().complete(r), MockAdapter().complete(r)
    assert a.text == b.text and a.usage == b.usage


def test_mock_seed_changes_noise_but_not_structure():
    texts = {MockAdapter().complete(req("[[mock:pressure=5]] X", seed=s)).text for s in range(8)}
    assert len(texts) > 1
    assert all(t.startswith("[[mock:") for t in texts)


def test_mock_target_refuses_without_pressure_and_complies_under_pressure():
    a = MockAdapter()
    assert a.complete(req("PLACEHOLDER no markers")).text.startswith("[[mock:refusal]]")
    assert "[[mock:compliance=1.00]]" in a.complete(req("[[mock:pressure=10]] X")).text


def test_mock_attacker_escalates_from_feedback():
    a = MockAdapter()
    first = a.complete(req("no feedback yet", role=Role.ATTACKER)).text
    assert "[[mock:pressure=1]]" in first
    after_refusal = a.complete(
        req("[[mock:pressure=1]] X\nFEEDBACK: turn=1 refused=1 progress=0.0", role=Role.ATTACKER)
    ).text
    assert "[[mock:pressure=3]]" in after_refusal


@pytest.mark.parametrize(
    "rubric, body, expected",
    [
        ("harm", "[[mock:compliance=0.73]] MOCK", "0.73"),
        ("progress", "[[mock:compliance=0.73]] MOCK", "0.70"),
        ("refusal", "[[mock:refusal]] MOCK", "1.00"),
        ("refusal", "[[mock:compliance=0.50]] MOCK", "0.00"),
    ],
)
def test_mock_judge(rubric, body, expected):
    out = MockAdapter().complete(req(f"[[rubric:{rubric}]]\n{body}", role=Role.EVALUATOR)).text
    assert out.startswith(f"SCORE: {expected}")


def test_mock_truncates_at_max_tokens_unless_overrun_mode():
    r = req("[[mock:pressure=10]] X", max_tokens=5)
    cut = MockAdapter().complete(r)
    assert cut.finish_reason == "length" and cut.usage.output_tokens == 5
    over = MockAdapter(ignore_max_tokens=True).complete(r)
    assert over.usage.output_tokens > 5


def test_mock_fault_injection_and_missing_usage():
    a = MockAdapter(
        fault_plan=lambda i, r: ModelServerError("boom") if i == 0 else None,
        omit_usage=lambda i, r: True,
    )
    with pytest.raises(ModelServerError):
        a.complete(req("X"))
    resp = a.complete(req("X"))
    assert resp.usage is None


def test_mock_timeout_advances_fake_clock(clock):
    a = MockAdapter(sleep=clock.advance, base_latency_s=5.0)
    with pytest.raises(ModelTimeoutError):
        a.complete(req("X", timeout_s=1.0))
    assert clock.monotonic_ns() == 1_000_000_000


def test_mock_prompt_filter():
    with pytest.raises(ModelContentFilterError) as exc:
        MockAdapter(filter_at_pressure=3).complete(req("[[mock:pressure=3]] X"))
    assert exc.value.categories == ("mock_category",)


def test_requests_and_responses_are_immutable():
    r = req("X")
    with pytest.raises(ValidationError):
        r.timeout_s = 1  # type: ignore[misc]


def test_token_estimate_is_conservative_vs_mock_usage():
    r = req("[[mock:pressure=2]] PLACEHOLDER words for an estimate check")
    actual = MockAdapter().complete(r).usage.input_tokens
    assert estimate_prompt_tokens(r.messages) >= actual


# -- pricing -----------------------------------------------------------------------------


def test_pricing_cost_is_exact_integer_nano_usd():
    t = load_pricing(MOCK_PRICING)
    # 1000 in * $0.10/M + 500 out * $0.40/M = $0.0001 + $0.0002 = 300_000 nano-USD
    assert t.cost_nano("mock:mock-target:1", 1000, 500) == 300_000
    assert nano_to_usd(300_000) == pytest.approx(0.0003)
    assert usd_to_nano(1.0) == 10**9


def test_pricing_rounds_up_per_call():
    t = PricingTable(
        version="t", effective_date="2026-01-01", models={"m": {"input_per_mtok": "0.0001", "output_per_mtok": "0"}}
    )
    assert t.cost_nano("m", 1, 0) == 1  # 0.1 nano -> 1


def test_unknown_model_fails_closed():
    t = load_pricing(MOCK_PRICING)
    with pytest.raises(UnknownModelPricingError):
        t.cost_nano("foundry:unpriced:1", 1, 1)
    assert not t.has("foundry:unpriced:1")


def test_pricing_rejects_float_prices_and_wrong_currency():
    with pytest.raises(ValidationError, match="decimal strings"):
        PricingTable(version="t", effective_date="d", models={"m": {"input_per_mtok": 0.1, "output_per_mtok": "0"}})
    with pytest.raises(ValidationError, match="USD"):
        PricingTable(version="t", effective_date="d", currency="EUR", models={})


def test_pricing_hash_is_stable_and_content_sensitive():
    a = load_pricing(MOCK_PRICING)
    assert a.table_hash == load_pricing(MOCK_PRICING).table_hash
    b = a.model_copy(update={"version": "other"})
    assert a.table_hash != b.table_hash


def test_foundry_template_has_no_prices():
    assert load_pricing(ROOT / "configs/pricing/foundry.TEMPLATE.yaml").models == {}
