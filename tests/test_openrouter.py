"""OpenRouter adapter tests: sanitized fixtures over httpx.MockTransport. No network."""

import json

import httpx
import pytest

from models.adapters.openrouter import OpenRouterAdapter, endpoint_from_env, parse_chat_completion
from models.budget import BudgetLedger, BudgetLimits
from models.errors import (
    ModelAuthError,
    ModelBadRequestError,
    ModelConnectionError,
    ModelContentFilterError,
    ModelRateLimitError,
    ModelResponseFormatError,
    ModelServerError,
    ModelTimeoutError,
    UnknownModelError,
)
from models.metered import MeteredClient
from models.pricing import PricingTable
from models.types import ChatMessage, ModelRequest, ModelSpec, Role, SamplingParams
from storage.blobs import BlobStore
from storage.events import EventSink, read_events
from tests.conftest import ROOT

FIX = ROOT / "tests/fixtures/openrouter"
SECRET = "sk-or-SANITIZED-TEST-KEY-do-not-log"
ENV = {"OPENROUTER_API_KEY": SECRET}
SPEC = ModelSpec(provider="openrouter", model_id="openai/gpt-6-luna", version="openai/gpt-6-luna-20260922",
                 params=SamplingParams(max_tokens=64, temperature=0.0))


def fixture(name):
    return json.loads((FIX / f"{name}.json").read_text())


def transport(*names, seen=None):
    queue = [fixture(n) for n in names]

    def handler(request: httpx.Request) -> httpx.Response:
        if seen is not None:
            seen.append(request)
        f = queue.pop(0) if len(queue) > 1 else queue[0]
        return httpx.Response(f["status"], headers=f["headers"], json=f["body"])

    return httpx.MockTransport(handler)


def req(spec=SPEC, seed=7):
    return ModelRequest(spec=spec, messages=(ChatMessage(role="user", content="PLACEHOLDER"),),
                        role=Role.TARGET, timeout_s=10, seed=seed)


def adapter(*names, env=ENV, seen=None):
    return OpenRouterAdapter(env, transport=transport(*names, seen=seen))


# -- parsing -----------------------------------------------------------------------------


def test_parse_ok():
    f = fixture("chat_ok")
    r = parse_chat_completion(f["body"], f["headers"])
    assert r.text == "PLACEHOLDER sanitized assistant reply."
    assert r.finish_reason == "stop" and not r.content_filtered
    assert (r.usage.input_tokens, r.usage.output_tokens, r.usage.estimated) == (21, 6, False)
    assert r.provider_model == "openai/gpt-6-luna" and r.provider_request_id == "gen-SANITIZED0001"
    assert r.upstream_provider == "OpenAI"


def test_parse_length_and_missing_usage():
    assert parse_chat_completion(fixture("chat_length")["body"]).finish_reason == "length"
    f = fixture("chat_missing_usage")
    r = parse_chat_completion(f["body"], f["headers"])
    assert r.usage is None and r.provider_request_id == "req-SANITIZED0003" and r.upstream_provider is None


def test_parse_output_content_filter():
    r = parse_chat_completion(fixture("chat_output_filtered")["body"])
    assert r.text == "" and r.finish_reason == "content_filter" and r.content_filtered


def test_upstream_error_in_200_body_is_a_retryable_server_error():
    with pytest.raises(ModelServerError) as info:
        parse_chat_completion(fixture("chat_upstream_error")["body"])
    assert info.value.retryable and "SANITIZED:" not in str(info.value)


def test_parse_malformed_fails_closed():
    with pytest.raises(ModelResponseFormatError):
        parse_chat_completion(fixture("chat_malformed")["body"])
    with pytest.raises(ModelResponseFormatError):
        parse_chat_completion({"choices": [{"message": {"content": 5}}]})


# -- request building and HTTP ------------------------------------------------------------


def test_request_shape():
    seen = []
    resp = adapter("chat_ok", seen=seen).complete(req())
    assert resp.usage.total == 27
    r = seen[0]
    assert str(r.url) == "https://openrouter.ai/api/v1/chat/completions"
    assert r.headers["authorization"] == f"Bearer {SECRET}"
    body = json.loads(r.content)
    assert body == {"model": "openai/gpt-6-luna", "messages": [{"role": "user", "content": "PLACEHOLDER"}],
                    "max_tokens": 64, "temperature": 0.0, "top_p": 1.0, "seed": 7}


def test_endpoint_ref_selects_env_prefix_and_base_url():
    spec = SPEC.model_copy(update={"endpoint_ref": "OPENROUTER_EVAL"})
    env = {"OPENROUTER_EVAL_API_KEY": "k", "OPENROUTER_EVAL_BASE_URL": "https://gateway.example.com/v1"}
    seen = []
    adapter("chat_ok", env=env, seen=seen).complete(req(spec))
    assert str(seen[0].url) == "https://gateway.example.com/v1/chat/completions"


@pytest.mark.parametrize(
    "name, exc, check",
    [
        ("error_moderation", ModelContentFilterError, lambda e: e.categories == ("harassment", "violence")),
        ("error_rate_limit", ModelRateLimitError, lambda e: e.retry_after_s == 4.0),
        ("error_auth", ModelAuthError, lambda e: e.status == 401),
        ("error_credits", ModelAuthError, lambda e: e.status == 402 and not e.retryable),
        ("error_server", ModelServerError, lambda e: e.retryable),
        ("error_bad_model", ModelBadRequestError, lambda e: "invalid_model" in str(e)),
    ],
)
def test_http_errors_map_to_typed_errors(name, exc, check):
    with pytest.raises(exc) as info:
        adapter(name).complete(req())
    assert check(info.value)
    assert "SANITIZED:" not in str(info.value)  # provider messages (may echo prompts) are dropped
    assert SECRET not in str(info.value)


def test_transport_failures():
    def raiser(error):
        def handler(request):
            raise error
        return OpenRouterAdapter(ENV, transport=httpx.MockTransport(handler))

    with pytest.raises(ModelTimeoutError):
        raiser(httpx.ReadTimeout("x")).complete(req())
    with pytest.raises(ModelConnectionError) as e:
        raiser(httpx.ConnectError("x")).complete(req())
    assert not e.value.input_billable and e.value.retryable
    with pytest.raises(ModelServerError):
        raiser(httpx.RemoteProtocolError("x")).complete(req())


def test_non_json_body():
    t = httpx.MockTransport(lambda r: httpx.Response(200, text="<html>gateway</html>"))
    with pytest.raises(ModelResponseFormatError):
        OpenRouterAdapter(ENV, transport=t).complete(req())


def test_missing_configuration_fails_preflight_without_leaking_values():
    assert "OPENROUTER_API_KEY" in OpenRouterAdapter({}).check(SPEC)
    with pytest.raises(ValueError) as info:
        endpoint_from_env("OPENROUTER", ENV | {"OPENROUTER_BASE_URL": "http://insecure"})
    assert "base_url" in str(info.value) and SECRET not in str(info.value)


def test_secret_is_masked_in_endpoint_repr():
    ep = endpoint_from_env("OPENROUTER", ENV)
    assert SECRET not in repr(ep) and SECRET not in ep.model_dump_json()


# -- through MeteredClient ---------------------------------------------------------------

PRICING = PricingTable(version="test-openrouter", effective_date="2026-01-01",
                       models={SPEC.pricing_key: {"input_per_mtok": "0.10", "output_per_mtok": "0.50"}})


def metered(tmp_path, clock, a):
    sink = EventSink(tmp_path / "ep.jsonl", "run", "ep", clock, durable=False)
    return MeteredClient(
        models={"t": SPEC}, adapters={"openrouter": a}, pricing=PRICING,
        ledger=BudgetLedger(BudgetLimits(turns=1, queries=5, tokens=10_000, cost_usd=1, wall_s=60), clock),
        clock=clock, blobs=BlobStore(tmp_path / "blobs"), sink=sink, sleep=clock.advance,
    ), sink


def test_metered_openrouter_call_reconciles_provider_usage(tmp_path, clock):
    c, sink = metered(tmp_path, clock, adapter("error_rate_limit", "chat_ok"))
    r = c.call(Role.TARGET, "t", [ChatMessage(role="user", content="PLACEHOLDER")], purpose="t", seed=1)
    assert [a.outcome for a in r.attempts] == ["rate_limit", "ok"]
    assert r.usage.input_tokens == 21 and r.usage.output_tokens == 6 and not r.usage.estimated
    assert c.ledger.consumed.cost_nano == PRICING.cost_nano(SPEC.pricing_key, 21, 6)
    assert clock.monotonic_ns() == 4_000_000_000  # honoured Retry-After
    assert SECRET not in sink.path.read_text()
    call = [e.payload for e in read_events(sink.path) if e.event_type == "model_call"][-1]
    assert call["provider_request_id"] == "gen-SANITIZED0001" and call["upstream_provider"] == "OpenAI"


def test_metered_client_refuses_unconfigured_openrouter_model(tmp_path, clock):
    with pytest.raises(UnknownModelError, match="OPENROUTER_API_KEY"):
        metered(tmp_path, clock, OpenRouterAdapter({}))
