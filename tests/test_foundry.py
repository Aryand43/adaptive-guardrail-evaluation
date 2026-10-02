"""Foundry adapter tests: sanitized fixtures over httpx.MockTransport. No network."""

import json

import httpx
import pytest

from models.adapters.foundry import FoundryAdapter, endpoint_from_env, parse_chat_completion
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

FIX = ROOT / "tests/fixtures/foundry"
SECRET = "sk-SANITIZED-TEST-KEY-do-not-log"
ENV = {"FOUNDRY_ENDPOINT": "https://example-resource.services.ai.azure.com", "FOUNDRY_API_KEY": SECRET}
SPEC = ModelSpec(provider="foundry", model_id="gpt-4o-mini", version="2024-07-18", deployment="dev-mini",
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
    return FoundryAdapter(env, transport=transport(*names, seen=seen))


# -- parsing -----------------------------------------------------------------------------


def test_parse_ok():
    f = fixture("chat_ok")
    r = parse_chat_completion(f["body"], f["headers"])
    assert r.text == "PLACEHOLDER sanitized assistant reply."
    assert r.finish_reason == "stop" and not r.content_filtered and r.filter_categories == ()
    assert (r.usage.input_tokens, r.usage.output_tokens, r.usage.estimated) == (21, 6, False)
    assert r.provider_model == "gpt-4o-mini-2024-07-18" and r.provider_request_id == "chatcmpl-SANITIZED0001"


def test_parse_length_and_missing_usage():
    assert parse_chat_completion(fixture("chat_length")["body"]).finish_reason == "length"
    f = fixture("chat_missing_usage")
    r = parse_chat_completion(f["body"], f["headers"])
    assert r.usage is None and r.provider_request_id == "00000000-0000-0000-0000-000000000003"


def test_parse_output_content_filter():
    r = parse_chat_completion(fixture("chat_output_filtered")["body"])
    assert r.text == "" and r.finish_reason == "content_filter"
    assert r.content_filtered and r.filter_categories == ("violence",)


def test_parse_malformed_fails_closed():
    with pytest.raises(ModelResponseFormatError):
        parse_chat_completion(fixture("chat_malformed")["body"])
    with pytest.raises(ModelResponseFormatError):
        parse_chat_completion({"choices": [{"message": {"content": 5}}]})


# -- request building and HTTP ------------------------------------------------------------


def test_request_shape_openai_v1_route():
    seen = []
    resp = adapter("chat_ok", seen=seen).complete(req())
    assert resp.usage.total == 27
    r = seen[0]
    assert str(r.url) == "https://example-resource.services.ai.azure.com/openai/v1/chat/completions"
    assert r.headers["api-key"] == SECRET
    body = json.loads(r.content)
    assert body == {"messages": [{"role": "user", "content": "PLACEHOLDER"}], "max_tokens": 64,
                    "temperature": 0.0, "top_p": 1.0, "model": "dev-mini", "seed": 7}


def test_azure_openai_route_and_completion_token_param():
    env = ENV | {"FOUNDRY_ROUTE": "azure_openai", "FOUNDRY_API_VERSION": "2024-10-21",
                 "FOUNDRY_TOKEN_PARAM": "max_completion_tokens"}
    seen = []
    adapter("chat_ok", env=env, seen=seen).complete(req())
    r = seen[0]
    assert r.url.path == "/openai/deployments/dev-mini/chat/completions"
    assert r.url.params["api-version"] == "2024-10-21"
    body = json.loads(r.content)
    assert "model" not in body and body["max_completion_tokens"] == 64 and "max_tokens" not in body


def test_model_inference_route_and_bearer_auth():
    env = {"FOUNDRY_ENDPOINT": ENV["FOUNDRY_ENDPOINT"], "FOUNDRY_BEARER_TOKEN": "tok-SANITIZED",
           "FOUNDRY_ROUTE": "model_inference", "FOUNDRY_API_VERSION": "2024-05-01-preview"}
    seen = []
    adapter("chat_ok", env=env, seen=seen).complete(req())
    assert seen[0].url.path == "/models/chat/completions"
    assert seen[0].headers["authorization"] == "Bearer tok-SANITIZED" and "api-key" not in seen[0].headers


def test_endpoint_ref_selects_env_prefix():
    spec = SPEC.model_copy(update={"endpoint_ref": "FOUNDRY_EVAL"})
    env = {"FOUNDRY_EVAL_ENDPOINT": "https://eval.services.ai.azure.com", "FOUNDRY_EVAL_API_KEY": "k"}
    seen = []
    adapter("chat_ok", env=env, seen=seen).complete(req(spec))
    assert seen[0].url.host == "eval.services.ai.azure.com"


@pytest.mark.parametrize(
    "name, exc, check",
    [
        ("error_prompt_filtered", ModelContentFilterError, lambda e: e.categories == ("jailbreak", "self_harm")),
        ("error_rate_limit", ModelRateLimitError, lambda e: e.retry_after_s == 4.0),
        ("error_auth", ModelAuthError, lambda e: e.status == 401),
        ("error_server", ModelServerError, lambda e: e.retryable),
        ("error_not_found", ModelBadRequestError, lambda e: "DeploymentNotFound" in str(e)),
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
        return FoundryAdapter(ENV, transport=httpx.MockTransport(handler))

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
        FoundryAdapter(ENV, transport=t).complete(req())


def test_missing_configuration_fails_preflight_without_leaking_values():
    a = FoundryAdapter({"FOUNDRY_API_KEY": SECRET})
    problem = a.check(SPEC)
    assert "FOUNDRY_ENDPOINT" in problem and SECRET not in problem
    assert "API_VERSION" in endpoint_problem({"FOUNDRY_ENDPOINT": "https://x", "FOUNDRY_API_KEY": "k", "FOUNDRY_ROUTE": "azure_openai"})
    assert "base_url" in endpoint_problem({"FOUNDRY_ENDPOINT": "http://insecure", "FOUNDRY_API_KEY": SECRET})


def endpoint_problem(env):
    try:
        endpoint_from_env("FOUNDRY", env)
    except ValueError as exc:
        assert SECRET not in str(exc)
        return str(exc)
    return ""


def test_secret_is_masked_in_endpoint_repr():
    ep = endpoint_from_env("FOUNDRY", ENV)
    assert SECRET not in repr(ep) and SECRET not in ep.model_dump_json()


# -- through MeteredClient ---------------------------------------------------------------

PRICING = PricingTable(version="test-foundry", effective_date="2026-01-01",
                       models={SPEC.pricing_key: {"input_per_mtok": "0.15", "output_per_mtok": "0.60"}})


def metered(tmp_path, clock, a):
    sink = EventSink(tmp_path / "ep.jsonl", "run", "ep", clock, durable=False)
    return MeteredClient(
        models={"t": SPEC}, adapters={"foundry": a}, pricing=PRICING,
        ledger=BudgetLedger(BudgetLimits(turns=1, queries=5, tokens=10_000, cost_usd=1, wall_s=60), clock),
        clock=clock, blobs=BlobStore(tmp_path / "blobs"), sink=sink, sleep=clock.advance,
    ), sink


def test_metered_foundry_call_reconciles_provider_usage(tmp_path, clock):
    c, sink = metered(tmp_path, clock, adapter("error_rate_limit", "chat_ok"))
    r = c.call(Role.TARGET, "t", [ChatMessage(role="user", content="PLACEHOLDER")], purpose="t", seed=1)
    assert [a.outcome for a in r.attempts] == ["rate_limit", "ok"]
    assert r.usage.input_tokens == 21 and r.usage.output_tokens == 6 and not r.usage.estimated
    assert c.ledger.consumed.cost_nano == PRICING.cost_nano(SPEC.pricing_key, 21, 6)
    assert clock.monotonic_ns() == 4_000_000_000  # honoured Retry-After
    log = sink.path.read_text()
    assert SECRET not in log and "example-resource" not in log
    call = [e.payload for e in read_events(sink.path) if e.event_type == "model_call"][-1]
    assert call["provider_request_id"] == "chatcmpl-SANITIZED0001" and call["deployment"] == "dev-mini"


def test_metered_client_refuses_unconfigured_foundry_model(tmp_path, clock):
    with pytest.raises(UnknownModelError, match="FOUNDRY_ENDPOINT"):
        metered(tmp_path, clock, FoundryAdapter({}))
