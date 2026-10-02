"""Microsoft Foundry adapter (production). One HTTPS attempt per `complete()`; no retries here.

Endpoints are configured only through environment variables, named by `ModelSpec.endpoint_ref`
(default `FOUNDRY`):

    <REF>_ENDPOINT      base URL, e.g. https://<resource>.services.ai.azure.com
    <REF>_API_KEY       key (sent as `api-key`), or use <REF>_BEARER_TOKEN for Entra ID tokens
    <REF>_ROUTE         openai_v1 (default) | azure_openai | model_inference
    <REF>_API_VERSION   required for azure_openai and model_inference routes
    <REF>_TOKEN_PARAM   max_tokens (default) | max_completion_tokens

Routes:
- openai_v1:       POST {endpoint}/openai/v1/chat/completions                (model = deployment)
- azure_openai:    POST {endpoint}/openai/deployments/{deployment}/chat/completions?api-version=..
- model_inference: POST {endpoint}/models/chat/completions?api-version=..    (model = deployment)

Secrets never appear in exceptions, logs or manifests. Provider error bodies may echo prompt
text, so only the HTTP status and the provider's error *code* are surfaced.
"""

import os
from collections.abc import Mapping
from typing import Any, Literal

import httpx
from pydantic import Field, SecretStr

from models.adapters.base import ModelAdapter
from models.errors import (
    ModelAuthError,
    ModelBadRequestError,
    ModelConnectionError,
    ModelContentFilterError,
    ModelError,
    ModelRateLimitError,
    ModelResponseFormatError,
    ModelServerError,
    ModelTimeoutError,
)
from models.types import FinishReason, ModelRequest, ModelResponse, ModelSpec, Usage
from storage.versioning import Frozen

Route = Literal["openai_v1", "azure_openai", "model_inference"]
DEFAULT_REF = "FOUNDRY"
_FINISH: dict[str, FinishReason] = {"stop": "stop", "length": "length", "content_filter": "content_filter"}


class FoundryEndpoint(Frozen):
    ref: str
    base_url: str = Field(pattern=r"^https://")
    api_key: SecretStr | None = None
    bearer_token: SecretStr | None = None
    route: Route = "openai_v1"
    api_version: str | None = None
    token_param: Literal["max_tokens", "max_completion_tokens"] = "max_tokens"

    def url(self, deployment: str) -> str:
        base = self.base_url.rstrip("/")
        if self.route == "openai_v1":
            return f"{base}/openai/v1/chat/completions"
        if self.route == "azure_openai":
            return f"{base}/openai/deployments/{deployment}/chat/completions"
        return f"{base}/models/chat/completions"

    def headers(self) -> dict[str, str]:
        if self.bearer_token is not None:
            return {"Authorization": f"Bearer {self.bearer_token.get_secret_value()}"}
        assert self.api_key is not None
        return {"api-key": self.api_key.get_secret_value()}

    def params(self) -> dict[str, str]:
        return {"api-version": self.api_version} if self.route != "openai_v1" else {}


def endpoint_from_env(ref: str, environ: Mapping[str, str]) -> FoundryEndpoint:
    """Raises ValueError naming only the *variables* that are missing, never their values."""
    get = lambda k: environ.get(f"{ref}_{k}") or None  # noqa: E731
    base = get("ENDPOINT")
    key, token = get("API_KEY"), get("BEARER_TOKEN")
    missing = [f"{ref}_ENDPOINT"] if base is None else []
    if key is None and token is None:
        missing.append(f"{ref}_API_KEY or {ref}_BEARER_TOKEN")
    route = get("ROUTE") or "openai_v1"
    if route in ("azure_openai", "model_inference") and get("API_VERSION") is None:
        missing.append(f"{ref}_API_VERSION")
    if missing:
        raise ValueError("missing environment variables: " + ", ".join(missing))
    try:
        return FoundryEndpoint(
            ref=ref, base_url=base, api_key=key, bearer_token=token, route=route,
            api_version=get("API_VERSION"), token_param=get("TOKEN_PARAM") or "max_tokens",
        )
    except ValueError as exc:
        fields = sorted({str(err["loc"][0]) for err in exc.errors()}) if hasattr(exc, "errors") else []
        raise ValueError(f"invalid Foundry endpoint configuration for {ref}: {', '.join(fields)}") from None


def _filtered_categories(results: Any) -> tuple[str, ...]:
    """Category names flagged `filtered: true` in an Azure content_filter_result(s) object."""
    if not isinstance(results, dict):
        return ()
    return tuple(sorted(k for k, v in results.items() if isinstance(v, dict) and v.get("filtered") is True))


def _retry_after(resp: httpx.Response) -> float | None:
    for header, scale in (("retry-after-ms", 1e-3), ("retry-after", 1.0)):
        value = resp.headers.get(header)
        if value is not None:
            try:
                return float(value) * scale
            except ValueError:
                return None
    return None


def _error_code(resp: httpx.Response) -> tuple[str | None, dict]:
    try:
        err = resp.json().get("error") or {}
    except (ValueError, AttributeError):
        return None, {}
    return (err.get("code") if isinstance(err.get("code"), str) else None), err


def map_http_error(resp: httpx.Response) -> ModelError:
    status = resp.status_code
    code, err = _error_code(resp)
    label = f"HTTP {status}" + (f" ({code})" if code else "")
    if status == 400 and code == "content_filter":
        inner = err.get("innererror") or {}
        cats = _filtered_categories(inner.get("content_filter_result"))
        return ModelContentFilterError(f"prompt rejected by provider content filter: {label}", status=status, categories=cats)
    if status in (401, 403):
        return ModelAuthError(label, status=status)
    if status == 408:
        return ModelTimeoutError(label, status=status)
    if status == 429:
        return ModelRateLimitError(label, status=status, retry_after_s=_retry_after(resp))
    if status >= 500:
        return ModelServerError(label, status=status)
    return ModelBadRequestError(label, status=status)


def parse_chat_completion(body: Any, headers: Mapping[str, str] | None = None) -> ModelResponse:
    """Normalise a chat-completions JSON body. Raises ModelResponseFormatError if malformed."""
    try:
        choice = body["choices"][0]
        message = choice.get("message") or {}
        text = message.get("content") or ""
        if not isinstance(text, str):
            raise TypeError("content is not a string")
        raw_finish = choice.get("finish_reason")
    except (KeyError, IndexError, TypeError, AttributeError) as exc:
        raise ModelResponseFormatError(f"unexpected chat completion shape: {type(exc).__name__}") from None
    finish = _FINISH.get(raw_finish or "", "other")
    categories = _filtered_categories(choice.get("content_filter_results"))
    usage = None
    u = body.get("usage")
    if isinstance(u, dict) and isinstance(u.get("prompt_tokens"), int) and isinstance(u.get("completion_tokens"), int):
        usage = Usage(input_tokens=u["prompt_tokens"], output_tokens=u["completion_tokens"])
    headers = headers or {}
    request_id = body.get("id") or headers.get("apim-request-id") or headers.get("x-request-id")
    return ModelResponse(
        text=text,
        finish_reason=finish,
        usage=usage,
        provider_model=body.get("model"),
        provider_request_id=request_id,
        content_filtered=finish == "content_filter" or bool(categories),
        filter_categories=categories,
    )


class FoundryAdapter(ModelAdapter):
    adapter_id = "foundry"

    def __init__(
        self,
        environ: Mapping[str, str] | None = None,
        *,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        self._environ = environ if environ is not None else os.environ
        self._transport = transport
        self._endpoints: dict[str, FoundryEndpoint] = {}

    @classmethod
    def from_env(cls) -> "FoundryAdapter":
        return cls(os.environ)

    def endpoint(self, spec: ModelSpec) -> FoundryEndpoint:
        ref = spec.endpoint_ref or DEFAULT_REF
        if ref not in self._endpoints:
            self._endpoints[ref] = endpoint_from_env(ref, self._environ)
        return self._endpoints[ref]

    def check(self, spec: ModelSpec) -> str | None:
        try:
            self.endpoint(spec)
        except ValueError as exc:
            return str(exc)
        return None

    def build_request(self, request: ModelRequest) -> tuple[str, dict[str, str], dict[str, str], dict[str, Any]]:
        spec = request.spec
        ep = self.endpoint(spec)
        deployment = spec.deployment or spec.model_id
        body: dict[str, Any] = {
            "messages": [{"role": m.role, "content": m.content} for m in request.messages],
            ep.token_param: spec.params.max_tokens,
            "temperature": spec.params.temperature,
            "top_p": spec.params.top_p,
        }
        if ep.route != "azure_openai":
            body["model"] = deployment
        seed = request.seed if request.seed is not None else spec.params.seed
        if seed is not None:
            body["seed"] = seed
        return ep.url(deployment), ep.params(), ep.headers() | {"Content-Type": "application/json"}, body

    def complete(self, request: ModelRequest) -> ModelResponse:
        try:
            url, params, headers, body = self.build_request(request)
        except ValueError as exc:
            raise ModelAuthError(f"Foundry endpoint not configured: {exc}") from None
        try:
            with httpx.Client(transport=self._transport, timeout=httpx.Timeout(request.timeout_s)) as client:
                resp = client.post(url, params=params, headers=headers, json=body)
        except httpx.TimeoutException:
            raise ModelTimeoutError(f"no response within {request.timeout_s:.1f}s") from None
        except (httpx.ConnectError, httpx.UnsupportedProtocol) as exc:
            raise ModelConnectionError(type(exc).__name__) from None
        except httpx.TransportError as exc:
            raise ModelServerError(type(exc).__name__) from None
        if resp.status_code >= 400:
            raise map_http_error(resp)
        try:
            payload = resp.json()
        except ValueError:
            raise ModelResponseFormatError(f"HTTP {resp.status_code}: body is not JSON") from None
        return parse_chat_completion(payload, resp.headers)
