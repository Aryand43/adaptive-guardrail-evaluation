"""OpenRouter adapter (production). One HTTPS attempt per `complete()`; no retries here.

Endpoints are configured only through environment variables, named by `ModelSpec.endpoint_ref`
(default `OPENROUTER`):

    <REF>_API_KEY       key (sent as `Authorization: Bearer`)
    <REF>_BASE_URL      optional, default https://openrouter.ai/api/v1

Every model is called through the OpenAI-compatible route `POST {base_url}/chat/completions`
with `model = ModelSpec.model_id` (an OpenRouter slug such as `openai/gpt-6-luna`). OpenRouter
drops sampling parameters a model does not support instead of rejecting the request.

Secrets never appear in exceptions, logs or manifests. Provider error bodies may echo prompt
text, so only the HTTP status and the provider's error *code* are surfaced.
"""

import os
from collections.abc import Mapping
from typing import Any

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

DEFAULT_REF = "OPENROUTER"
DEFAULT_BASE_URL = "https://openrouter.ai/api/v1"
_FINISH: dict[str, FinishReason] = {"stop": "stop", "length": "length", "content_filter": "content_filter"}


class OpenRouterEndpoint(Frozen):
    ref: str
    base_url: str = Field(pattern=r"^https://")
    api_key: SecretStr

    def url(self) -> str:
        return f"{self.base_url.rstrip('/')}/chat/completions"

    def headers(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {self.api_key.get_secret_value()}"}


def endpoint_from_env(ref: str, environ: Mapping[str, str]) -> OpenRouterEndpoint:
    """Raises ValueError naming only the *variables* that are missing, never their values."""
    key = environ.get(f"{ref}_API_KEY") or None
    if key is None:
        raise ValueError(f"missing environment variables: {ref}_API_KEY")
    try:
        return OpenRouterEndpoint(ref=ref, base_url=environ.get(f"{ref}_BASE_URL") or DEFAULT_BASE_URL, api_key=key)
    except ValueError as exc:
        fields = sorted({str(err["loc"][0]) for err in exc.errors()}) if hasattr(exc, "errors") else []
        raise ValueError(f"invalid OpenRouter endpoint configuration for {ref}: {', '.join(fields)}") from None


def _retry_after(resp: httpx.Response) -> float | None:
    value = resp.headers.get("retry-after")
    if value is None:
        return None
    try:
        return float(value)
    except ValueError:
        return None


def _error(resp: httpx.Response) -> dict:
    try:
        err = resp.json().get("error") or {}
    except (ValueError, AttributeError):
        return {}
    return err if isinstance(err, dict) else {}


def _moderation_reasons(err: dict) -> tuple[str, ...]:
    reasons = (err.get("metadata") or {}).get("reasons")
    if not isinstance(reasons, list):
        return ()
    return tuple(sorted(r for r in reasons if isinstance(r, str)))


def map_http_error(resp: httpx.Response) -> ModelError:
    status = resp.status_code
    err = _error(resp)
    code = err.get("code")
    label = f"HTTP {status}" + (f" ({code})" if isinstance(code, str) and code else "")
    if status == 403 and (err.get("metadata") or {}).get("reasons") is not None:
        # OpenRouter moderation flagged the input before it reached the model.
        return ModelContentFilterError(f"prompt rejected by provider moderation: {label}", status=status,
                                       categories=_moderation_reasons(err))
    if status == 402:
        return ModelAuthError(f"{label} (insufficient credits)", status=status)
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
        if choice.get("error") or choice.get("finish_reason") == "error":
            # The upstream provider failed mid-generation; OpenRouter still returns HTTP 200.
            raise ModelServerError("upstream provider error during generation")
        message = choice.get("message") or {}
        text = message.get("content") or ""
        if not isinstance(text, str):
            raise TypeError("content is not a string")
        raw_finish = choice.get("finish_reason")
    except (KeyError, IndexError, TypeError, AttributeError) as exc:
        raise ModelResponseFormatError(f"unexpected chat completion shape: {type(exc).__name__}") from None
    finish = _FINISH.get(raw_finish or "", "other")
    usage = None
    u = body.get("usage")
    if isinstance(u, dict) and isinstance(u.get("prompt_tokens"), int) and isinstance(u.get("completion_tokens"), int):
        usage = Usage(input_tokens=u["prompt_tokens"], output_tokens=u["completion_tokens"])
    headers = headers or {}
    upstream = body.get("provider")
    return ModelResponse(
        text=text,
        finish_reason=finish,
        usage=usage,
        provider_model=body.get("model"),
        provider_request_id=body.get("id") or headers.get("x-request-id"),
        upstream_provider=upstream if isinstance(upstream, str) else None,
        content_filtered=finish == "content_filter",
    )


class OpenRouterAdapter(ModelAdapter):
    adapter_id = "openrouter"

    def __init__(
        self,
        environ: Mapping[str, str] | None = None,
        *,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        self._environ = environ if environ is not None else os.environ
        self._transport = transport
        self._endpoints: dict[str, OpenRouterEndpoint] = {}

    @classmethod
    def from_env(cls) -> "OpenRouterAdapter":
        return cls(os.environ)

    def endpoint(self, spec: ModelSpec) -> OpenRouterEndpoint:
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

    def build_request(self, request: ModelRequest) -> tuple[str, dict[str, str], dict[str, Any]]:
        spec = request.spec
        ep = self.endpoint(spec)
        body: dict[str, Any] = {
            "model": spec.deployment or spec.model_id,
            "messages": [{"role": m.role, "content": m.content} for m in request.messages],
            "max_tokens": spec.params.max_tokens,
            "temperature": spec.params.temperature,
            "top_p": spec.params.top_p,
        }
        seed = request.seed if request.seed is not None else spec.params.seed
        if seed is not None:
            body["seed"] = seed
        return ep.url(), ep.headers() | {"Content-Type": "application/json"}, body

    def complete(self, request: ModelRequest) -> ModelResponse:
        try:
            url, headers, body = self.build_request(request)
        except ValueError as exc:
            raise ModelAuthError(f"OpenRouter endpoint not configured: {exc}") from None
        try:
            with httpx.Client(transport=self._transport, timeout=httpx.Timeout(request.timeout_s)) as client:
                resp = client.post(url, headers=headers, json=body)
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
