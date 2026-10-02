"""Typed model-call errors. Adapters translate provider failures into exactly one of these.

`retryable` drives MeteredClient's retry loop. `input_billable` says whether the provider
may have consumed (and billed) the prompt before failing; accounting charges an estimate.
"""


class ModelError(Exception):
    kind = "error"
    retryable = False
    input_billable = False

    def __init__(self, message: str = "", *, status: int | None = None) -> None:
        super().__init__(message or self.kind)
        self.status = status


class ModelTimeoutError(ModelError):
    kind = "timeout"
    retryable = True
    input_billable = True  # the request may have been processed server-side


class ModelRateLimitError(ModelError):
    kind = "rate_limit"
    retryable = True

    def __init__(self, message: str = "", *, status: int | None = 429, retry_after_s: float | None = None):
        super().__init__(message, status=status)
        self.retry_after_s = retry_after_s


class ModelServerError(ModelError):
    kind = "server_error"
    retryable = True
    input_billable = True


class ModelConnectionError(ModelServerError):
    """The request never reached the provider (DNS, connect, TLS). Retryable, not billable."""

    kind = "connection_error"
    input_billable = False


class ModelBadRequestError(ModelError):
    kind = "bad_request"


class ModelAuthError(ModelError):
    kind = "auth"


class ModelResponseFormatError(ModelError):
    """The provider answered but the body could not be parsed into a ModelResponse."""

    kind = "bad_response"
    input_billable = True


class ModelContentFilterError(ModelError):
    """The provider's safety filter rejected the *prompt* (no completion was produced).

    Not a failure of the harness: for a target model this is a guardrail block and is
    recorded as such. Never retried.
    """

    kind = "content_filter"

    def __init__(self, message: str = "", *, status: int | None = 400, categories: tuple[str, ...] = ()):
        super().__init__(message, status=status)
        self.categories = categories


class UnknownModelError(ModelError):
    """No adapter or no pricing for a model. Fails closed before any call is made."""

    kind = "unknown_model"
