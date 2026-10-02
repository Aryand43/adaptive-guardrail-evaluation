"""Deterministic token estimation, used for reservations and when providers omit usage.

The estimate is deliberately conservative (~3 chars/token, rounded up, plus per-message
overhead) so reservations rarely undershoot. Any estimated figure is flagged as such.
"""

import math
from collections.abc import Iterable

from models.types import ChatMessage

CHARS_PER_TOKEN = 3
PER_MESSAGE_OVERHEAD = 4


def estimate_text_tokens(text: str) -> int:
    return math.ceil(len(text) / CHARS_PER_TOKEN)


def estimate_prompt_tokens(messages: Iterable[ChatMessage]) -> int:
    return sum(estimate_text_tokens(m.content) + PER_MESSAGE_OVERHEAD for m in messages) + 2
