"""Canonical JSON serialisation and SHA-256 helpers.

Every hash in the system (config, events, manifests, blobs) goes through these
functions so that identical content always yields an identical hash.
"""

import hashlib
import json
import re
from typing import Any

from pydantic import BaseModel

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
GENESIS_HASH = "0" * 64


def to_jsonable(obj: Any) -> Any:
    if isinstance(obj, BaseModel):
        return obj.model_dump(mode="json")
    return obj


def canonical_json(obj: Any) -> str:
    """Deterministic JSON: sorted keys, no whitespace, UTF-8, NaN/Infinity rejected."""
    return json.dumps(
        to_jsonable(obj),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_text(text: str) -> str:
    return sha256_bytes(text.encode("utf-8"))


def hash_obj(obj: Any) -> str:
    return sha256_text(canonical_json(obj))


def is_sha256(value: Any) -> bool:
    return isinstance(value, str) and bool(_SHA256_RE.match(value))
