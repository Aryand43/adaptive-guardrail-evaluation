"""Append-only, hash-chained event log. One chain (JSONL file) per episode.

Each event's hash covers its whole envelope including `prev_hash`, so any edit,
deletion, reordering or insertion breaks verification from that point on.
Raw text is never stored here; payload fields ending in `_ref` hold blob hashes.
"""

import os
from enum import StrEnum
from pathlib import Path
from typing import Any

from pydantic import Field

from storage.clock import Clock, iso_utc
from storage.errors import ChainSealedError, IntegrityError, RawTextInEventError
from storage.hashing import GENESIS_HASH, canonical_json, hash_obj, is_sha256
from storage.versioning import Frozen, Versioned

# Longest string allowed anywhere in a payload. Model text must go to the BlobStore.
MAX_PAYLOAD_STRING = 512


class EventType(StrEnum):
    EPISODE_START = "episode_start"
    BUDGET_RESERVE = "budget_reserve"
    BUDGET_DENIED = "budget_denied"
    MODEL_CALL = "model_call"
    BUDGET_RECONCILE = "budget_reconcile"
    ATTACKER_ACTION = "attacker_action"
    FILTER_DECISION = "filter_decision"
    TARGET_RESPONSE = "target_response"
    EVALUATION = "evaluation"
    STATE_TRANSITION = "state_transition"
    STOP = "stop"
    EPISODE_RECORD = "episode_record"
    ERROR = "error"
    EPISODE_SEAL = "episode_seal"


class Event(Versioned):
    run_id: str
    episode_id: str
    seq: int = Field(ge=0)
    event_id: str
    ts_utc: str
    mono_ns: int
    event_type: EventType
    payload: dict[str, Any]
    prev_hash: str
    hash: str

    def compute_hash(self) -> str:
        return hash_obj(self.model_dump(mode="json", exclude={"hash"}))


class ChainInfo(Frozen):
    run_id: str
    episode_id: str
    head_hash: str
    n_events: int
    sealed: bool


def assert_no_raw_text(payload: Any, path: str = "payload") -> None:
    """Reject payloads that look like they carry raw model text instead of blob refs."""
    if isinstance(payload, dict):
        for key, value in payload.items():
            sub = f"{path}.{key}"
            if str(key).endswith("_ref") and value is not None and not is_sha256(value):
                raise RawTextInEventError(f"{sub} must be a sha256 blob reference")
            if str(key).endswith("_refs") and value is not None:
                if not isinstance(value, list) or not all(is_sha256(v) for v in value):
                    raise RawTextInEventError(f"{sub} must be a list of sha256 blob references")
            assert_no_raw_text(value, sub)
    elif isinstance(payload, list | tuple):
        for i, value in enumerate(payload):
            assert_no_raw_text(value, f"{path}[{i}]")
    elif isinstance(payload, str) and len(payload) > MAX_PAYLOAD_STRING:
        raise RawTextInEventError(
            f"{path} is a {len(payload)}-char string; store text in the BlobStore and log its hash"
        )


class EventSink:
    """Writer for a single episode chain. Created exclusively; never reopened for writing."""

    def __init__(
        self, path: Path, run_id: str, episode_id: str, clock: Clock, *, durable: bool = True
    ) -> None:
        self.path = Path(path)
        self.run_id = run_id
        self.episode_id = episode_id
        self._clock = clock
        self._durable = durable
        self.path.parent.mkdir(parents=True, exist_ok=True)
        # "xb": fail if the chain already exists. Resume discards unsealed chains instead.
        self._fh = open(self.path, "xb")
        self._seq = 0
        self._head = GENESIS_HASH
        self._sealed = False

    @property
    def head(self) -> str:
        return self._head

    @property
    def seq(self) -> int:
        return self._seq

    @property
    def sealed(self) -> bool:
        return self._sealed

    def append(self, event_type: EventType, payload: dict[str, Any]) -> Event:
        if self._sealed:
            raise ChainSealedError(f"episode {self.episode_id} is sealed")
        if event_type == EventType.EPISODE_SEAL:
            raise ValueError("use seal() to close a chain")
        return self._write(event_type, payload)

    def seal(self) -> str:
        if self._sealed:
            raise ChainSealedError(f"episode {self.episode_id} is already sealed")
        self._write(EventType.EPISODE_SEAL, {"sealed_head": self._head, "n_events": self._seq})
        self._sealed = True
        self.close()
        return self._head

    def close(self) -> None:
        if not self._fh.closed:
            self._fh.close()

    def _write(self, event_type: EventType, payload: dict[str, Any]) -> Event:
        assert_no_raw_text(payload)
        draft = {
            "schema_version": Event.model_fields["schema_version"].default,
            "run_id": self.run_id,
            "episode_id": self.episode_id,
            "seq": self._seq,
            "event_id": f"{self.episode_id}/{self._seq:06d}",
            "ts_utc": iso_utc(self._clock.now_utc()),
            "mono_ns": self._clock.monotonic_ns(),
            "event_type": event_type,
            "payload": payload,
            "prev_hash": self._head,
            "hash": "",
        }
        unhashed = Event.model_validate(draft)
        event = unhashed.model_copy(update={"hash": unhashed.compute_hash()})
        line = canonical_json(event) + "\n"
        self._fh.write(line.encode("utf-8"))
        self._fh.flush()
        if self._durable:
            os.fsync(self._fh.fileno())
        self._seq += 1
        self._head = event.hash
        return event


def read_events(path: Path) -> list[Event]:
    """Parse a chain without verifying it. Use verify_chain for integrity."""
    return [Event.model_validate_json(line) for line in _lines(Path(path))]


def verify_chain(
    path: Path, *, run_id: str | None = None, episode_id: str | None = None
) -> ChainInfo:
    """Verify every link of an episode chain. Raises IntegrityError at the first defect."""
    path = Path(path)
    head = GENESIS_HASH
    sealed = False
    first: Event | None = None
    n = 0
    for i, line in enumerate(_lines(path)):
        try:
            event = Event.model_validate_json(line)
        except ValueError as exc:
            raise IntegrityError(f"{path.name}: line {i} is not a valid event: {exc}") from exc
        first = first or event
        if sealed:
            raise IntegrityError(f"{path.name}: event after seal at seq {event.seq}")
        if event.seq != i:
            raise IntegrityError(f"{path.name}: expected seq {i}, found {event.seq}")
        if event.prev_hash != head:
            raise IntegrityError(f"{path.name}: broken link at seq {i}")
        if event.compute_hash() != event.hash:
            raise IntegrityError(f"{path.name}: hash mismatch at seq {i}")
        if (event.run_id, event.episode_id) != (first.run_id, first.episode_id):
            raise IntegrityError(f"{path.name}: run/episode id changed at seq {i}")
        assert_no_raw_text(event.payload)
        if event.event_type == EventType.EPISODE_SEAL:
            if event.payload.get("sealed_head") != head:
                raise IntegrityError(f"{path.name}: seal does not match chain head")
            sealed = True
        head = event.hash
        n += 1
    if first is None:
        raise IntegrityError(f"{path.name}: empty chain")
    if run_id is not None and first.run_id != run_id:
        raise IntegrityError(f"{path.name}: run_id {first.run_id} != expected {run_id}")
    if episode_id is not None and first.episode_id != episode_id:
        raise IntegrityError(f"{path.name}: episode_id {first.episode_id} != expected {episode_id}")
    return ChainInfo(
        run_id=first.run_id, episode_id=first.episode_id, head_hash=head, n_events=n, sealed=sealed
    )


def _lines(path: Path) -> list[str]:
    raw = path.read_bytes()
    if raw and not raw.endswith(b"\n"):
        raise IntegrityError(f"{path.name}: truncated final line (incomplete write)")
    return [ln for ln in raw.decode("utf-8").split("\n") if ln]
