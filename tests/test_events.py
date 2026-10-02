import json

import pytest

from storage.errors import ChainSealedError, IntegrityError, RawTextInEventError
from storage.events import EventSink, EventType, read_events, verify_chain
from storage.hashing import GENESIS_HASH, sha256_text


def _chain(tmp_path, clock, n=3, seal=True):
    path = tmp_path / "ep-1.jsonl"
    sink = EventSink(path, "run-1", "ep-1", clock, durable=False)
    sink.append(EventType.EPISODE_START, {"objective_id": "PLACEHOLDER-001"})
    for i in range(n - 1):
        clock.advance(0.5)
        sink.append(EventType.MODEL_CALL, {"role": "target", "response_ref": sha256_text(str(i))})
    head = sink.seal() if seal else sink.head
    return path, sink, head


def _rewrite(path, lines):
    path.write_text("".join(json.dumps(x, sort_keys=True, separators=(",", ":")) + "\n" for x in lines))


def _load(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def test_chain_verifies(tmp_path, clock):
    path, sink, head = _chain(tmp_path, clock)
    info = verify_chain(path, run_id="run-1", episode_id="ep-1")
    assert info.sealed and info.head_hash == head and info.n_events == 4
    events = read_events(path)
    assert events[0].prev_hash == GENESIS_HASH
    assert [e.seq for e in events] == [0, 1, 2, 3]
    assert events[-1].event_type == EventType.EPISODE_SEAL
    assert events[1].event_id == "ep-1/000001"


def test_payload_edit_detected(tmp_path, clock):
    path, *_ = _chain(tmp_path, clock)
    lines = _load(path)
    lines[1]["payload"]["role"] = "attacker"
    _rewrite(path, lines)
    with pytest.raises(IntegrityError, match="hash mismatch at seq 1"):
        verify_chain(path)


def test_rehashed_edit_breaks_next_link(tmp_path, clock):
    from storage.events import Event

    path, *_ = _chain(tmp_path, clock)
    lines = _load(path)
    lines[1]["payload"]["role"] = "attacker"
    ev = Event.model_validate(lines[1])
    lines[1]["hash"] = ev.compute_hash()  # attacker recomputes this event's hash...
    _rewrite(path, lines)
    with pytest.raises(IntegrityError, match="broken link at seq 2"):  # ...but the chain breaks
        verify_chain(path)


@pytest.mark.parametrize(
    "tamper, msg",
    [
        (lambda ls: ls.pop(1), "expected seq 1"),
        (lambda ls: ls.__setitem__(slice(1, 3), [ls[2], ls[1]]), "expected seq 1"),
        (lambda ls: ls.pop(), None),  # dropping the seal makes the chain unsealed
    ],
)
def test_deletion_reorder_truncation(tmp_path, clock, tamper, msg):
    path, *_ = _chain(tmp_path, clock)
    lines = _load(path)
    tamper(lines)
    _rewrite(path, lines)
    if msg:
        with pytest.raises(IntegrityError, match=msg):
            verify_chain(path)
    else:
        assert verify_chain(path).sealed is False


def test_partial_write_detected(tmp_path, clock):
    path, *_ = _chain(tmp_path, clock, seal=False)
    with open(path, "ab") as fh:
        fh.write(b'{"seq": 3, "trunc')
    with pytest.raises(IntegrityError, match="truncated"):
        verify_chain(path)


def test_append_after_seal_rejected(tmp_path, clock):
    path, sink, _ = _chain(tmp_path, clock)
    with pytest.raises(ChainSealedError):
        sink.append(EventType.ERROR, {})
    with pytest.raises(ChainSealedError):
        sink.seal()


def test_event_after_seal_detected(tmp_path, clock):
    path, *_ = _chain(tmp_path, clock)
    lines = _load(path)
    extra = dict(lines[-1], seq=len(lines), prev_hash=lines[-1]["hash"])
    _rewrite(path, [*lines, extra])
    with pytest.raises(IntegrityError, match="after seal"):
        verify_chain(path)


def test_existing_chain_cannot_be_reopened(tmp_path, clock):
    path, *_ = _chain(tmp_path, clock)
    with pytest.raises(FileExistsError):
        EventSink(path, "run-1", "ep-1", clock)


def test_wrong_ids_detected(tmp_path, clock):
    path, *_ = _chain(tmp_path, clock)
    with pytest.raises(IntegrityError, match="run_id"):
        verify_chain(path, run_id="run-2")


@pytest.mark.parametrize(
    "payload",
    [
        {"note": "x" * 600},
        {"response_ref": "this is raw text, not a hash"},
        {"nested": [{"message_ref": "raw"}]},
        {"message_refs": ["raw"]},
    ],
)
def test_raw_text_guard(tmp_path, clock, payload):
    sink = EventSink(tmp_path / "e.jsonl", "r", "e", clock, durable=False)
    with pytest.raises(RawTextInEventError):
        sink.append(EventType.TARGET_RESPONSE, payload)
    assert sink.seq == 0  # nothing written


def test_seal_event_type_reserved(tmp_path, clock):
    sink = EventSink(tmp_path / "e.jsonl", "r", "e", clock, durable=False)
    with pytest.raises(ValueError):
        sink.append(EventType.EPISODE_SEAL, {})


def test_identical_inputs_give_identical_chains(tmp_path):
    from storage.clock import FakeClock

    a, *_ = _chain(tmp_path / "a", FakeClock())
    b, *_ = _chain(tmp_path / "b", FakeClock())
    assert a.read_bytes() == b.read_bytes()
