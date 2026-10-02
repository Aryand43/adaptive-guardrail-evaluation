import json
import shutil

import pytest

from metrics.aggregate import aggregate, distribution, mcnemar_exact, wilson
from metrics.episode import EpisodeRecord, compute_episode_record
from storage.errors import IntegrityError
from storage.events import EventSink, EventType
from tests.conftest import ROOT
from tests.fixtures.golden.regenerate import WEIGHTS

GOLDEN = ROOT / "tests/fixtures/golden"


@pytest.fixture
def golden():
    return compute_episode_record(GOLDEN / "episode.jsonl", WEIGHTS)


def test_golden_log_reproduces_expected_record_exactly(golden):
    expected = EpisodeRecord.model_validate_json((GOLDEN / "expected_record.json").read_text())
    assert golden == expected


def test_golden_log_hand_checked_accounting(golden):
    # Values verified by hand against configs/pricing/mock.yaml:
    # target 503 attempt: 19 estimated input tokens -> 1_900 nano; retry 7+7 -> 3_500 nano;
    # judge 57+6 -> 12_150 nano; turn-2 target usage missing -> 63+136 estimated.
    assert golden.success and golden.success_turn == 3 and golden.stop_reason == "success"
    assert golden.used.queries == 6 and golden.used.tokens == 589
    assert golden.used.cost_usd == pytest.approx(135_850e-9)
    t, e = golden.per_role["target"], golden.per_role["evaluator"]
    assert (t.queries, t.attempts, t.failed_attempts, t.estimated_usage_attempts) == (3, 4, 1, 2)
    assert (e.queries, e.attempts) == (3, 3)
    assert t.cost_nano + e.cost_nano == 135_850
    assert golden.usage_estimated and not golden.budget_overrun
    assert [x.harm for x in golden.turns] == [0.0, 0.57, 0.97]
    assert [x.progress for x in golden.turns] == [0.0, 0.6, 1.0]
    assert [x.refusal_type for x in golden.turns] == ["refusal", "none", "none"]
    assert golden.at_success.queries == 6 and golden.at_success.turns == 3
    assert golden.remaining.queries == 60 - 6 and golden.remaining.turns == 3
    b = golden.budget
    expected_j = 1 - 0.1 * (6 / b.queries + 589 / b.tokens + golden.used.wall_s / b.wall_s + golden.used.cost_usd / b.cost_usd)
    assert golden.objective_score == pytest.approx(expected_j)


def test_tampered_golden_log_is_rejected(tmp_path):
    p = tmp_path / "episode.jsonl"
    shutil.copy(GOLDEN / "episode.jsonl", p)
    lines = p.read_text().splitlines()
    ev = json.loads(lines[13])
    ev["payload"]["score"] = 0.99
    lines[13] = json.dumps(ev, sort_keys=True, separators=(",", ":"))
    p.write_text("\n".join(lines) + "\n")
    with pytest.raises(IntegrityError):
        compute_episode_record(p)


def test_unsealed_chain_rejected(tmp_path, clock):
    sink = EventSink(tmp_path / "e.jsonl", "r", "e", clock, durable=False)
    sink.append(EventType.EPISODE_START, {"x": 1})
    sink.close()
    with pytest.raises(IntegrityError, match="not sealed"):
        compute_episode_record(tmp_path / "e.jsonl")


def test_ledger_mismatch_rejected(tmp_path, clock):
    src = [json.loads(x) for x in (GOLDEN / "episode.jsonl").read_text().splitlines()]
    sink = EventSink(tmp_path / "e.jsonl", "r", "e", clock, durable=False)
    for ev in src[:-1]:
        payload = ev["payload"]
        if ev["event_type"] == "stop":
            payload["budget"]["consumed"]["tokens"] += 1  # ledger disagrees with model calls
        sink.append(EventType(ev["event_type"]), payload)
    sink.seal()
    with pytest.raises(IntegrityError, match="do not match the ledger"):
        compute_episode_record(tmp_path / "e.jsonl")


# -- aggregate ---------------------------------------------------------------------------


def test_wilson_and_mcnemar_reference_values():
    ci = wilson(0, 10)
    assert ci.low == 0 and ci.high == pytest.approx(0.2775, abs=1e-4)
    ci = wilson(5, 10)
    assert (ci.low, ci.high) == (pytest.approx(0.2366, abs=1e-4), pytest.approx(0.7634, abs=1e-4))
    assert mcnemar_exact(0, 6) == pytest.approx(0.03125)
    assert mcnemar_exact(3, 3) == 1.0 and mcnemar_exact(0, 0) == 1.0


def test_distribution():
    d = distribution([3, 1, 2])
    assert (d.n, d.median, d.min, d.max, d.mean) == (3, 2, 1, 3, 2)
    assert distribution([]).mean is None


def _variant(rec, **kw):
    return rec.model_copy(update=kw)


def test_aggregate_cells_curves_and_paired_comparisons(golden):
    fail = dict(success=False, success_turn=None, at_success=None, status="failure", stop_reason="turn_limit")
    recs = []
    for i in range(4):  # policy A succeeds on objectives 0-2, policy B only on 0
        for pol, ok in (("A", i < 3), ("B", i == 0)):
            kw = {} if ok else fail
            recs.append(_variant(golden, episode_id=f"{pol}-{i}", objective_id=f"o{i}", policy=pol,
                                 cell_key=f"{pol}|t", **kw))
    rep = aggregate(recs)
    assert rep.n_episodes == 8
    a, b = rep.cells
    assert (a.policy, a.n_success, a.asr) == ("A@0", 3, 0.75)
    assert (b.n_success, b.asr) == (1, 0.25)
    assert a.turns_to_success.values == [3, 3, 3]
    assert [p.model_dump() for p in a.success_vs_budget["queries"]] == [{"budget": 6, "success_rate": 0.75}]
    assert a.trajectory[0].turn == 1 and a.trajectory[0].refusal_rate == 1.0
    cmp = [c for c in rep.comparisons if c.factor == "policy"][0]
    assert (cmp.a_only, cmp.b_only, cmp.both, cmp.neither, cmp.n_pairs) == (2, 0, 1, 1, 4)
    assert cmp.asr_difference == 0.5 and cmp.mcnemar_p == 0.5
    assert a.stop_reasons == {"success": 3, "turn_limit": 1}
