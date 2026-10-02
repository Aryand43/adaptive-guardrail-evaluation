import pytest

from models.budget import (
    Amount,
    BudgetDenied,
    BudgetLedger,
    BudgetLimits,
    ReservationError,
)

LIMITS = BudgetLimits(turns=3, queries=10, tokens=1000, cost_usd=0.001, wall_s=10)


@pytest.fixture
def ledger(clock):
    return BudgetLedger(LIMITS, clock)


def A(q=0, t=0, c=0):
    return Amount(queries=q, tokens=t, cost_nano=c)


def test_limits_convert_cost_to_nano(ledger):
    assert ledger.limit == A(10, 1000, 1_000_000)


def test_reserve_holds_and_commit_reconciles(ledger):
    r = ledger.reserve(A(1, 300, 1000), "x")
    assert ledger.outstanding == A(1, 300, 1000)
    assert ledger.free == A(9, 700, 999_000)
    rec = ledger.commit(r, A(1, 120, 400))
    assert rec.released == A(0, 180, 600) and rec.overrun.is_zero
    assert ledger.outstanding.is_zero and ledger.consumed == A(1, 120, 400)


def test_reservation_that_does_not_fit_is_denied(ledger):
    ledger.reserve(A(1, 900, 0), "big")
    with pytest.raises(BudgetDenied) as exc:
        ledger.reserve(A(1, 200, 0), "too-much")
    assert exc.value.dimensions == ["tokens"]
    assert ledger.outstanding == A(1, 900, 0)  # denied reservation changed nothing


def test_overrun_is_recorded_in_full_not_clipped(ledger):
    r = ledger.reserve(A(1, 100, 100), "x")
    rec = ledger.commit(r, A(1, 1500, 100))
    assert rec.overrun == A(0, 1400, 0)
    assert ledger.consumed.tokens == 1500 and ledger.overrun.tokens == 1400
    assert ledger.exhausted() == ["tokens"]
    assert ledger.free.tokens == 0
    with pytest.raises(BudgetDenied):
        ledger.reserve(A(0, 1, 0), "after-overrun")


def test_child_reservation_draws_from_parent_then_global(ledger):
    turn = ledger.reserve(A(4, 400, 0), "turn")
    child = ledger.reserve(A(1, 100, 0), "call", parent=turn)
    assert turn.held == A(3, 300, 0)
    assert ledger.outstanding == A(4, 400, 0)  # carving does not double-count
    big = ledger.reserve(A(1, 500, 0), "call2", parent=turn)  # 300 from parent, 200 extra
    assert turn.held == A(2, 0, 0) and ledger.outstanding == A(4, 600, 0)
    ledger.commit(child, A(1, 80, 0))
    ledger.commit(big, A(1, 450, 0))
    ledger.release(turn)
    assert ledger.outstanding.is_zero and ledger.consumed == A(2, 530, 0)


def test_release_with_open_children_and_double_close_fail(ledger):
    turn = ledger.reserve(A(2, 200, 0), "turn")
    child = ledger.reserve(A(1, 100, 0), "c", parent=turn)
    with pytest.raises(ReservationError, match="open child"):
        ledger.release(turn)
    ledger.commit(child, A(1, 10, 0))
    with pytest.raises(ReservationError, match="already closed"):
        ledger.commit(child, A(1, 10, 0))
    ledger.release(turn)
    with pytest.raises(ReservationError):
        ledger.reserve(A(0, 1, 0), "x", parent=turn)


def test_turns_and_wall_clock(ledger, clock):
    for _ in range(3):
        ledger.record_turn()
    assert ledger.remaining_turns == 0
    with pytest.raises(ReservationError):
        ledger.record_turn()
    clock.advance(4.5)
    assert ledger.elapsed_s == pytest.approx(4.5) and ledger.remaining_wall_s == pytest.approx(5.5)
    clock.advance(10)
    assert ledger.remaining_wall_s == 0


def test_snapshot(ledger):
    r = ledger.reserve(A(1, 10, 10), "x")
    ledger.commit(r, A(1, 20, 5))
    s = ledger.snapshot()
    assert s.consumed == A(1, 20, 5) and s.overrun == A(0, 10, 0)
    assert s.remaining == A(9, 980, 999_995)
