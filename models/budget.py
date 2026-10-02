"""Budget limits and the per-episode BudgetLedger.

Accounting dimensions that can be reserved: queries, tokens, cost (integer nano-USD).
Turns are counted by the orchestrator; wall time is measured on the monotonic clock.

Protocol: reserve -> call -> commit(actual). A reservation may be carved from a parent
reservation (the orchestrator reserves a whole turn, each call draws from it). Actual
usage is always recorded in full, even when it exceeds what was reserved: overruns are
measured and reported, never clipped.
"""

from dataclasses import dataclass, field
from itertools import count

from pydantic import Field

from models.pricing import usd_to_nano
from storage.clock import Clock
from storage.versioning import Frozen

DIMENSIONS = ("queries", "tokens", "cost_nano")


class BudgetLimits(Frozen):
    """Per-episode hard limits. Queries, tokens and cost include every role."""

    turns: int = Field(gt=0)
    queries: int = Field(gt=0)
    tokens: int = Field(gt=0)
    cost_usd: float = Field(gt=0)
    wall_s: float = Field(gt=0)


class Amount(Frozen):
    queries: int = Field(default=0, ge=0)
    tokens: int = Field(default=0, ge=0)
    cost_nano: int = Field(default=0, ge=0)

    def __add__(self, other: "Amount") -> "Amount":
        return Amount(**{d: getattr(self, d) + getattr(other, d) for d in DIMENSIONS})

    def minus_floor(self, other: "Amount") -> "Amount":
        """Component-wise subtraction floored at zero."""
        return Amount(**{d: max(0, getattr(self, d) - getattr(other, d)) for d in DIMENSIONS})

    def exceeds(self, other: "Amount") -> list[str]:
        return [d for d in DIMENSIONS if getattr(self, d) > getattr(other, d)]

    @property
    def is_zero(self) -> bool:
        return not any(getattr(self, d) for d in DIMENSIONS)


ZERO = Amount()


class BudgetDenied(Exception):
    def __init__(self, label: str, dimensions: list[str]) -> None:
        self.label = label
        self.dimensions = dimensions
        super().__init__(f"budget denied for {label}: {', '.join(dimensions)}")


class ReservationError(Exception):
    pass


@dataclass(eq=False)
class Reservation:
    """A live hold on budget. `held` shrinks as child reservations are carved from it."""

    id: int
    label: str
    held: Amount
    parent: "Reservation | None" = None
    open: bool = True
    children: list["Reservation"] = field(default_factory=list)


class Reconciliation(Frozen):
    reservation_id: int
    reserved: Amount
    actual: Amount
    overrun: Amount
    released: Amount


class BudgetSnapshot(Frozen):
    turns_used: int
    consumed: Amount
    outstanding: Amount
    overrun: Amount
    elapsed_s: float
    remaining: Amount
    remaining_turns: int
    remaining_wall_s: float


class BudgetLedger:
    def __init__(self, limits: BudgetLimits, clock: Clock) -> None:
        self.limits = limits
        self.limit = Amount(
            queries=limits.queries, tokens=limits.tokens, cost_nano=usd_to_nano(limits.cost_usd)
        )
        self._clock = clock
        self._start_ns = clock.monotonic_ns()
        self._ids = count(1)
        self.consumed = ZERO
        self.outstanding = ZERO
        self.overrun = ZERO
        self.turns_used = 0

    # -- time and turns ---------------------------------------------------------------

    @property
    def elapsed_s(self) -> float:
        return (self._clock.monotonic_ns() - self._start_ns) / 1e9

    @property
    def remaining_wall_s(self) -> float:
        return max(0.0, self.limits.wall_s - self.elapsed_s)

    @property
    def remaining_turns(self) -> int:
        return max(0, self.limits.turns - self.turns_used)

    def record_turn(self) -> None:
        if self.remaining_turns == 0:
            raise ReservationError("turn limit already reached")
        self.turns_used += 1

    # -- reservable dimensions -------------------------------------------------------

    @property
    def free(self) -> Amount:
        return self.limit.minus_floor(self.consumed + self.outstanding)

    def shortfall(self, amount: Amount) -> list[str]:
        """Dimensions in which `amount` does not fit the free budget. Empty means it fits."""
        return amount.exceeds(self.free)

    def exhausted(self) -> list[str]:
        """Dimensions fully consumed (or overrun). Wall time and turns are checked separately."""
        return [d for d in DIMENSIONS if getattr(self.consumed, d) >= getattr(self.limit, d)]

    def reserve(self, amount: Amount, label: str, parent: Reservation | None = None) -> Reservation:
        if parent is not None:
            if not parent.open:
                raise ReservationError(f"parent reservation {parent.label} is closed")
            from_parent = Amount(
                **{d: min(getattr(amount, d), getattr(parent.held, d)) for d in DIMENSIONS}
            )
            extra = amount.minus_floor(from_parent)
        else:
            from_parent, extra = ZERO, amount
        short = self.shortfall(extra)
        if short:
            raise BudgetDenied(label, short)
        if parent is not None:
            parent.held = parent.held.minus_floor(from_parent)
        self.outstanding = self.outstanding + extra
        res = Reservation(id=next(self._ids), label=label, held=amount, parent=parent)
        if parent is not None:
            parent.children.append(res)
        return res

    def commit(self, res: Reservation, actual: Amount) -> Reconciliation:
        """Record actual usage against a reservation and release the remainder."""
        self._close(res)
        overrun = actual.minus_floor(res.held)
        released = res.held.minus_floor(actual)
        self.outstanding = self.outstanding.minus_floor(res.held)
        self.consumed = self.consumed + actual
        self.overrun = self.overrun + overrun
        return Reconciliation(
            reservation_id=res.id, reserved=res.held, actual=actual, overrun=overrun, released=released
        )

    def release(self, res: Reservation) -> Amount:
        """Return an unused (or partly carved) reservation to the free pool."""
        if any(c.open for c in res.children):
            raise ReservationError(f"{res.label} still has open child reservations")
        self._close(res)
        self.outstanding = self.outstanding.minus_floor(res.held)
        return res.held

    @staticmethod
    def _close(res: Reservation) -> None:
        if not res.open:
            raise ReservationError(f"reservation {res.label} already closed")
        res.open = False

    def snapshot(self) -> BudgetSnapshot:
        return BudgetSnapshot(
            turns_used=self.turns_used,
            consumed=self.consumed,
            outstanding=self.outstanding,
            overrun=self.overrun,
            elapsed_s=self.elapsed_s,
            remaining=self.limit.minus_floor(self.consumed),
            remaining_turns=self.remaining_turns,
            remaining_wall_s=self.remaining_wall_s,
        )
