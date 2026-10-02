"""Injectable clocks. Wall time is for timestamps; monotonic time is for latency and budgets."""

import time
from datetime import UTC, datetime, timedelta
from typing import Protocol


class Clock(Protocol):
    def now_utc(self) -> datetime: ...
    def monotonic_ns(self) -> int: ...


class SystemClock:
    def now_utc(self) -> datetime:
        return datetime.now(UTC)

    def monotonic_ns(self) -> int:
        return time.monotonic_ns()


class FakeClock:
    """Deterministic clock for tests and mock runs. Time only moves when advanced."""

    def __init__(self, start: datetime | None = None, start_ns: int = 0) -> None:
        self._wall = start or datetime(2026, 1, 1, tzinfo=UTC)
        self._mono = start_ns

    def now_utc(self) -> datetime:
        return self._wall

    def monotonic_ns(self) -> int:
        return self._mono

    def advance(self, seconds: float) -> None:
        if seconds < 0:
            raise ValueError("clock cannot move backwards")
        self._mono += int(seconds * 1e9)
        self._wall += timedelta(seconds=seconds)


def iso_utc(dt: datetime) -> str:
    if dt.tzinfo is None:
        raise ValueError("naive datetime; use timezone-aware UTC")
    return dt.astimezone(UTC).isoformat().replace("+00:00", "Z")
