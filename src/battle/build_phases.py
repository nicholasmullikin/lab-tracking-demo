"""Phase timing for the inference-free builders.

A rebuild spends its time in a few distinct places: validating inputs, walking the
masks for geometry, and logging the recording. Printing that split makes a regression
visible in the next run instead of the next profiling session.
"""

from __future__ import annotations

import time
from collections.abc import Iterator
from contextlib import contextmanager


class PhaseTimer:
    """Record wall time per named phase and print the split when the build ends."""

    def __init__(self, label: str, *, enabled: bool = True) -> None:
        self.label = label
        self.enabled = enabled
        self.elapsed: dict[str, float] = {}
        self._open: dict[str, float] = {}
        self._started = time.perf_counter()

    @contextmanager
    def phase(self, name: str) -> Iterator[None]:
        self.start(name)
        try:
            yield
        finally:
            self.stop(name)

    def start(self, name: str) -> None:
        """Open a phase that ends somewhere else in a long, linear build."""
        self._open[name] = time.perf_counter()

    def stop(self, name: str) -> None:
        started = self._open.pop(name, None)
        if started is None:
            return
        self.elapsed[name] = self.elapsed.get(name, 0.0) + (time.perf_counter() - started)

    def report(self) -> str:
        total = time.perf_counter() - self._started
        measured = " ".join(f"{name} {seconds:.1f}s" for name, seconds in self.elapsed.items())
        other = total - sum(self.elapsed.values())
        return f"{self.label}: total {total:.1f}s ({measured} other {other:.1f}s)"

    def print_report(self) -> None:
        if self.enabled:
            print(self.report())
