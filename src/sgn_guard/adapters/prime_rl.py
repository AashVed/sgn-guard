"""Prime‑RL adapter (placeholder).

Expose a minimal interface to feed staleness and metrics into SGNGuard.
No external dependencies; users can plug this into their runners.
"""
from __future__ import annotations

from typing import Optional

from ..guard import SGNGuard


class PrimeRLAdapter:
    def __init__(self, guard: SGNGuard):
        self.guard = guard

    def on_batch_begin(self, steps_behind: Optional[int] = None) -> None:
        if steps_behind is not None:
            self.guard.observe_staleness(int(steps_behind))

    def on_before_optimizer_step(self) -> None:
        self.guard.adjust_before_step()
