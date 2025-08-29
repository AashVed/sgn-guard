"""sgn_guard package.

Stability-first learning-rate guard for decentralized training.
This package intentionally uses standard, non-novel heuristics.
"""
from .guard import SGNGuard, SGNGuardConfig

__all__ = ["SGNGuard", "SGNGuardConfig"]
