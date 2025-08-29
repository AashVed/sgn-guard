from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Iterable

import torch

def _flatten_grads(params: Iterable[torch.nn.Parameter]) -> Optional[torch.Tensor]:
    vecs = []
    for p in params:
        if p.grad is None:
            continue
        g = p.grad.detach().reshape(-1)
        if g.numel() > 0:
            vecs.append(g)
    if not vecs:
        return None
    return torch.cat(vecs)


def _cosine(a: torch.Tensor, b: torch.Tensor, eps: float = 1e-12) -> float:
    an = a.norm() + eps
    bn = b.norm() + eps
    return float((a @ b) / (an * bn))


@dataclass
class SGNGuardConfig:
    """Configuration for SGNGuard.

    This config encodes conservative, non-novel LR policies inspired by
    contraction-style safety margins.
    """
    lr_min: float = 1e-6
    lr_max: float = 5e-2
    reduce_factor: float = 0.5  # multiplicative LR decrease on instability
    increase_factor: float = 1.05  # slow LR increase on alignment
    cos_decrease_threshold: float = 0.0  # decrease if grad cosine < 0
    cos_increase_threshold: float = 0.7  # consider increase if cosine > 0.7
    grad_growth_threshold: float = 2.0  # decrease if ||g|| grows > x over window
    staleness_tolerance: int = 0  # decrease if steps_behind > tolerance
    smoothing: float = 0.9  # EMA smoothing for grad norm stats


class SGNGuard:
    """LR guard that adapts optimizer LR before each step.

    Usage:
        guard = SGNGuard(optimizer, SGNGuardConfig())
        ... compute grads ...
        guard.adjust_before_step()
        optimizer.step()

    Notes:
        - Heuristic signals: gradient cosine, gradient norm growth, staleness.
        - Non-novel, conservative by design.
    """

    def __init__(self, optimizer: torch.optim.Optimizer, config: SGNGuardConfig):
        self.opt = optimizer
        self.cfg = config
        self._prev_grad: Optional[torch.Tensor] = None
        self._ema_gnorm: Optional[float] = None
        self._steps_behind: int = 0

    def observe_staleness(self, steps_behind: int) -> None:
        self._steps_behind = int(max(0, steps_behind))

    @torch.no_grad()
    def adjust_before_step(self) -> None:
        # Gather current grads
        params = [p for g in self.opt.param_groups for p in g.get("params", [])]
        g_cur = _flatten_grads(params)
        if g_cur is None:
            return

        # Update simple gradient norm EMA
        gnorm = float(g_cur.norm())
        if self._ema_gnorm is None:
            self._ema_gnorm = gnorm
        else:
            alpha = self.cfg.smoothing
            self._ema_gnorm = alpha * self._ema_gnorm + (1 - alpha) * gnorm

        # Compute signals
        cos = None
        if self._prev_grad is not None and self._prev_grad.numel() == g_cur.numel():
            try:
                cos = _cosine(self._prev_grad, g_cur)
            except Exception:
                cos = None

        # Decide action per param group (same policy for all groups here)
        for group in self.opt.param_groups:
            lr = float(group.get("lr", 1e-3))
            new_lr = lr

            # Instability triggers: negative cosine, rapid grad growth, staleness
            instability = False
            if cos is not None and cos < self.cfg.cos_decrease_threshold:
                instability = True
            if self._ema_gnorm is not None and gnorm > self.cfg.grad_growth_threshold * max(1e-12, self._ema_gnorm):
                # If current norm spikes above EMA by a factor, treat as instability
                instability = True
            if self._steps_behind > self.cfg.staleness_tolerance:
                instability = True

            if instability:
                new_lr = max(self.cfg.lr_min, lr * self.cfg.reduce_factor)
            else:
                if cos is not None and cos > self.cfg.cos_increase_threshold and self._steps_behind <= self.cfg.staleness_tolerance:
                    new_lr = min(self.cfg.lr_max, lr * self.cfg.increase_factor)

            group["lr"] = float(new_lr)

        # Save for next step
        self._prev_grad = g_cur.clone()
