from __future__ import annotations

import csv
import json
from dataclasses import dataclass, asdict
from typing import List, Optional


@dataclass
class MetricsRow:
    step: int
    lr: float
    grad_cos: Optional[float]
    grad_norm: Optional[float]
    staleness: int
    action: str  # "reduce" | "increase" | "none"
    # Optional extras for stability/convergence visualization
    state_norm: Optional[float] = None
    loss_abs: Optional[float] = None


class MetricsLogger:
    def __init__(self) -> None:
        self.rows: List[MetricsRow] = []

    def log(self, *, step: int, lr: float, grad_cos: Optional[float], grad_norm: Optional[float], staleness: int, action: str, state_norm: Optional[float] = None, loss_abs: Optional[float] = None) -> None:
        self.rows.append(MetricsRow(
            step=step,
            lr=float(lr),
            grad_cos=grad_cos,
            grad_norm=grad_norm,
            staleness=int(staleness),
            action=str(action),
            state_norm=state_norm,
            loss_abs=loss_abs,
        ))

    def to_csv(self, path: str) -> None:
        if not self.rows:
            return
        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(asdict(self.rows[0]).keys()))
            writer.writeheader()
            for r in self.rows:
                writer.writerow(asdict(r))

    def summary(self) -> dict:
        if not self.rows:
            return {}
        n = len(self.rows)
        lrs = [r.lr for r in self.rows]
        cos_vals = [r.grad_cos for r in self.rows if r.grad_cos is not None]
        norms = [r.grad_norm for r in self.rows if r.grad_norm is not None]
        st = [r.staleness for r in self.rows]
        actions = [r.action for r in self.rows]
        state_norms = [r.state_norm for r in self.rows if r.state_norm is not None]
        loss_abs_vals = [r.loss_abs for r in self.rows if r.loss_abs is not None]
        neg_cos = sum(1 for c in cos_vals if c is not None and c < 0.0)
        reduces = sum(1 for a in actions if a == "reduce")
        increases = sum(1 for a in actions if a == "increase")
        none_ct = sum(1 for a in actions if a == "none")
        return {
            "num_steps": n,
            "lr_avg": sum(lrs) / n,
            "lr_min": min(lrs),
            "lr_max": max(lrs),
            "grad_cos_avg": (sum(cos_vals) / len(cos_vals)) if cos_vals else None,
            "grad_cos_min": (min(cos_vals) if cos_vals else None),
            "grad_cos_max": (max(cos_vals) if cos_vals else None),
            "grad_norm_avg": (sum(norms) / len(norms)) if norms else None,
            "staleness_avg": sum(st) / n,
            "reductions": reduces,
            "increases": increases,
            "none_actions": none_ct,
            "reductions_rate": reduces / n,
            "neg_cos_rate": (neg_cos / len(cos_vals)) if cos_vals else None,
            "state_norm_avg": (sum(state_norms) / len(state_norms)) if state_norms else None,
            "loss_abs_avg": (sum(loss_abs_vals) / len(loss_abs_vals)) if loss_abs_vals else None,
        }

    def to_json(self, path: str) -> None:
        with open(path, "w") as f:
            json.dump(self.summary(), f, indent=2)
