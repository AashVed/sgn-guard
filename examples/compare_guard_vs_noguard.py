"""Compare SGN-Guard vs no-guard on the toy bilinear game.

This script runs two short trainings with identical seeds and logs JSON summaries:
- examples/metrics_compare_guard.json
- examples/metrics_compare_noguard.json

Additionally, it saves per-step CSV traces for plotting:
- examples/trace_compare_guard.csv
- examples/trace_compare_noguard.csv

Run from repo root after installing: `pip install -e ".[dev]"`
"""
from __future__ import annotations

import json
from typing import Optional

import torch

from sgn_guard.guard import SGNGuard, SGNGuardConfig
from sgn_guard.metrics import MetricsLogger
from sgn_guard.estimators import flatten_grads, cosine_similarity, grad_norm


def run_with_guard(steps: int = 200, seed: int = 0, save_csv_path: Optional[str] = None) -> dict:
    torch.manual_seed(seed)
    device = torch.device("cpu")
    x = torch.nn.Parameter(torch.randn(2, device=device))
    y = torch.nn.Parameter(torch.randn(2, device=device))
    # Harder regime: introduce off-diagonal coupling to induce rotational dynamics
    A = torch.tensor([[1.0, 0.8], [-0.8, -1.0]], device=device)

    # Stress baseline with a higher LR; guard will rein in when needed
    opt = torch.optim.SGD([x, y], lr=1e-1)
    guard = SGNGuard(
        opt,
        SGNGuardConfig(
            lr_min=1e-5,
            lr_max=1e-1,
            reduce_factor=0.5,
            increase_factor=1.01,
            cos_decrease_threshold=0.99,
            cos_increase_threshold=0.997,
            grad_growth_threshold=1.1,
            staleness_tolerance=0,
            smoothing=0.8,
        ),
    )
    metrics = MetricsLogger()

    g_prev: Optional[torch.Tensor] = None

    def loss_fn(xv: torch.Tensor, yv: torch.Tensor) -> torch.Tensor:
        return torch.dot(xv, A @ yv)

    for step in range(steps):
        opt.zero_grad()
        L = loss_fn(x, y)
        gx, gy = torch.autograd.grad(L, (x, y), retain_graph=True)
        y_grad_ascent = torch.autograd.grad(-L, y)[0]
        x.grad = gx
        y.grad = y_grad_ascent

        # No synthetic staleness in this regime for fairness
        staleness = 0
        guard.observe_staleness(staleness)

        g_cur = flatten_grads([x, y])
        cos = None
        gnorm = None
        if g_cur is not None:
            if g_prev is not None and g_prev.numel() == g_cur.numel():
                try:
                    cos = cosine_similarity(g_prev, g_cur)
                except Exception:
                    cos = None
            gnorm = grad_norm(g_cur)

        lr_before = float(opt.param_groups[0]["lr"])
        guard.adjust_before_step()
        lr_after = float(opt.param_groups[0]["lr"])  
        if lr_after < lr_before:
            action = "reduce"
        elif lr_after > lr_before:
            action = "increase"
        else:
            action = "none"

        opt.step()

        # Compute stability/convergence indicators AFTER the step
        with torch.no_grad():
            state_norm = float(torch.sqrt(x.detach().pow(2).sum() + y.detach().pow(2).sum()))
            loss_after_abs = float(torch.abs(loss_fn(x, y)))

        metrics.log(
            step=step,
            lr=lr_after,
            grad_cos=cos,
            grad_norm=gnorm,
            staleness=staleness,
            action=action,
            state_norm=state_norm,
            loss_abs=loss_after_abs,
        )
        g_prev = g_cur.clone() if g_cur is not None else None

    if save_csv_path:
        metrics.to_csv(save_csv_path)
    return metrics.summary()


def run_no_guard(steps: int = 200, seed: int = 0, save_csv_path: Optional[str] = None) -> dict:
    torch.manual_seed(seed)
    device = torch.device("cpu")
    x = torch.nn.Parameter(torch.randn(2, device=device))
    y = torch.nn.Parameter(torch.randn(2, device=device))
    A = torch.tensor([[1.0, 0.8], [-0.8, -1.0]], device=device)

    opt = torch.optim.SGD([x, y], lr=1e-1)
    metrics = MetricsLogger()

    g_prev: Optional[torch.Tensor] = None

    def loss_fn(xv: torch.Tensor, yv: torch.Tensor) -> torch.Tensor:
        return torch.dot(xv, A @ yv)

    for step in range(steps):
        opt.zero_grad()
        L = loss_fn(x, y)
        gx, gy = torch.autograd.grad(L, (x, y), retain_graph=True)
        y_grad_ascent = torch.autograd.grad(-L, y)[0]
        x.grad = gx
        y.grad = y_grad_ascent

        g_cur = flatten_grads([x, y])
        cos = None
        gnorm = None
        if g_cur is not None:
            if g_prev is not None and g_prev.numel() == g_cur.numel():
                try:
                    cos = cosine_similarity(g_prev, g_cur)
                except Exception:
                    cos = None
            gnorm = grad_norm(g_cur)

        # No LR adaptation; record constant LR and action 'none'
        lr_after = float(opt.param_groups[0]["lr"])
        opt.step()

        with torch.no_grad():
            state_norm = float(torch.sqrt(x.detach().pow(2).sum() + y.detach().pow(2).sum()))
            loss_after_abs = float(torch.abs(loss_fn(x, y)))

        metrics.log(
            step=step,
            lr=lr_after,
            grad_cos=cos,
            grad_norm=gnorm,
            staleness=0,
            action="none",
            state_norm=state_norm,
            loss_abs=loss_after_abs,
        )
        g_prev = g_cur.clone() if g_cur is not None else None

    if save_csv_path:
        metrics.to_csv(save_csv_path)
    return metrics.summary()


def main() -> None:
    guard_summary = run_with_guard(save_csv_path="examples/trace_compare_guard.csv")
    no_guard_summary = run_no_guard(save_csv_path="examples/trace_compare_noguard.csv")

    with open("examples/metrics_compare_guard.json", "w") as f:
        json.dump(guard_summary, f, indent=2)
    with open("examples/metrics_compare_noguard.json", "w") as f:
        json.dump(no_guard_summary, f, indent=2)

    print("Guard summary:", json.dumps(guard_summary, indent=2))
    print("No-guard summary:", json.dumps(no_guard_summary, indent=2))


if __name__ == "__main__":
    main()
