"""Toy two‑player bilinear game with SGN‑Guard.

Min player x, Max player y on L(x, y) = x^T A y.
This example demonstrates LR adaptation; it's intentionally simple.
"""
from __future__ import annotations

import torch

from sgn_guard.guard import SGNGuard, SGNGuardConfig
from sgn_guard.metrics import MetricsLogger
from sgn_guard.estimators import flatten_grads, cosine_similarity, grad_norm


def main(seed: int = 0) -> None:
    torch.manual_seed(seed)
    device = torch.device("cpu")

    # Parameters for both players
    x = torch.nn.Parameter(torch.randn(2, device=device))
    y = torch.nn.Parameter(torch.randn(2, device=device))

    A = torch.tensor([[1.0, 0.0], [0.0, -1.0]], device=device)

    # Single optimizer updating both params (we manually flip sign for y)
    opt = torch.optim.SGD([x, y], lr=1e-2)
    guard = SGNGuard(opt, SGNGuardConfig(lr_min=1e-5, lr_max=5e-2))

    metrics = MetricsLogger()
    g_prev = None

    def loss_fn(xv: torch.Tensor, yv: torch.Tensor) -> torch.Tensor:
        return torch.dot(xv, A @ yv)

    for step in range(200):
        opt.zero_grad()
        L = loss_fn(x, y)
        # Compute grads: x descends on L, y ascends on L -> grad(y) on -L
        gx, gy = torch.autograd.grad(L, (x, y), retain_graph=True)
        y_grad_ascent = torch.autograd.grad(-L, y)[0]

        # Assign gradients
        x.grad = gx
        y.grad = y_grad_ascent

        # Optionally: report staleness (0 here)
        guard.observe_staleness(0)

        # Prepare metrics signals before LR change
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

        # Adjust LR based on signals and record action
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

        if step % 20 == 0:
            lr = opt.param_groups[0]["lr"]
            print(f"step={step:03d} L={float(L):+.4f} lr={lr:.5f} action={action} grad_cos={cos if cos is not None else 'NA'}")

        # Log metrics row and update prev grad
        metrics.log(step=step, lr=lr_after, grad_cos=cos, grad_norm=gnorm, staleness=0, action=action)
        g_prev = g_cur.clone() if g_cur is not None else None

    # Persist metrics
    metrics.to_csv("examples/metrics_simple_game.csv")
    metrics.to_json("examples/metrics_simple_game.json")


if __name__ == "__main__":
    main()
