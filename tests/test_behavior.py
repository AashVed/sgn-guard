from __future__ import annotations

import torch

from sgn_guard.guard import SGNGuard, SGNGuardConfig


def test_lr_reduces_on_staleness():
    p = torch.nn.Parameter(torch.tensor([1.0]))
    opt = torch.optim.SGD([p], lr=1e-2)
    guard = SGNGuard(opt, SGNGuardConfig(lr_min=1e-5, lr_max=1e-1, reduce_factor=0.5, staleness_tolerance=0))

    # Provide a benign gradient
    p.grad = torch.tensor([1.0])
    guard.observe_staleness(1)  # exceed tolerance
    guard.adjust_before_step()
    assert opt.param_groups[0]["lr"] < 1e-2, "LR should reduce when staleness exceeds tolerance"


def test_lr_increases_on_positive_cosine():
    # Two steps with aligned gradients should increase LR (subject to caps)
    p = torch.nn.Parameter(torch.tensor([1.0], dtype=torch.float32))
    opt = torch.optim.SGD([p], lr=1e-3)
    guard = SGNGuard(
        opt,
        SGNGuardConfig(
            lr_min=1e-6,
            lr_max=1e-1,
            increase_factor=1.2,
            cos_increase_threshold=0.5,
            staleness_tolerance=0,
        ),
    )

    # Step 1: set gradient
    p.grad = torch.tensor([0.5])
    guard.observe_staleness(0)
    guard.adjust_before_step()
    lr1 = float(opt.param_groups[0]["lr"])  # may remain similar on first step

    # Step 2: same direction -> positive cosine
    p.grad = torch.tensor([0.5])
    guard.observe_staleness(0)
    guard.adjust_before_step()
    lr2 = float(opt.param_groups[0]["lr"])  # should increase

    assert lr2 >= lr1, "LR should not decrease under positive cosine without staleness"
    assert lr2 > 1e-3, "LR should increase above the initial value under alignment"
