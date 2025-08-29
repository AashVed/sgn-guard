from __future__ import annotations

import torch

from sgn_guard.guard import SGNGuard, SGNGuardConfig


def test_lr_reduces_on_negative_cosine():
    p = torch.nn.Parameter(torch.tensor([1.0]))
    opt = torch.optim.SGD([p], lr=1e-2)
    guard = SGNGuard(opt, SGNGuardConfig(lr_min=1e-5, lr_max=1e-1, reduce_factor=0.5))

    # First grad
    p.grad = torch.tensor([1.0])
    guard.adjust_before_step()
    lr1 = opt.param_groups[0]["lr"]

    # Second grad opposite direction -> negative cosine
    p.grad = torch.tensor([-1.0])
    guard.adjust_before_step()
    lr2 = opt.param_groups[0]["lr"]

    assert lr2 < lr1, "LR should reduce on negative cosine"
