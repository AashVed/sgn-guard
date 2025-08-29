from __future__ import annotations

from typing import Iterable, Optional, Tuple

import torch


def flatten_grads(params: Iterable[torch.nn.Parameter]) -> Optional[torch.Tensor]:
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


def cosine_similarity(a: torch.Tensor, b: torch.Tensor, eps: float = 1e-12) -> float:
    an = a.norm() + eps
    bn = b.norm() + eps
    return float((a @ b) / (an * bn))


def grad_norm(t: torch.Tensor) -> float:
    return float(t.norm())


def simple_alpha_beta_proxy(prev_g: torch.Tensor, cur_g: torch.Tensor) -> Tuple[float, float]:
    """Return crude proxies for contraction margin (alpha) and Lipschitz (beta).

    This is NOT a certificate—just a heuristic signal. Non-novel.
    """
    cos = cosine_similarity(prev_g, cur_g)
    beta = float((cur_g - prev_g).norm()) / (float(prev_g.norm()) + 1e-12)
    alpha = max(0.0, cos) * float(cur_g.norm())
    return alpha, beta
