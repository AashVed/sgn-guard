from __future__ import annotations

import csv
from pathlib import Path
from typing import Tuple

import numpy as np
import matplotlib.pyplot as plt


def _parse_float(val: str) -> float:
    if val is None or val == "" or val == "None":
        return float("nan")
    try:
        return float(val)
    except Exception:
        return float("nan")


def load_trace(path: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    steps, lrs, cos, norms, state_norms, loss_abs = [], [], [], [], [], []
    with open(path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            steps.append(int(row["step"]))
            lrs.append(_parse_float(row.get("lr")))
            cos.append(_parse_float(row.get("grad_cos")))
            norms.append(_parse_float(row.get("grad_norm")))
            state_norms.append(_parse_float(row.get("state_norm")))
            loss_abs.append(_parse_float(row.get("loss_abs")))
    return (
        np.array(steps),
        np.array(lrs),
        np.array(cos),
        np.array(norms),
        np.array(state_norms),
        np.array(loss_abs),
    )


def main() -> None:
    guard_csv = Path("examples/trace_compare_guard.csv")
    noguard_csv = Path("examples/trace_compare_noguard.csv")

    if not guard_csv.exists() or not noguard_csv.exists():
        print("Missing trace CSVs. Run first: python examples/compare_guard_vs_noguard.py")
        return

    gs, glr, gcos, gnorm, gstate, gloss = load_trace(guard_csv)
    ns, nlr, ncos, nnorm, nstate, nloss = load_trace(noguard_csv)

    fig, axes = plt.subplots(2, 3, figsize=(16, 8), dpi=150)

    # Learning rate
    ax = axes[0, 0]
    ax.plot(gs, glr, label="guard", color="#1f77b4")
    ax.plot(ns, nlr, label="no-guard", color="#ff7f0e")
    ax.set_title("Learning rate vs step")
    ax.set_xlabel("step")
    ax.set_ylabel("lr")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Gradient cosine
    ax = axes[0, 1]
    ax.plot(gs, gcos, label="guard", color="#1f77b4")
    ax.plot(ns, ncos, label="no-guard", color="#ff7f0e")
    ax.set_title("Gradient cosine vs step")
    ax.set_xlabel("step")
    ax.set_ylabel("grad_cos")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Gradient norm (proxy for stationarity)
    ax = axes[0, 2]
    ax.plot(gs, gnorm, label="guard", color="#1f77b4")
    ax.plot(ns, nnorm, label="no-guard", color="#ff7f0e")
    ax.set_title("Gradient norm vs step")
    ax.set_xlabel("step")
    ax.set_ylabel("grad_norm")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # State norm (stability)
    ax = axes[1, 0]
    ax.plot(gs, gstate, label="guard", color="#1f77b4")
    ax.plot(ns, nstate, label="no-guard", color="#ff7f0e")
    ax.set_title("State norm vs step")
    ax.set_xlabel("step")
    ax.set_ylabel("state_norm = ||[x;y]||")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Absolute loss (log scale if positive)
    ax = axes[1, 1]
    ax.plot(gs, gloss, label="guard", color="#1f77b4")
    ax.plot(ns, nloss, label="no-guard", color="#ff7f0e")
    ax.set_title("Absolute loss vs step")
    ax.set_xlabel("step")
    ax.set_ylabel("|L(x,y)|")
    if np.all(gloss > 0) and np.all(nloss > 0):
        ax.set_yscale("log")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Hide unused last axis
    axes[1, 2].axis("off")

    fig.tight_layout()
    out = Path("examples/compare_guard_vs_noguard.png")
    fig.savefig(out)
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
