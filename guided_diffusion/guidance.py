from __future__ import annotations

import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from diffusion import (
    _betas, _alphas, _sqrt_1ma, _posterior_var,
    _acp, _sqrt_acp,
)
from schedule import T
from denoiser import MLPDenoiser
from verifiers import Verifier, HalfPlaneVerifier, TargetPointVerifier


# This performs a single reverse diffusion step and optionally nudges the sample using verifiers
def guided_p_sample_step(
    model: MLPDenoiser,
    verifiers: list,
    xt: torch.Tensor,
    t: int,
    w: float = 1.0,
) -> torch.Tensor:

    B = xt.size(0)
    t_batch = torch.full((B,), t, dtype=torch.long)

    # Predict noise and compute the standard DDPM mean
    with torch.no_grad():
        eps_pred = model(xt, t_batch)

    coef = _betas[t] / _sqrt_1ma[t]
    mean = (1.0 / _alphas[t].sqrt()) * (xt - coef * eps_pred)

    # If guidance is enabled, shift the mean using gradients from all verifiers
    if verifiers and w != 0.0:
        grad = sum(v.grad_log_value(xt.detach()) for v in verifiers)
        var = _posterior_var[t].clamp(min=1e-20)
        mean = mean + w * var * grad

    # Add noise unless we are at the final step
    if t == 0:
        return mean

    std = _posterior_var[t].clamp(min=1e-20).sqrt()
    noise = torch.randn_like(xt)
    return mean + std * noise


# This runs the full reverse diffusion process starting from random noise
@torch.no_grad()
def guided_sample(
    model: MLPDenoiser,
    verifiers: list,
    n_samples: int = 2000,
    input_dim: int = 2,
    w: float = 1.0,
) -> torch.Tensor:

    model.eval()
    xt = torch.randn(n_samples, input_dim)

    for t in reversed(range(T)):
        xt = guided_p_sample_step(model, verifiers, xt, t, w=w)

    return xt


# This helps visualize how different guidance strengths affect the samples
def plot_guidance_comparison(
    samples_by_w: dict,
    verifier: Verifier,
    title: str,
    path: str,
) -> None:

    n = len(samples_by_w)
    fig, axes = plt.subplots(1, n, figsize=(6 * n, 5), facecolor="#0d0d0d")
    fig.suptitle(title, color="white", fontsize=13, y=1.01)

    if n == 1:
        axes = [axes]

    colors = ["#4fc3f7", "#ff7043", "#a5d6a7", "#ce93d8"]

    for ax, (w, pts), color in zip(axes, samples_by_w.items(), colors):
        pts = pts.numpy()
        ax.set_facecolor("#0d0d0d")
        ax.scatter(
            pts[:, 0], pts[:, 1],
            s=4, alpha=0.6, color=color, linewidths=0
        )
        ax.set_title(f"w = {w}", color="white", fontsize=11, pad=8)
        ax.set_xlim(-5, 5)
        ax.set_ylim(-5, 5)
        ax.set_aspect("equal")

        for sp in ax.spines.values():
            sp.set_edgecolor("#333")

        ax.tick_params(colors="#555")
        ax.set_xlabel("x", color="#666")
        ax.set_ylabel("y", color="#666")

    plt.tight_layout()
    plt.savefig(
        path,
        dpi=150,
        bbox_inches="tight",
        facecolor=fig.get_facecolor()
    )
    print(f"Saved → {path}")


if __name__ == "__main__":
    import argparse

    # Load trained model checkpoint
    ckpt_path = "/Users/ameygupta/sop_task/images/ddpm_model.pt"
    ckpt = torch.load(ckpt_path, map_location="cpu")

    model = MLPDenoiser(input_dim=ckpt["input_dim"])
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    print(f"Loaded checkpoint ({ckpt['input_dim']}-D model)")

    # Test 1 checks whether guidance pushes samples toward the right side
    print("\nHalfPlaneVerifier guidance (w=0 vs w=10)")
    hpv = HalfPlaneVerifier(dim=2, axis=0, temperature=1.0)

    samples_hp = {}
    for w in [0, 10]:
        print(f"sampling w={w}")
        samples_hp[w] = guided_sample(model, [hpv], n_samples=2000, w=w)

    mean_x_w0 = samples_hp[0][:, 0].mean().item()
    mean_x_w10 = samples_hp[10][:, 0].mean().item()

    print(f"mean x | w=0: {mean_x_w0:.3f}   w=10: {mean_x_w10:.3f}")

    assert mean_x_w10 > mean_x_w0, "Guidance should shift points to the right!"
    print("points shift right with guidance")

    plot_guidance_comparison(
        samples_hp,
        hpv,
        title="HalfPlane guidance prefer right (w=0 vs w=10)",
        path="/Users/ameygupta/sop_task/images/guidance_halfplane.png",
    )

    # Test 2 checks whether guidance pulls samples toward a specific point
    print("\nTargetPointVerifier guidance (w=0 vs w=10)")

    target = torch.tensor([2.0, 0.0])
    tpv = TargetPointVerifier(target=target, sigma=1.0)

    samples_tp = {}
    for w in [0, 10]:
        print(f"sampling w={w}")
        samples_tp[w] = guided_sample(model, [tpv], n_samples=2000, w=w)

    dist_w0 = (samples_tp[0] - target).norm(dim=1).mean().item()
    dist_w10 = (samples_tp[10] - target).norm(dim=1).mean().item()

    print(f"mean distance to target | w=0: {dist_w0:.3f}   w=10: {dist_w10:.3f}")

    assert dist_w10 < dist_w0, "Guidance should pull points toward the target!"
    print("points move closer to target with guidance")

    plot_guidance_comparison(
        samples_tp,
        tpv,
        title="TargetPoint guidance prefer [2, 0] (w=0 vs w=10)",
        path="/Users/ameygupta/sop_task/images/guidance_targetpoint.png",
    )

    print("\nAll guidance checks passed")