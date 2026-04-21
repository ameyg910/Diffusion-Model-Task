import math
import sys
import os
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data import GaussianCircleDataset


T = 1000  #total no. of diffusion steps 
beta_start = 1e-4 # β₁ #negligible noise
beta_end = 0.02 # β_T #moderate noise

betas = np.linspace(beta_start, beta_end, T, dtype=np.float64)  
alphas = 1.0 - betas                                              
alphas_cumprod = np.cumprod(alphas)                                       

def q_sample(
    x0:  np.ndarray,
    t:   int,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """
    Parameters:
    x0  : clean data array, any shape
    t   : 0-indexed timestep in [0, T-1]
    rng : numpy Generator

    Returns:
    xₜ  : noisy data, same shape as x0
    """
    if rng is None:
        rng = np.random.default_rng()

    acp = alphas_cumprod[t] 
    eps = rng.standard_normal(x0.shape)

    x_t = math.sqrt(acp) * x0 + math.sqrt(1.0 - acp) * eps
    return x_t

if __name__ == "__main__":
    print(f"betas [{betas[0]:.5f} … {betas[-1]:.4f}]   shape {betas.shape}")
    print(f"alphas [{alphas[0]:.5f} … {alphas[-1]:.4f}]   shape {alphas.shape}")
    print(f"alphas_cumprod [{alphas_cumprod[0]:.5f} … {alphas_cumprod[-1]:.6f}]   shape {alphas_cumprod.shape}")

    #load data from GaussianCircleDataset (data.py)
    dataset = GaussianCircleDataset()
    x0     = dataset.points.numpy()   # [10000, 2]
    labels = dataset.labels.numpy()   # [10000]
    print(f"\nLoaded {dataset}")

    rng_vis = np.random.default_rng(7)
    steps = {
        "t = 0  (clean)":       0,
        f"t = {T//2}  (mid)":   T // 2,
        f"t = {T-1}  (max)":    T - 1,
    }

    fig, axes = plt.subplots(1, 3, figsize=(14, 5), facecolor="#0d0d0d")
    fig.suptitle(
        "DDPM forward process",
        color="white", fontsize=13, y=1.02,
    )

    for ax, (title, t) in zip(axes, steps.items()):
        xt  = q_sample(x0, t, rng=rng_vis)
        acp = alphas_cumprod[t]

        ax.set_facecolor("#0d0d0d")
        ax.scatter(xt[:, 0], xt[:, 1],
                   c=labels, cmap="tab10",
                   s=2, alpha=0.55, linewidths=0)
        ax.set_title(f"{title}\nᾱₜ = {acp:.4f}",
                     color="white", fontsize=10, pad=8)
        ax.set_xlim(-4.5, 4.5)
        ax.set_ylim(-4.5, 4.5)
        ax.set_aspect("equal")
        for spine in ax.spines.values():
            spine.set_edgecolor("#333")
        ax.tick_params(colors="#555")
        ax.set_xlabel("x", color="#666")
        ax.set_ylabel("y", color="#666")

    plt.tight_layout()
    out = "/Users/ameygupta/sop_task/images/ddpm_forward.png"
    plt.savefig(out, dpi=160, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    print(f"\nPlot saved → {out}")
    plt.show()
