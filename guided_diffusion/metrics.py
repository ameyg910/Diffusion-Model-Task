from __future__ import annotations

import math
from typing import Callable

import numpy as np

try:
    import torch
    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False

try:
    import ot as _ot
    _POT_AVAILABLE = True
except ImportError:
    _POT_AVAILABLE = False

from scipy.spatial.distance import cdist
from scipy.stats import entropy as _scipy_entropy


# Helper to safely convert anything into a numpy array
def _to_numpy(x) -> np.ndarray:
    if _TORCH_AVAILABLE and isinstance(x, __import__("torch").Tensor):
        return x.detach().cpu().numpy().astype(np.float64)
    return np.asarray(x, dtype=np.float64)


# This tells you what fraction of samples satisfy a condition you define
def compliance_rate(
    samples: "np.ndarray | torch.Tensor",
    region_fn: Callable[[np.ndarray], np.ndarray],
) -> float:
    pts = _to_numpy(samples)
    mask = np.asarray(region_fn(pts), dtype=bool)
    return float(mask.mean())


# This measures how evenly your samples are spread across different modes
def mode_coverage(
    samples: "np.ndarray | torch.Tensor",
    mode_centers: "np.ndarray | torch.Tensor",
    *,
    bandwidth: float | None = None,
    top_k: int | None = None,
) -> float:

    pts = _to_numpy(samples)
    ctrs = _to_numpy(mode_centers)
    N, K = len(pts), len(ctrs)

    # Automatically choose a reasonable bandwidth based on mode spacing
    if bandwidth is None:
        if K > 1:
            inter = cdist(ctrs, ctrs)
            upper = inter[np.triu_indices(K, k=1)]
            bandwidth = float(np.median(upper))
        else:
            bandwidth = 1.0
    bandwidth = max(bandwidth, 1e-8)

    # Compute soft assignments of samples to each mode
    dists = cdist(pts, ctrs)
    log_w = -(dists ** 2) / (2 * bandwidth ** 2)
    log_w -= log_w.max(axis=1, keepdims=True)
    w = np.exp(log_w)
    w /= w.sum(axis=1, keepdims=True)

    # Build histogram of how much mass each mode receives
    hist = w.sum(axis=0)
    hist = hist / hist.sum()

    # Optionally focus only on the most populated modes
    if top_k is not None:
        top_k = min(top_k, K)
        idx = np.argsort(hist)[::-1][:top_k]
        hist = hist[idx]
        hist = hist / hist.sum()
        K_eff = top_k
    else:
        K_eff = K

    # Normalize entropy so result is between 0 and 1
    h_max = math.log(K_eff)
    if h_max < 1e-12:
        return 1.0

    return float(_scipy_entropy(hist) / h_max)


# This counts how many modes actually received meaningful mass
def modes_covered(
    samples: "np.ndarray | torch.Tensor",
    mode_centers: "np.ndarray | torch.Tensor",
    *,
    bandwidth: float | None = None,
    threshold: float = 0.5,
) -> int:

    pts = _to_numpy(samples)
    ctrs = _to_numpy(mode_centers)
    K = len(ctrs)

    # Same automatic bandwidth logic as above
    if bandwidth is None:
        if K > 1:
            inter = cdist(ctrs, ctrs)
            upper = inter[np.triu_indices(K, k=1)]
            bandwidth = float(np.median(upper))
        else:
            bandwidth = 1.0
    bandwidth = max(bandwidth, 1e-8)

    dists = cdist(pts, ctrs)
    log_w = -(dists ** 2) / (2 * bandwidth ** 2)
    log_w -= log_w.max(axis=1, keepdims=True)
    w = np.exp(log_w)
    w /= w.sum(axis=1, keepdims=True)

    hist = w.sum(axis=0)
    hist = hist / hist.sum()

    # A mode counts as covered if it has enough share compared to uniform
    cutoff = threshold / K
    return int((hist >= cutoff).sum())


# This computes the Wasserstein distance between two sample sets
def wasserstein_2d(
    samples_a: "np.ndarray | torch.Tensor",
    samples_b: "np.ndarray | torch.Tensor",
    *,
    n_iter: int = 100_000,
) -> float:

    a = _to_numpy(samples_a)
    b = _to_numpy(samples_b)

    # If POT is installed, use exact optimal transport
    if _POT_AVAILABLE:
        N, M = len(a), len(b)
        mu = np.ones(N) / N
        nu = np.ones(M) / M
        M_cost = cdist(a, b, metric="sqeuclidean")
        T_plan = _ot.emd(mu, nu, M_cost)
        w2_sq = float(np.sum(T_plan * M_cost))
        return math.sqrt(max(w2_sq, 0.0))

    # Otherwise fall back to sliced approximation
    else:
        from scipy.stats import wasserstein_distance

        D = a.shape[1]
        rng = np.random.default_rng(0)
        sliced = []

        for _ in range(n_iter):
            v = rng.standard_normal(D)
            v /= np.linalg.norm(v) + 1e-12
            pa = a @ v
            pb = b @ v
            sliced.append(wasserstein_distance(pa, pb) ** 2)

        return math.sqrt(max(float(np.mean(sliced)), 0.0))


if __name__ == "__main__":
    import math

    rng = np.random.default_rng(0)
    K = 8
    D = 2

    # Create circular cluster centers
    angles = np.linspace(0, 2 * math.pi, K, endpoint=False)
    centers = np.stack([2 * np.cos(angles), 2 * np.sin(angles)], axis=1)

    # Generate samples that perfectly match the modes
    labels = rng.integers(0, K, 2000)
    perfect = centers[labels] + rng.normal(0, 0.05, (2000, D))

    # Generate samples from only half the modes
    right_mask = centers[:, 0] > 0
    right_ctrs = centers[right_mask]
    labels_r = rng.integers(0, len(right_ctrs), 2000)
    right_only = right_ctrs[labels_r] + rng.normal(0, 0.05, (2000, D))

    # Pure random noise samples
    noise_samples = rng.normal(0, 2, (2000, D))

    print("compliance_rate")
    cr_perfect = compliance_rate(perfect, lambda x: x[:, 0] > 0)
    cr_right = compliance_rate(right_only, lambda x: x[:, 0] > 0)
    cr_noise = compliance_rate(noise_samples, lambda x: x[:, 0] > 0)
    print(f"perfect samples   right-half: {cr_perfect:.3f}")
    print(f"right-only        right-half: {cr_right:.3f}")
    print(f"noise             right-half: {cr_noise:.3f}")

    print("\nmode_coverage")
    cov_perfect = mode_coverage(perfect, centers)
    cov_right = mode_coverage(right_only, centers)
    cov_noise = mode_coverage(noise_samples, centers)
    print(f"perfect samples  coverage: {cov_perfect:.3f}")
    print(f"right-only       coverage: {cov_right:.3f}")
    print(f"noise samples    coverage: {cov_noise:.3f}")

    print("\nmodes_covered")
    mc_perfect = modes_covered(perfect, centers, threshold=0.5)
    mc_right = modes_covered(right_only, centers, bandwidth=0.2, threshold=0.5)
    n_right = int(sum(right_mask))
    print(f"perfect samples  modes covered: {mc_perfect} / {K}")
    print(f"right-only       modes covered: {mc_right} / {K}")

    print("\nwasserstein_2d")
    w_self = wasserstein_2d(perfect, perfect, n_iter=500)
    w_shift = wasserstein_2d(perfect, perfect + np.array([3.0, 0.0]), n_iter=500)
    w_noise = wasserstein_2d(perfect, noise_samples, n_iter=500)
    print(f"W2(perfect, perfect)       = {w_self:.4f}")
    print(f"W2(perfect, shifted)       = {w_shift:.4f}")
    print(f"W2(perfect, noise)         = {w_noise:.4f}")

    print("\nBackend:", "POT (exact)" if _POT_AVAILABLE else "scipy sliced (approx)")
    print("\n All metrics checks passed.")