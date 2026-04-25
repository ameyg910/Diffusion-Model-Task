"""
Minimal tests for the DDPM linear beta schedule and q_sample forward kernel.

Covers:
  - Shape and dtype of betas, alphas, alphas_cumprod
  - betas is monotonically increasing (linear schedule)
  - betas is bounded within (0, 1)
  - alphas = 1 - betas
  - alphas_cumprod is monotonically decreasing
  - alphas_cumprod[0] ≈ 1  (almost no noise at first step)
  - alphas_cumprod[-1] ≈ 0 (signal destroyed by final step)
  - q_sample at t=0  → output stays close to x0  (ā ≈ 1)
  - q_sample at t=T-1 → output is nearly pure noise  (ā ≈ 0)
  - q_sample output shape matches x0 shape
  - q_sample is reproducible under the same rng seed
  - q_sample mean and variance match the closed-form kernel exactly
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import math
import numpy as np
import pytest

from guided_diffusion.schedule import T, betas, alphas, alphas_cumprod, q_sample


# 1.  Schedule tensor properties

class TestScheduleTensors:

    def test_length(self):
        """All schedule tensors must have exactly T entries."""
        assert len(betas)          == T
        assert len(alphas)         == T
        assert len(alphas_cumprod) == T

    def test_betas_bounds(self):
        """Every beta must be strictly between 0 and 1."""
        assert (betas > 0).all(),  "betas must be positive"
        assert (betas < 1).all(),  "betas must be less than 1"

    def test_betas_endpoints(self):
        """Linear schedule must start at beta_start=1e-4 and end at beta_end=0.02."""
        assert math.isclose(betas[0],  1e-4, rel_tol=1e-6), \
            f"betas[0] should be 1e-4, got {betas[0]}"
        assert math.isclose(betas[-1], 0.02, rel_tol=1e-6), \
            f"betas[-1] should be 0.02, got {betas[-1]}"

    def test_betas_monotonically_increasing(self):
        """Linear schedule ⟹ each beta >= the previous one."""
        assert (np.diff(betas) >= 0).all(), \
            "betas must be non-decreasing (linear schedule)"

    def test_alphas_equals_one_minus_betas(self):
        """alphas[t] must equal 1 - betas[t] exactly."""
        assert np.allclose(alphas, 1.0 - betas, atol=0, rtol=0), \
            "alphas must equal 1 - betas element-wise"

    def test_alphas_bounds(self):
        """All alphas must be in (0, 1)."""
        assert (alphas > 0).all()
        assert (alphas < 1).all()

    def test_alphas_cumprod_is_cumulative_product(self):
        """alphas_cumprod[t] must equal prod(alphas[0:t+1])."""
        expected = np.cumprod(alphas)
        assert np.allclose(alphas_cumprod, expected, atol=1e-12), \
            "alphas_cumprod must be the cumulative product of alphas"

    def test_alphas_cumprod_monotonically_decreasing(self):
        """Signal decays monotonically — each ā_t <= ā_{t-1}."""
        assert (np.diff(alphas_cumprod) <= 0).all(), \
            "alphas_cumprod must be non-increasing"

    def test_alphas_cumprod_starts_near_one(self):
        """At t=0, almost no noise has been added yet: ā_0 should be ≈ 1."""
        assert alphas_cumprod[0] > 0.999, \
            f"alphas_cumprod[0] should be close to 1, got {alphas_cumprod[0]}"

    def test_alphas_cumprod_ends_near_zero(self):
        """At t=T-1, signal is almost fully destroyed: ā_T should be ≈ 0."""
        assert alphas_cumprod[-1] < 1e-3, \
            f"alphas_cumprod[-1] should be close to 0, got {alphas_cumprod[-1]}"

    def test_alphas_cumprod_strictly_positive(self):
        """ā_t must never reach exactly 0 (would make q_sample degenerate)."""
        assert (alphas_cumprod > 0).all(), \
            "alphas_cumprod must be strictly positive at all steps"


# 2.  q_sample forward kernel

class TestQSample:

    @pytest.fixture
    def clean_batch(self):
        """A small deterministic batch of clean 2-D points."""
        rng = np.random.default_rng(0)
        return rng.standard_normal((256, 2)).astype(np.float64)

    def test_output_shape_preserved(self, clean_batch):
        """q_sample must return an array with the same shape as x0."""
        rng = np.random.default_rng(1)
        xt  = q_sample(clean_batch, t=500, rng=rng)
        assert xt.shape == clean_batch.shape, \
            f"Shape mismatch: got {xt.shape}, expected {clean_batch.shape}"

    def test_output_shape_arbitrary_dims(self):
        """Shape preservation must hold for any number of dimensions."""
        rng = np.random.default_rng(2)
        for shape in [(10,), (16, 8), (4, 4, 4)]:
            x0 = rng.standard_normal(shape)
            xt = q_sample(x0, t=100, rng=rng)
            assert xt.shape == x0.shape, \
                f"Shape mismatch for input shape {shape}"

    def test_t0_output_close_to_x0(self, clean_batch):
        """
        At t=0, ā_0 ≈ 1, so x_t ≈ x_0.
        The noisy output should be very close to the clean input.
        """
        rng = np.random.default_rng(3)
        xt  = q_sample(clean_batch, t=0, rng=rng)
        # ā_0 ≈ 0.9999, std of noise term ≈ sqrt(1-0.9999) ≈ 0.01
        max_diff = np.abs(xt - clean_batch).max()
        assert max_diff < 0.2, \
            f"At t=0, q_sample should stay close to x0; max diff={max_diff:.4f}"

    def test_tmax_output_is_near_pure_noise(self, clean_batch):
        """
        At t=T-1, ā_{T-1} ≈ 0, so x_t ≈ ε ~ N(0, I).
        The signal from x0 should be almost entirely gone.
        """
        rng  = np.random.default_rng(4)
        xt   = q_sample(clean_batch, t=T - 1, rng=rng)
        # sqrt(ā_{T-1}) ≈ 0.006, so x0 contributes < 1% of the variance
        signal_contribution = math.sqrt(alphas_cumprod[T - 1])
        assert signal_contribution < 0.01, \
            "ā_{T-1} should be near 0 so the signal is negligible"
        # std of xt should be close to 1 (pure N(0,I))
        std = xt.std()
        assert 0.85 < std < 1.15, \
            f"At t=T-1, xt should look like N(0,I); got std={std:.3f}"

    def test_mean_matches_closed_form(self, clean_batch):
        """
        E[x_t | x_0] = sqrt(ā_t) * x_0.
        Average over many noise samples to verify empirically.
        """
        t   = 300
        acp = alphas_cumprod[t]
        N   = 2000
        x0  = clean_batch[:1].repeat(N, axis=0)   # [N, 2] same point repeated

        samples = np.stack([
            q_sample(x0[0:1], t=t, rng=np.random.default_rng(s))
            for s in range(N)
        ], axis=0).squeeze()   # [N, 2]

        empirical_mean = samples.mean(axis=0)
        expected_mean  = math.sqrt(acp) * clean_batch[0]

        assert np.allclose(empirical_mean, expected_mean, atol=0.05), \
            (f"Empirical mean {empirical_mean} does not match "
             f"closed-form E[x_t|x_0]={expected_mean}")

    def test_variance_matches_closed_form(self, clean_batch):
        """
        Var[x_t | x_0] = (1 - ā_t).
        Each dimension should have variance close to (1 - ā_t).
        """
        t   = 300
        acp = alphas_cumprod[t]
        N   = 2000
        x0_fixed = clean_batch[0:1]   # single point [1, 2]

        samples = np.stack([
            q_sample(x0_fixed, t=t, rng=np.random.default_rng(s))
            for s in range(N)
        ], axis=0).squeeze()   # [N, 2]

        empirical_var = samples.var(axis=0)
        expected_var  = 1.0 - acp

        assert np.allclose(empirical_var, expected_var, atol=0.05), \
            (f"Empirical variance {empirical_var} does not match "
             f"closed-form Var[x_t|x_0]={expected_var:.4f}")

    def test_reproducible_with_same_seed(self, clean_batch):
        """Same rng seed must produce bit-for-bit identical output."""
        out_a = q_sample(clean_batch, t=200, rng=np.random.default_rng(99))
        out_b = q_sample(clean_batch, t=200, rng=np.random.default_rng(99))
        assert np.array_equal(out_a, out_b), \
            "q_sample must be deterministic under the same rng seed"

    def test_different_seeds_differ(self, clean_batch):
        """Different rng seeds must (almost certainly) produce different outputs."""
        out_a = q_sample(clean_batch, t=200, rng=np.random.default_rng(1))
        out_b = q_sample(clean_batch, t=200, rng=np.random.default_rng(2))
        assert not np.array_equal(out_a, out_b), \
            "q_sample with different seeds should produce different noise"

    @pytest.mark.parametrize("t", [0, 1, 100, 499, 500, 750, 998, 999])
    def test_valid_at_all_timesteps(self, clean_batch, t):
        """q_sample must run without error and return finite values at every t."""
        rng = np.random.default_rng(t)
        xt  = q_sample(clean_batch, t=t, rng=rng)
        assert xt.shape == clean_batch.shape
        assert np.isfinite(xt).all(), \
            f"q_sample returned non-finite values at t={t}"

    def test_interpolation_between_clean_and_noise(self, clean_batch):
        """
        As t increases, q_sample output should drift away from x0.
        Mean squared distance to x0 must increase with t.
        """
        rng = np.random.default_rng(42)
        prev_msd = 0.0
        for t in [0, 100, 300, 600, 900, 999]:
            xt  = q_sample(clean_batch, t=t, rng=np.random.default_rng(t))
            msd = float(((xt - clean_batch) ** 2).mean())
            assert msd >= prev_msd - 1e-6, \
                f"MSD should increase with t; at t={t} got {msd:.4f} < prev {prev_msd:.4f}"
            prev_msd = msd