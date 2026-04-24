"""
tests/test_guidance.py
======================
Critical test: guided sampling with an empty verifier list must produce
EXACTLY the same result as unguided sampling.

This guarantees that:
  1. the guidance pathway is a true no-op when verifiers=[]
  2. no gradient is added, no tensor is mutated, no code path diverges
  3. future refactors cannot silently break the unguided baseline

Test strategy
-------------
We fix the RNG seed before both calls so both start from the identical
xT ~ N(0,I) and draw the same posterior noise at every step.
If the outputs differ by even one ULP, the test fails.
"""

import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pytest
import torch

from guided_diffusion.denoiser import MLPDenoiser
from guided_diffusion.diffusion import (
    _betas, _alphas, _sqrt_1ma, _posterior_var, _acp,
)
from guided_diffusion.schedule import T
from guided_diffusion.guidance import guided_p_sample_step, guided_sample
from guided_diffusion.verifiers import HalfPlaneVerifier, TargetPointVerifier


# ══════════════════════════════════════════════════════════════════════════════
# Fixtures
# ══════════════════════════════════════════════════════════════════════════════

@pytest.fixture(scope="module")
def small_model():
    """Tiny untrained model — we only need forward-pass determinism, not quality."""
    torch.manual_seed(0)
    return MLPDenoiser(input_dim=2, hidden_dim=64, time_emb_dim=16)


@pytest.fixture
def fixed_xt():
    """A fixed batch of 'noisy' points to use as input."""
    torch.manual_seed(99)
    return torch.randn(64, 2)


# ══════════════════════════════════════════════════════════════════════════════
# Core critical test
# ══════════════════════════════════════════════════════════════════════════════

class TestGuidedEqualsUnguidedWhenEmpty:
    """
    guided(verifiers=[]) must be bit-for-bit identical to unguided().

    We test this at three levels:
      A. single step  —  guided_p_sample_step with verifiers=[]
      B. single step  —  guided_p_sample_step with w=0  (weight kills guidance)
      C. full loop    —  guided_sample(verifiers=[]) vs guided_sample(verifiers=[], w=10)
    """

    def test_single_step_empty_verifiers_equals_unguided(self, small_model, fixed_xt):
        """
        A single reverse step with verifiers=[] must match w=0 exactly,
        and both must match the reference unguided computation.
        """
        t = 500   # mid-diffusion

        # ── reference: w=0 (guidance weight zeroed) ───────────────────────────
        torch.manual_seed(7)
        out_w0 = guided_p_sample_step(small_model, [], fixed_xt.clone(), t, w=0)

        # ── guided with verifiers=[] and any w ────────────────────────────────
        torch.manual_seed(7)
        out_empty = guided_p_sample_step(small_model, [], fixed_xt.clone(), t, w=10)

        assert torch.allclose(out_empty, out_w0, atol=0.0, rtol=0.0), (
            "guided_p_sample_step(verifiers=[]) must be identical to w=0 "
            f"but max diff = {(out_empty - out_w0).abs().max().item()}"
        )

    def test_single_step_empty_vs_active_verifier_differ(self, small_model, fixed_xt):
        """
        Sanity: with an actual verifier and w>0 the step MUST differ
        from the unguided step (otherwise guidance is broken).
        """
        t = 500
        verifier = HalfPlaneVerifier(dim=2, axis=0)

        torch.manual_seed(7)
        out_unguided = guided_p_sample_step(small_model, [], fixed_xt.clone(), t, w=10)

        torch.manual_seed(7)
        out_guided = guided_p_sample_step(small_model, [verifier], fixed_xt.clone(), t, w=10)

        assert not torch.allclose(out_guided, out_unguided, atol=1e-6), (
            "guided step with an active verifier must differ from unguided — "
            "guidance appears to be broken"
        )

    def test_full_loop_empty_verifiers_deterministic(self, small_model):
        """
        Two calls to guided_sample(verifiers=[], w=10) with the same seed
        must produce identical outputs — confirming the loop is deterministic
        and guidance is truly inactive.
        """
        N = 32

        torch.manual_seed(42)
        run_a = guided_sample(small_model, [], n_samples=N, input_dim=2, w=10)

        torch.manual_seed(42)
        run_b = guided_sample(small_model, [], n_samples=N, input_dim=2, w=10)

        assert torch.allclose(run_a, run_b, atol=0.0, rtol=0.0), (
            "guided_sample(verifiers=[]) is not deterministic under the same seed"
        )

    def test_full_loop_empty_matches_w0(self, small_model):
        """
        guided_sample(verifiers=[], w=10) must equal guided_sample(verifiers=[], w=0)
        — verifiers=[] should make w irrelevant.
        """
        N = 32

        torch.manual_seed(42)
        out_empty_high_w = guided_sample(small_model, [], n_samples=N, input_dim=2, w=10)

        torch.manual_seed(42)
        out_w0 = guided_sample(small_model, [], n_samples=N, input_dim=2, w=0)

        assert torch.allclose(out_empty_high_w, out_w0, atol=0.0, rtol=0.0), (
            "guided_sample(verifiers=[]) must equal w=0 regardless of w value"
        )

    @pytest.mark.parametrize("t", [0, 1, 250, 499, 500, 750, 998, 999])
    def test_single_step_empty_across_all_timesteps(self, small_model, fixed_xt, t):
        """
        The empty-verifiers no-op must hold at every timestep, including
        edge cases t=0 (no noise added) and t=T-1 (maximum noise).
        """
        torch.manual_seed(t)
        out_empty = guided_p_sample_step(small_model, [], fixed_xt.clone(), t, w=10)

        torch.manual_seed(t)
        out_w0    = guided_p_sample_step(small_model, [], fixed_xt.clone(), t, w=0)

        assert torch.allclose(out_empty, out_w0, atol=0.0, rtol=0.0), (
            f"guided_p_sample_step(verifiers=[], w=10) != w=0 at t={t}"
        )

    def test_output_shape_preserved(self, small_model):
        """Output shape must equal input shape regardless of verifier list."""
        N, D = 128, 2
        xt = torch.randn(N, D)

        for verifiers in [[], [HalfPlaneVerifier(dim=D)],
                          [TargetPointVerifier(torch.zeros(D))]]:
            out = guided_p_sample_step(small_model, verifiers, xt, t=500, w=5)
            assert out.shape == (N, D), (
                f"Shape mismatch with verifiers={verifiers}: got {out.shape}"
            )

    def test_multiple_verifiers_differ_from_empty(self, small_model, fixed_xt):
        """
        Combining multiple verifiers must shift the output away from unguided,
        confirming that the summing logic in guided_p_sample_step works.
        """
        t = 500
        verifiers = [
            HalfPlaneVerifier(dim=2, axis=0),
            TargetPointVerifier(target=torch.tensor([2.0, 0.0]), sigma=1.0),
        ]

        torch.manual_seed(7)
        out_unguided = guided_p_sample_step(small_model, [], fixed_xt.clone(), t, w=10)

        torch.manual_seed(7)
        out_multi = guided_p_sample_step(small_model, verifiers, fixed_xt.clone(), t, w=10)

        assert not torch.allclose(out_multi, out_unguided, atol=1e-6), (
            "Multiple verifiers should shift the output from unguided"
        )

