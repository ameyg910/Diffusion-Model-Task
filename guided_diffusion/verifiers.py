from __future__ import annotations
 
import math
from abc import ABC, abstractmethod
 
import torch
 
 
# Base class
 
class Verifier(ABC):
    """
    Abstract base class for all verifiers.
 
    Parameter:
    x : FloatTensor
    """
 
    @abstractmethod
    def log_value(self, x: torch.Tensor) -> torch.Tensor:
        """
        Computing log p(x).
 
        Returns:
        FloatTensor  [*]   – one scalar log-prob per input point
        """
        ...
 
    # default: autograd fallback
 
    def grad_log_value(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute  ∇_x log p(x)  via autograd.
 
        Parameters:
        x : FloatTensor  [*, D]
 
        Returns:
        FloatTensor  [*, D]  – same shape as x
        """
        x = x.detach().requires_grad_(True)
        log_p = self.log_value(x).sum()   # sum over batch for scalar backward
        log_p.backward()
        return x.grad.detach()        
 
    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        #write verifier(x) directly.
        return self.log_value(x)
 
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"
 
# Concrete verifier 1 – single isotropic Gaussian
 
class GaussianVerifier(Verifier):
    """
 
    (constant terms are dropped as they don't affect the gradient.)
 
    Parameters:
    mean: FloatTensor [D]  or scalar   
    std: float – isotropic standard deviation σ
    """
 
    def __init__(self, mean: torch.Tensor, std: float = 1.0):
        self.mean = mean.float()
        self.std  = std
 
    def log_value(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters:
        x : [*, D]
        """
        diff = x - self.mean                         # [*, D]
        return -(diff ** 2).sum(dim=-1) / (2 * self.std ** 2)
 
    def grad_log_value(self, x: torch.Tensor) -> torch.Tensor:
        """
        Closed-form score
 
        Parameters:
        x : [*, D]
 
        Returns:
        [*, D]
        """
        return -(x - self.mean) / (self.std ** 2)
 
    def __repr__(self) -> str:
        return (f"GaussianVerifier(mean={self.mean.tolist()}, std={self.std})")
 
# Concrete verifier 2 – Gaussian Mixture (matches the GMM dataset)
class GaussianMixtureVerifier(Verifier):
    """
    Log-density of an isotropic Gaussian Mixture Model
    Computed via log-sum-exp for numerical stability.
 
    By default the mixture is built to match GaussianCircleDataset:
        K=8 centres on a circle of radius=2, std=0.1
    """
 
    def __init__(self, means: torch.Tensor, std: float = 0.1):
        self.means = means.float()    # [K, D]
        self.std   = std
        self.K     = means.shape[0]
 
    # factory: build from GaussianCircleDataset defaults
 
    @classmethod
    def from_circle(
        cls,
        n_clusters: int   = 8,
        radius:     float = 2.0,
        std:        float = 0.1,
    ) -> "GaussianMixtureVerifier":
        """Construct a GMM verifier whose centres match the training dataset."""
        angles = torch.linspace(0, 2 * math.pi, n_clusters + 1)[:-1]
        means  = torch.stack([radius * torch.cos(angles),
                               radius * torch.sin(angles)], dim=1)   # [K, 2]
        return cls(means=means, std=std)
 
    # ── core methods ──────────────────────────────────────────────────────────
 
    def log_value(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters:
        x : [*, D]
 
        Returns:
        [*]   log p(x)  via log-sum-exp over K components
        """
        diff = x.unsqueeze(-2) - self.means     
        sq_dist = (diff ** 2).sum(dim=-1)               
        log_comps = -sq_dist / (2 * self.std ** 2)       
        return torch.logsumexp(log_comps, dim=-1)          
 
    def grad_log_value(self, x: torch.Tensor) -> torch.Tensor:
        """
        Closed-form score for a GMM
        """
        diff      = x.unsqueeze(-2) - self.means      
        sq_dist   = (diff ** 2).sum(dim=-1)               
        log_comps = -sq_dist / (2 * self.std ** 2)       
 
        #responsibilities  r_k = softmax over log_comps
        r = torch.softmax(log_comps, dim=-1)              
 
        # weighted sum of per-component gradients
        # per-component grad
        grad = -(r.unsqueeze(-1) * diff).sum(dim=-2) / (self.std ** 2)
        return grad                                         
 
    def __repr__(self) -> str:
        return (f"GaussianMixtureVerifier("
                f"K={self.K}, std={self.std})")

class HalfPlaneVerifier(Verifier):
    """
    Linear verifier that prefers the right half-plane.
 
    Log-density:
        log p(x) = x[axis] / temperature

 
    Gradient (closed-form, constant)
 
    Parameters:
    dim: dimensionality of the input
    axis: which coordinate to prefer 
    temperature: scale of the log-density  (lower temperature → sharper preference for the right side)
    """
 
    def __init__(
        self,
        dim:         int   = 2,
        axis:        int   = 0,
        temperature: float = 1.0,
    ):
        assert 0 <= axis < dim, f"axis {axis} out of range for dim {dim}"
        assert temperature > 0, "temperature must be positive"
        self.dim         = dim
        self.axis        = axis
        self.temperature = temperature
 
        # pre-build the constant gradient vector  e_axis / temperature
        _grad = torch.zeros(dim)
        _grad[axis] = 1.0 / temperature
        self._grad = _grad                   
 
    def log_value(self, x: torch.Tensor) -> torch.Tensor:
        return x[..., self.axis] / self.temperature
 
    def grad_log_value(self, x: torch.Tensor) -> torch.Tensor:
        return self._grad.expand_as(x) # broadcast to match batch shape
 
    def __repr__(self) -> str:
        direction = ["right", "up"][self.axis] if self.dim == 2 else f"axis={self.axis}"
        return (f"HalfPlaneVerifier("
                f"dim={self.dim}, axis={self.axis}, "
                f"temperature={self.temperature})  "
                f"# prefers {direction}")
    
class TargetPointVerifier(Verifier):
    """
    Verifier that prefers points close to a single target location.
 
 
    This is the log of an isotropic Gaussian centred at `target`, so:
      - x exactly at the target scores highest  (0 before the constant)
      - score decreases smoothly as x moves away
      - σ controls how sharp the preference is:
          small σ → tight peak, large σ → gentle hill
 
    Gradient (closed-form, points toward target)
    """
 
    def __init__(self, target: torch.Tensor, sigma: float = 1.0):
        assert sigma > 0, "sigma must be positive"
        self.target = target.float()   # [D]
        self.sigma  = sigma
 
    def log_value(self, x: torch.Tensor) -> torch.Tensor:
        diff = x - self.target                        
        return -(diff ** 2).sum(dim=-1) / (2 * self.sigma ** 2)
 
    def grad_log_value(self, x: torch.Tensor) -> torch.Tensor:
        return (self.target - x) / (self.sigma ** 2)
 
    def __repr__(self) -> str:
        return (f"TargetPointVerifier("
                f"target={self.target.tolist()}, sigma={self.sigma})")
 
 

 
if __name__ == "__main__":
    torch.manual_seed(0)
    B, D = 16, 2
 
    print("GaussianVerifier")
    mu  = torch.zeros(D)
    gv  = GaussianVerifier(mean=mu, std=1.0)
    x   = torch.randn(B, D)
 
    lv  = gv.log_value(x)
    glv = gv.grad_log_value(x)
 
    print(gv)
    print(f"log_value      : {lv.shape}  min={lv.min():.3f}  max={lv.max():.3f}")
    print(f"grad_log_value : {glv.shape}")
 
    # closed-form vs autograd must match
    glv_auto = Verifier.grad_log_value(gv, x)   # force base autograd path
    assert torch.allclose(glv, glv_auto, atol=1e-5), "Gradient mismatch"
    print("closed-form == autograd \n")
    print("GaussianMixtureVerifier: ")
    gmv = GaussianMixtureVerifier.from_circle()
    x   = torch.randn(B, D)
 
    lv  = gmv.log_value(x)
    glv = gmv.grad_log_value(x)
 
    print(gmv)
    print(f"log_value : {lv.shape}  min={lv.min():.3f}  max={lv.max():.3f}")
    print(f"grad_log_value : {glv.shape}")
 
    glv_auto = Verifier.grad_log_value(gmv, x)
    assert torch.allclose(glv, glv_auto, atol=1e-5), "Gradient mismatch!"
    print("closed-form == autograd\n")
 
    # log_value at cluster centres should be high 
    print("log_value sanity: centres vs random :")
    centres = gmv.means # [8, 2] — right on top of modes
    noise   = torch.randn(8, 2) * 3  # far from modes
 
    lv_centres = gmv.log_value(centres)
    lv_noise   = gmv.log_value(noise)
 
    print(f"At cluster centres : {lv_centres.mean():.3f}  (should be HIGH)")
    print(f"At random points   : {lv_noise.mean():.3f}  (should be LOWER)")
    assert lv_centres.mean() > lv_noise.mean(), "log_value not higher at modes"
    print("log_value higher at modes\n")
 
    # grad points toward nearest mode 
    print("grad sanity: gradient points toward nearest centre: ")
    x_test  = torch.tensor([[2.5, 0.0]]) # slightly outside cluster 0
    g       = gmv.grad_log_value(x_test)
    print(f"x         = {x_test.squeeze().tolist()}")
    print(f"∇log p(x) = {g.squeeze().tolist()}  (should point toward [2, 0])")
    assert g[0, 0] < 0, "x-component of grad should be negative (push left toward mode)"
    print("gradient direction correct")

    print("\nHalfPlaneVerifier:")
    hpv = HalfPlaneVerifier(dim=2, axis=0, temperature=1.0)
    print(hpv)
 
    # log_value: bigger x → bigger score
    x_left   = torch.tensor([[-3.0, 0.0], [-1.0, 0.0]])
    x_right  = torch.tensor([[ 1.0, 0.0], [ 3.0, 0.0]])
    lv_left  = hpv.log_value(x_left)
    lv_right = hpv.log_value(x_right)
    print(f"log_value(x_left)  = {lv_left.tolist()}   ← should be low")
    print(f"log_value(x_right) = {lv_right.tolist()}  ← should be high")
    assert (lv_right > lv_left).all(), "right side must score higher than left"
    print("bigger x → better")
 
    # gradient: constant vector pointing right, for ANY input
    x_batch = torch.randn(B, D) * 10  # wildly scattered points
    grad    = hpv.grad_log_value(x_batch)  # [B, 2]
    print(f"\ngrad_log_value shape : {grad.shape}")
    print(f"gradient (first 4 rows):\n{grad[:4]}")
 
    # every row must be exactly [1, 0] (e_axis / temperature)
    expected = torch.tensor([1.0, 0.0]).expand(B, -1)
    assert torch.allclose(grad, expected), "gradient is not constant [1, 0]"
    print("gradient is constant [1, 0] for all inputs")
 
    # gradient always points right (positive x-component, zero y-component)
    assert (grad[:, 0] > 0).all(),  "x-component must be positive (points right)"
    assert (grad[:, 1] == 0).all(), "y-component must be zero"
    print("gradient always points right")
 
    # closed-form == autograd
    glv_auto = Verifier.grad_log_value(hpv, x_batch)
    assert torch.allclose(grad, glv_auto, atol=1e-5), "closed-form != autograd"
    print("closed-form == autograd")
 
    # temperature scaling
    hpv2 = HalfPlaneVerifier(dim=2, axis=0, temperature=0.5)
    grad2 = hpv2.grad_log_value(x_batch)
    assert torch.allclose(grad2[:, 0], torch.full((B,), 2.0)), \
        "temperature=0.5 should double gradient magnitude"
    print("temperature scaling correct")

    print("\n TargetPointVerifier:")
    target = torch.tensor([2.0, 0.0])  # cluster 0 centre on the circle
    tpv    = TargetPointVerifier(target=target, sigma=1.0)
    print(tpv)
 
    # closer → better log_value
    x_near = torch.tensor([[2.1,  0.0], [2.0,  0.1]])   # very close to target
    x_far  = torch.tensor([[5.0,  4.0], [-3.0, -3.0]])  # far from target
    lv_near = tpv.log_value(x_near)
    lv_far  = tpv.log_value(x_far)
    print(f"log_value near target : {lv_near.tolist()}  ← high")
    print(f"log_value far  target : {lv_far.tolist()}  ← low")
    assert (lv_near > lv_far).all(), "closer points must score higher"
    print("closer → better")
 
    # log_value is maximised exactly at the target (should be 0)
    lv_at_target = tpv.log_value(target.unsqueeze(0))
    assert torch.allclose(lv_at_target, torch.zeros(1)), \
        f"log_value at target should be 0, got {lv_at_target.item()}"
    print(f"log_value at target = {lv_at_target.item():.4f}  (max = 0)")
 
    # gradient points toward the target from any position
    x_batch = torch.randn(B, D) * 3 # random points scattered around
    grad    = tpv.grad_log_value(x_batch)   
 
    # vector from x to target = (target - x) — must be parallel & same-dir as grad
    to_target = target - x_batch             
    dot = (grad * to_target).sum(dim=-1) # positive iff pointing toward target
    assert (dot >= 0).all(), "gradient must point toward target"
    print(f"gradient · (target - x) ≥ 0 for all {B} random points")
 
    # gradient is zero exactly at the target
    grad_at_target = tpv.grad_log_value(target.unsqueeze(0))
    assert torch.allclose(grad_at_target, torch.zeros(1, D), atol=1e-6), \
        "gradient at target must be zero"
    print("gradient = [0, 0] at target")
 
    # magnitude grows with distance
    x_close = target + torch.tensor([[0.1, 0.0]])
    x_away  = target + torch.tensor([[2.0, 0.0]])
    mag_close = tpv.grad_log_value(x_close).norm()
    mag_away  = tpv.grad_log_value(x_away).norm()
    assert mag_away > mag_close, "further away → stronger gradient pull"
    print(f"‖grad‖ close={mag_close:.3f}  far={mag_away:.3f}  (further → stronger)")
 
    # closed-form == autograd
    glv_auto = Verifier.grad_log_value(tpv, x_batch)
    assert torch.allclose(grad, glv_auto, atol=1e-5), "closed-form != autograd"
    print("closed-form == autograd")
 
    print("\nAll verifier checks passed.")
 