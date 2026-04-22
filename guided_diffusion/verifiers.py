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