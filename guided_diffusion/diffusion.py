import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
 
import torch
import torch.nn as nn
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
 
from data import GaussianCircleDataset
from schedule import T, betas, alphas, alphas_cumprod
from denoiser import MLPDenoiser

# Pre-computing tensors once
_acp = torch.tensor(alphas_cumprod, dtype=torch.float32) 
_sqrt_acp = _acp.sqrt()                                         
_sqrt_1ma = (1.0 - _acp).sqrt()                               


def q_sample_torch(x0: torch.Tensor, t: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    s1  = _sqrt_acp[t].unsqueeze(1)   # [B, 1]
    s2  = _sqrt_1ma[t].unsqueeze(1)   #same as s1
    eps = torch.randn_like(x0)
    return s1 * x0 + s2 * eps, eps

def train(
    n_epochs:   int   = 500,
    batch_size: int   = 512,
    lr:         float = 3e-4,
    hidden_dim: int   = 256,
    time_emb_dim: int = 32,
    log_every:  int   = 50,
    seed:       int   = 0,
) -> MLPDenoiser:
 
    torch.manual_seed(seed)
 
    # data
    dataset = GaussianCircleDataset()
    loader  = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)
 
    # model + optimiser + loss
    model     = MLPDenoiser(input_dim=2, hidden_dim=hidden_dim, time_emb_dim=time_emb_dim)
    optimiser = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    print(model)
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Dataset size: {len(dataset):,}   batch: {batch_size}   epochs: {n_epochs}\n")
    print(f"{'Epoch':>6}  {'Loss':>10}")
    print("─" * 20)

    epoch_losses = []
 
    for epoch in range(1, n_epochs + 1):
        model.train()
        running = 0.0
 
        for x0, _ in loader:   #sample data                
            x0 = x0.float()
 
            #picking random timestep per sample
            t = torch.randint(0, T, (x0.size(0),))  # [B]
 
            #adding noise
            xt, eps = q_sample_torch(x0, t) # [B,2], [B,2]
 
            # 4. predict noise
            eps_pred = model(xt, t) # [B,2]
 
            # 5. compute loss
            loss = criterion(eps_pred, eps)
 
            # 6. update
            optimiser.zero_grad()
            loss.backward()
            optimiser.step()
 
            running += loss.item()
 
        epoch_loss = running / len(loader)
        epoch_losses.append(epoch_loss)
 
        if epoch % log_every == 0 or epoch == 1:
            print(f"{epoch:>6}  {epoch_loss:>10.6f}")
 
    print("─" * 20)
    print(f"Final loss : {epoch_losses[-1]:.6f}")
    return model, epoch_losses


#Loss curve Plotting Function

def plot_loss(losses: list[float], path: str) -> None:
    fig, ax = plt.subplots(figsize=(8, 4), facecolor="#0d0d0d")
    ax.set_facecolor("#0d0d0d")
 
    epochs = range(1, len(losses) + 1)
    ax.plot(epochs, losses, color="#4fc3f7", linewidth=1.5, label="train loss")
 
    # smooth trend
    window = max(1, len(losses) // 20)
    smooth = np.convolve(losses, np.ones(window) / window, mode="valid")
    ax.plot(range(window, len(losses) + 1), smooth,
            color="#ff7043", linewidth=2.5, label=f"smoothed (w={window})")
 
    ax.set_xlabel("Epoch", color="#aaa")
    ax.set_ylabel("MSE Loss", color="#aaa")
    ax.set_title("DDPM Training Loss", color="white", fontsize=13)
    ax.tick_params(colors="#555")
    for spine in ax.spines.values():
        spine.set_edgecolor("#333")
    ax.legend(facecolor="#1a1a1a", edgecolor="#333",
              labelcolor="white", fontsize=9)
 
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    print(f"Loss curve  → {path}")


# Pre-computing reverse-process coefficients once
_betas    = torch.tensor(betas,   dtype=torch.float32)           
_alphas   = torch.tensor(alphas,  dtype=torch.float32)          
# posterior variance  β̃_t = β_t * (1 - ᾱ_{t-1}) / (1 - ᾱ_t)
_acp_prev = torch.cat([torch.ones(1), _acp[:-1]]) # ᾱ_{t-1}, pad t=0 with 1
_posterior_var = _betas * (1.0 - _acp_prev) / (1.0 - _acp)     
 
 
@torch.no_grad()
def p_sample_step(
    model: MLPDenoiser,
    xt:    torch.Tensor,   # [B, D]  current noisy sample
    t:     int,            # scalar timestep
) -> torch.Tensor:
    #One reverse step and DDPM reverse kernel
    B = xt.size(0)
    t_batch = torch.full((B,), t, dtype=torch.long)  
 
    # predicting noise
    eps_pred = model(xt, t_batch)                    
 
    # reverse-process mean
    coef = _betas[t] / _sqrt_1ma[t]                  
    mean = (1.0 / _alphas[t].sqrt()) * (xt - coef * eps_pred)
 
    if t == 0:
        return mean # no noise on final step
 
    # adding posterior noise
    noise = torch.randn_like(xt)
    std   = _posterior_var[t].clamp(min=1e-20).sqrt()
    return mean + std * noise
 
 
@torch.no_grad()
def p_sample_loop(
    model:     MLPDenoiser,
    n_samples: int = 2000,
    input_dim: int = 2,
) -> torch.Tensor:
    """
    Full reverse diffusion: start from pure noise, denoise T → 0.
 
    Returns:
    x0 : FloatTensor [n_samples, input_dim]
    """
    model.eval()
    xt = torch.randn(n_samples, input_dim)        
 
    for t in reversed(range(T)):                      
        xt = p_sample_step(model, xt, t)
 
    return xt # [n_samples, input_dim]
 
# Plot generated samples
 
def plot_samples(samples: torch.Tensor, path: str) -> None:
    pts = samples.numpy()
 
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), facecolor="#0d0d0d")
    fig.suptitle("DDPM unguided sampling  —  2 000 generated points",
                 color="white", fontsize=13, y=1.01)
 
    # generated samples
    ax = axes[0]
    ax.set_facecolor("#0d0d0d")
    ax.scatter(pts[:, 0], pts[:, 1],
               s=4, alpha=0.6, color="#4fc3f7", linewidths=0)
    ax.set_title("Generated (no labels)", color="white", fontsize=10)
    ax.set_xlim(-4.5, 4.5); ax.set_ylim(-4.5, 4.5)
    ax.set_aspect("equal")
    for sp in ax.spines.values(): sp.set_edgecolor("#333")
    ax.tick_params(colors="#555")
 
    # overlay on real data for comparison
    dataset = GaussianCircleDataset()
    real    = dataset.points.numpy()
 
    ax2 = axes[1]
    ax2.set_facecolor("#0d0d0d")
    ax2.scatter(real[:, 0], real[:, 1],
                s=2, alpha=0.25, color="#aaa", linewidths=0, label="real")
    ax2.scatter(pts[:, 0],  pts[:, 1],
                s=4, alpha=0.7, color="#ff7043", linewidths=0, label="generated")
    ax2.set_title("Generated vs Real", color="white", fontsize=10)
    ax2.set_xlim(-4.5, 4.5); ax2.set_ylim(-4.5, 4.5)
    ax2.set_aspect("equal")
    for sp in ax2.spines.values(): sp.set_edgecolor("#333")
    ax2.tick_params(colors="#555")
    ax2.legend(facecolor="#1a1a1a", edgecolor="#333",
               labelcolor="white", fontsize=9, markerscale=3)
 
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    print(f"Samples plot → {path}")
 
 

if __name__ == "__main__":
    model, losses = train(
        n_epochs   = 500,
        batch_size = 512,
        lr         = 3e-4,
        log_every  = 50,
    )
 
    plot_loss(losses, "/Users/ameygupta/sop_task/images/ddpm_loss.png")
 
    # save checkpoint
    ckpt_path = "/Users/ameygupta/sop_task/images/ddpm_model.pt"
    torch.save({
        "model_state": model.state_dict(),
        "input_dim":   model.input_dim,
        "losses":      losses,
    }, ckpt_path)
    print(f"Checkpoint  → {ckpt_path}")

    print("\nRunning reverse diffusion (2 000 samples) ...")
    samples = p_sample_loop(model, n_samples=2000, input_dim=2)
    print(f"Generated   : {samples.shape}")
 
    plot_samples(samples, "/Users/ameygupta/sop_task/images/ddpm_samples.png")