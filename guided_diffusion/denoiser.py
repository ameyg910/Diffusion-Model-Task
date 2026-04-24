import math
import torch
import torch.nn as nn


class SinusoidalTimeEmbedding(nn.Module):
    """
    Maps a scalar timestep t  →  vector of shape (time_emb_dim,).

    Parameters:
    time_emb_dim : embedding dimension  (must be even)
    """

    def __init__(self, time_emb_dim: int = 32):
        super().__init__()
        assert time_emb_dim % 2 == 0, "false input"
        self.dim = time_emb_dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """
        Parameters:
        t : LongTensor  [B]   – 0-indexed timesteps

        Returns:
        emb : FloatTensor  [B, time_emb_dim]
        """
        device = t.device
        half   = self.dim // 2
        freqs  = torch.exp(
            -math.log(10_000) * torch.arange(half, device=device) / (half - 1)
        )                                        # [half]
        args   = t.float().unsqueeze(1) * freqs  # [B, half]
        emb    = torch.cat([torch.sin(args), torch.cos(args)], dim=1)  # [B, dim]
        return emb


class MLPDenoiser(nn.Module):
    """
    Predicts the noise  ε  added to x at timestep t.

    Parameters:
    input_dim : dimensionality of the data
    hidden_dim : width of the two hidden layers        
    time_emb_dim : size of the sinusoidal time embedding
    """

    def __init__(
        self,
        input_dim: int = 2,
        hidden_dim: int = 256,
        time_emb_dim: int = 32,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.time_emb_dim = time_emb_dim

        self.time_mlp = SinusoidalTimeEmbedding(time_emb_dim)

        in_features = input_dim + time_emb_dim   # concat before first layer

        self.net = nn.Sequential(
            nn.Linear(in_features, hidden_dim), 
            nn.ReLU(),
            nn.Linear(hidden_dim,  hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim,  input_dim),   # output matches input shape
        )

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Parameters:
        x : FloatTensor  [B, input_dim] 
        t : LongTensor   [B]    

        Returns:
        FloatTensor  [B, input_dim] (predicted noise)
        """
        t_emb = self.time_mlp(t)               # [B, time_emb_dim]
        h     = torch.cat([x, t_emb], dim=1)   # [B, input_dim + time_emb_dim]
        return self.net(h)                      # [B, input_dim]  ✓

    def __repr__(self) -> str:
        return (
            f"MLPDenoiser(input_dim={self.input_dim}, "
            f"hidden_dim={self.net[0].out_features}, "
            f"time_emb_dim={self.time_emb_dim})"
        )


if __name__ == "__main__":
    from schedule import T, q_sample, betas, alphas, alphas_cumprod
    from data import GaussianCircleDataset
    import numpy as np

    #building the model
    model = MLPDenoiser(input_dim=2)      #2-D data
    print(model)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters : {total_params:,}")

    #load dataset
    dataset = GaussianCircleDataset()
    x0      = dataset.points                        # [10000, 2]

    #sample a mini-batch and random timesteps
    B   = 64
    idx = torch.randperm(len(dataset))[:B]
    x0_batch = x0[idx]                             # [64, 2]

    t_batch  = torch.randint(0, T, (B,))           # [64] 

    #add noise
    x0_np = x0_batch.numpy()
    xt_np = q_sample(x0_np, t=t_batch[0].item())   # one t for demo
    xt    = torch.from_numpy(xt_np).float()         # [64, 2]

    #forward pass
    model.eval()
    with torch.no_grad():
        eps_pred = model(xt, t_batch)

    # shape check 
    assert eps_pred.shape == xt.shape, \
        f"Mismatch, got {eps_pred.shape}, expected {xt.shape}"

    print(f"\nInput  x  : {xt.shape}")
    print(f"Output eps  : {eps_pred.shape}  same shape as x")

    # trying it one some other input dimensions
    for d in [1, 4, 16, 128]:
        m   = MLPDenoiser(input_dim=d)
        xd  = torch.randn(B, d)
        td  = torch.randint(0, T, (B,))
        out = m(xd, td)
        assert out.shape == xd.shape
        print(f"input_dim={d:4d}  →  output {out.shape}")