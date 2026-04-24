from __future__ import annotations

import os
import random
import numpy as np
import torch

def set_seed(seed: int = 42) -> None:

    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")

    if torch.backends.mps.is_available():
        return torch.device("mps")

    return torch.device("cpu")


def to_device(x, device: torch.device):

    return x.to(device)

def sample_noise_like(x: torch.Tensor) -> torch.Tensor:

    return torch.randn_like(x)

def sample_timesteps(
    batch_size: int,
    T: int,
    device: torch.device | None = None,
) -> torch.Tensor:

    t = torch.randint(
        low=0,
        high=T,
        size=(batch_size,),
        dtype=torch.long,
    )

    if device is not None:
        t = t.to(device)

    return t


def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    path: str,
) -> None:

    os.makedirs(os.path.dirname(path), exist_ok=True)

    checkpoint = {
        "epoch": epoch,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "input_dim": getattr(model, "input_dim", None),
    }

    torch.save(checkpoint, path)

    print(f"Saved checkpoint → {path}")


def load_checkpoint(
    path: str,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    map_location: str = "cpu",
) -> int:
    checkpoint = torch.load(
        path,
        map_location=map_location,
    )

    model.load_state_dict(checkpoint["model_state"])

    if optimizer is not None:
        optimizer.load_state_dict(
            checkpoint["optimizer_state"]
        )

    epoch = checkpoint.get("epoch", 0)

    print(f"Loaded checkpoint ← {path}")

    return epoch

def count_parameters(model: torch.nn.Module) -> int:

    return sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )

def ensure_dir(path: str) -> None:

    os.makedirs(path, exist_ok=True)