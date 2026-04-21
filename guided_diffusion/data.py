import math
import torch 
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt

class GaussianCircleDataset(Dataset):
    def __init__(self, n_samples: int = 10_000, n_clusters: int = 8, radius: float = 2.0, std: float = 0.1, seed: int = 42):
      super().__init__()
      self.n_samples = n_samples
      self.n_clusters = n_clusters
      self.radius = radius 
      self.std = std

      random_number_generator = torch.Generator()
      random_number_generator.manual_seed(seed)
      
      #to initialize cluster centers, creating n evenly spaced angles, using linspace to create a 1-D tensor, that creates even space from 0 to 2pi
      angles = torch.linspace(0, 2 * math.pi, n_clusters + 1)[:-1] #-1 to remove 2pi
      #generating polar coordinate for centres, (rcosθ and rsinθ)
      self.centres = torch.stack(
          [radius*torch.cos(angles), radius*torch.sin(angles)], dim=1
      )
      
      #assigning samples to clusters (uniformly)
      labels = torch.randint(0, n_clusters, (n_samples,), generator=random_number_generator)
      
      #sampling 2-D Gaussian noise around each cenetre
      noise = torch.randn(n_samples, 2, generator=random_number_generator)*std
      points = self.centres[labels] + noise
      
      self.points: torch.Tensor = points
      self.labels: torch.Tensor = labels
    
    def __len__(self) -> int: 
      return self.n_samples
    
    def __getitem__(self, idx):
      return self.points[idx], self.labels[idx]
    
    def __repr__(self) -> str:
      return (
            f"n_samples={self.n_samples}, "
            f"n_clusters={self.n_clusters}, "
            f"radius={self.radius}, "
            f"std={self.std})"
      )

if __name__ == "__main__": 
    dataset = GaussianCircleDataset()
    print(dataset)
    print({dataset.points.shape})
    print({dataset.labels.shape})

    loader = DataLoader(dataset, batch_size=256, shuffle=True)
    batch_points, batch_label = next(iter(loader))
    print({batch_points.shape}, {batch_label.shape})

    point = dataset.points.numpy()
    label = dataset.labels.numpy()

    fig, ax = plt.subplots(figsize=(6, 6))
    scatter = ax.scatter(point[:, 0], point[:, 1], c=label, cmap="tab10", s=3, alpha=0.5, linewidth=0)

    cx, cy = dataset.centres[:, 0].numpy(), dataset.centres[:, 1].numpy()
    ax.scatter(cx, cy, marker="x", s=120, c="black", linewidths=2, zorder=5)

    ax.set_aspect("equal")
    ax.set_title("8 Gaussian clusters on a circle  (radius≈2, std≈0.1, N=10 000)")
    ax.set_xlabel("x"); ax.set_ylabel("y")
    plt.tight_layout()
    plt.savefig("/Users/ameygupta/clusters.png", dpi=150)
    plt.show()
    print("Saved")