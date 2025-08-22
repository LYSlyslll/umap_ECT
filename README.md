# DeepECT: A PyTorch Implementation of The Deep Embedded Cluster Tree

This repository contains a PyTorch implementation of the **Deep Embedded Cluster Tree (DeepECT)**, a novel divisive hierarchical embedded clustering method, as proposed in the paper:

> Mautz, D., Plant, C., & Böhm, C. (2020). **DeepECT: The Deep Embedded Cluster Tree**. *Data Science and Engineering, 5*, 419-432. https://doi.org/10.1007/s41019-020-00134-0

DeepECT simultaneously learns a feature representation for a dataset and a hierarchical clustering structure, without requiring the number of clusters to be specified beforehand.

## About This Implementation

This implementation provides a clean, modular, and easy-to-use version of the DeepECT algorithm. It is built entirely in Python using PyTorch.

* **Modular Design:** The code is structured into `TreeNode` and `DeepECT` classes, making it easy to understand and extend.
* **Full Loss Function:** Implements the complete three-part loss function described in the paper:
  1. **Reconstruction Loss (`L_rec`)**: Ensures the embedding preserves essential information from the original data.
  2. **Node Center Loss (`L_nc`)**: Pulls leaf node centers towards the mean of their assigned data points.
  3. **Node Data Compression Loss (`L_dc`)**: A novel projection-based loss that enhances cluster separation while preserving orthogonal structural information.
* **Dynamic Tree Management:** Includes the core logic for dynamically growing the tree by splitting high-variance nodes and pruning "dead" nodes with low weights.
* **Utilities:** Provides convenient methods for training (`train`), prediction (`predict`), and model persistence (`save_model`, `load_model`).
* **Manual Pruning:** An additional utility method `prune_subtree` is included to manually prune the tree at a specified node, allowing for interactive exploration of the hierarchy.

## Key Features of DeepECT

- **Hierarchical Clustering:** Instead of a flat set of clusters, DeepECT builds a binary tree that represents a hierarchy of clusters, capturing relationships between populations and subpopulations.
- **Dynamic Tree Structure:** The tree is not fixed. It grows and prunes itself automatically during training to adapt to the structure of the data in the learned embedding space.
- **No Pre-specified Number of Clusters (k):** Unlike most deep clustering methods (e.g., DEC, IDEC), you don't need to know the number of clusters in advance. You only set a maximum number of leaves as a stopping condition.
- **Simultaneous Learning:** The embedding model (e.g., an Autoencoder) and the cluster tree are optimized jointly, allowing them to improve each other iteratively.

## Citation and License

This work is an implementation of the original paper. If you use this algorithm or code in your research, please cite the original authors:

```bibtex
@article{Mautz2020,
  author    = {Dominik Mautz and Claudia Plant and Christian B{\"{o}}hm},
  title     = {{DeepECT:} The Deep Embedded Cluster Tree},
  journal   = {Data Science and Engineering},
  volume    = {5},
  number    = {4},
  pages     = {419--432},
  year      = {2020},
  url       = {https://doi.org/10.1007/s41019-020-00134-0},
  doi       = {10.1007/s41019-020-00134-0}
}
```

The original paper is distributed under the Creative Commons Attribution 4.0 International License (http://creativecommons.org/licenses/by/4.0/), which permits use, sharing, adaptation, distribution, and reproduction in any medium or format, as long as you give appropriate credit to the original author(s) and the source.

## Requirements

The necessary libraries are listed in `requirements.txt`.

- Python 3.8+
- PyTorch
- scikit-learn
- NumPy
- tqdm

## Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/u2er/DeepECT.git
   cd DeepECT
   ```

2. Install the required packages:

   ```bash
   pip install -r requirements.txt
   ```

## Usage Example

Here is a simple example of how to use the `DeepECT` model on synthetic data.

```python
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.datasets import make_blobs

# Assuming dect.py is in the same directory
from dect import DeepECT

# 1. Define an embedding model (e.g., a simple Autoencoder)
class Autoencoder(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, latent_dim)
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ReLU(),
            nn.Linear(128, input_dim)
        )

    def forward(self, x):
        z = self.encoder(x)
        x_hat = self.decoder(z)
        return z, x_hat

# 2. Prepare Data
# Create synthetic data with clear clusters
X, y = make_blobs(n_samples=1500, centers=4, n_features=20, cluster_std=1.5, random_state=42)
X = torch.tensor(X, dtype=torch.float32)

# Create a DataLoader
dataset = TensorDataset(X)
dataloader = DataLoader(dataset, batch_size=256, shuffle=True)
# For prediction, we use the full dataset
fulldataloader = DataLoader(dataset, batch_size=512)


# 3. Initialize Models
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
INPUT_DIM = X.shape[1]
LATENT_DIM = 5

embedding_model = Autoencoder(input_dim=INPUT_DIM, latent_dim=LATENT_DIM).to(DEVICE)
dect_model = DeepECT(embedding_model=embedding_model, latent_dim=LATENT_DIM, device=DEVICE)

# 4. Train the DeepECT model
print("Starting training...")
dect_model.train(
    dataloader=dataloader,
    iterations=1000,
    lr=1e-3,
    max_leaves=10,          # Stop growing when the tree has 10 leaves
    split_interval=200,     # Check for splits every 200 iterations
    pruning_threshold=0.05, # Prune nodes with weight < 0.05
    split_count_per_growth=2 # Split the 2 best candidate nodes each time
)
print("Training finished.")

# 5. Get Cluster Assignments
print("\nPredicting cluster assignments...")
assignments = dect_model.predict(fulldataloader)
print(f"Data points assigned to {len(torch.unique(assignments))} clusters.")
print(f"First 10 assignments: {assignments[:10]}")

# 6. Save and Load the Model
MODEL_PATH = "dect_model.pth"
print(f"\nSaving model to {MODEL_PATH}...")
dect_model.save_model(MODEL_PATH)

# Create a new instance to load the model into
new_embedding_model = Autoencoder(input_dim=INPUT_DIM, latent_dim=LATENT_DIM).to(DEVICE)
loaded_dect_model = DeepECT(embedding_model=new_embedding_model, latent_dim=LATENT_DIM, device=DEVICE)

print(f"Loading model from {MODEL_PATH}...")
# The load_model method is not defined in the provided code snippet.
# Assuming it should be:
# loaded_dect_model.load_state_dict(torch.load(MODEL_PATH))
# Based on your implementation, it is:
loaded_dect_model.load_model(MODEL_PATH)

# Verify by predicting again
loaded_assignments = loaded_dect_model.predict(fulldataloader)
assert torch.equal(assignments, loaded_assignments)
print("Model loaded successfully and predictions match!")

```
