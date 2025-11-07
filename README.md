# DeepECT: A PyTorch Implementation of The Deep Embedded Cluster Tree

This repository contains a PyTorch implementation of the **Deep Embedded Cluster Tree (DeepECT)**, a novel divisive hierarchical embedded clustering method, as proposed in the paper:

> Mautz, D., Plant, C., & Böhm, C. (2020). **DeepECT: The Deep Embedded Cluster Tree**. *Data Science and Engineering, 5*, 419-432. https://doi.org/10.1007/s41019-020-00134-0

DeepECT simultaneously learns a feature representation for a dataset and a hierarchical clustering structure, without requiring the number of clusters to be specified beforehand.

## About This Implementation

This implementation provides a clean, modular, and easy-to-use version of the DeepECT algorithm. It is built entirely in Python using PyTorch.

* **Modular Design:** The code is structured into `TreeNode` and `DeepECT` classes, making it easy to understand and extend.
* **Cosine-distance Losses:** Implements the complete three-part loss function described in the paper using cosine-distance met
rics:
  1. **Reconstruction Loss (`L_rec`)**: Ensures the embedding preserves essential information from the original data.
  2. **Node Center Loss (`L_nc`)**: Pulls leaf node centers towards the mean of their assigned data points.
  3. **Node Data Compression Loss (`L_dc`)**: A novel projection-based loss that enhances cluster separation while preserving orthogonal structural information.
* **Dynamic Tree Management:** Includes the core logic for dynamically growing the tree by splitting high-variance nodes and pruning "dead" nodes with low weights.
* **Utilities:** Provides convenient methods for training (`train`), prediction (`predict`), and model persistence (`save_model`, `load_model`).
* **Manual Pruning:** An additional utility method `prune_subtree` is included to manually prune the tree at a specified node, allowing for interactive exploration of the hierarchy.
* **Training Quality-of-life:** Built-in support for Adam with step-wise learning-rate decay, transparent loss history logging, and optional multi-GPU (`nn.DataParallel`) execution.

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
- matplotlib

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
import json
from pathlib import Path
from typing import Any, List, Tuple

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.datasets import make_blobs

# Assuming dect.py is in the same directory
from dect import DeepECT


class Autoencoder(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Linear(512, 256),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Linear(256, latent_dim)
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Linear(256, 512),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Linear(512, input_dim)
        )

    def forward(self, x):
        z = self.encoder(x)
        x_hat = self.decoder(z)
        return z, x_hat


def create_sample_jsonl(path: Path, num_samples: int = 200, embedding_dim: int = 768) -> None:
    """Creates a small synthetic JSONL dataset for demonstration purposes."""
    data, labels = make_blobs(
        n_samples=num_samples,
        centers=4,
        n_features=embedding_dim,
        cluster_std=1.5,
        random_state=42,
    )

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8') as f:
        for idx, (embedding, label) in enumerate(zip(data, labels)):
            record = {
                "idx": idx,
                "错误类型": int(label),
                "embedding": embedding.astype(float).tolist(),
            }
            f.write(json.dumps(record, ensure_ascii=False) + '\n')


class JsonlEmbeddingDataset(Dataset):
    def __init__(self, jsonl_path: Path):
        self.jsonl_path = Path(jsonl_path)
        self.indices: List[Any] = []
        self.error_types: List[Any] = []
        self.embeddings: List[List[float]] = []

        with self.jsonl_path.open('r', encoding='utf-8') as f:
            for line_number, line in enumerate(f, start=1):
                print(f"正在读取第{line_number}行数据")
                line = line.strip()
                if not line:
                    continue
                record = json.loads(line)
                self.indices.append(record["idx"])
                self.error_types.append(record.get("错误类型"))
                self.embeddings.append(record["embedding"])

        if not self.embeddings:
            raise ValueError(f"No embeddings found in {self.jsonl_path}")

        self.embedding_dim = len(self.embeddings[0])

    def __len__(self) -> int:
        return len(self.embeddings)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, Any, Any]:
        embedding = torch.tensor(self.embeddings[idx], dtype=torch.float32)
        return embedding, self.indices[idx], self.error_types[idx]


def jsonl_collate_fn(batch):
    embeddings = torch.stack([item[0] for item in batch], dim=0)
    indices = [item[1] for item in batch]
    error_types = [item[2] for item in batch]
    return embeddings, indices, error_types


JSONL_PATH = Path("usage_examples/sample_embeddings.jsonl")
if not JSONL_PATH.exists():
    print(f"Creating sample dataset at {JSONL_PATH}...")
    create_sample_jsonl(JSONL_PATH)

dataset = JsonlEmbeddingDataset(JSONL_PATH)
dataloader = DataLoader(dataset, batch_size=256, shuffle=True, collate_fn=jsonl_collate_fn)
fulldataloader = DataLoader(dataset, batch_size=512, shuffle=False, collate_fn=jsonl_collate_fn)


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
INPUT_DIM = dataset.embedding_dim
LATENT_DIM = 5

base_embedding_model = Autoencoder(input_dim=INPUT_DIM, latent_dim=LATENT_DIM)
base_embedding_model = base_embedding_model.to(DEVICE)
if torch.cuda.device_count() > 1:
    print(f"Using {torch.cuda.device_count()} GPUs for training.")
    embedding_model = nn.DataParallel(base_embedding_model)
else:
    embedding_model = base_embedding_model

dect_model = DeepECT(embedding_model=embedding_model, latent_dim=LATENT_DIM, device=DEVICE)


print("Starting training...")
training_history = dect_model.train(
    dataloader=dataloader,
    iterations=1000,
    lr=1e-3,
    max_leaves=10,
    split_interval=200,
    pruning_threshold=0.05,
    split_count_per_growth=2,
    lr_decay_step=100,
    lr_decay_gamma=0.95
)
print("Training finished.")

LOSS_PLOT_PATH = Path("usage_examples/training_loss.png")
LOSS_PLOT_PATH.parent.mkdir(parents=True, exist_ok=True)
plt.figure(figsize=(10, 6))
plt.plot(training_history['total'], label='Total Loss')
plt.plot(training_history['reconstruction'], label='Reconstruction Loss')
plt.plot(training_history['node_center'], label='Node Center Loss')
plt.plot(training_history['node_compression'], label='Node Compression Loss')
plt.xlabel('Iteration')
plt.ylabel('Cosine Distance Loss')
plt.title('DeepECT Training Loss Curve')
plt.legend()
plt.tight_layout()
plt.savefig(LOSS_PLOT_PATH, dpi=300)
plt.close()
print(f"Training loss curve saved to {LOSS_PLOT_PATH}.")


print("\nPredicting cluster assignments...")
prediction_result = dect_model.predict(fulldataloader)
assignments = prediction_result["assignments"].cpu()
indices = prediction_result.get("idx", [])
print(f"Data points assigned to {len(torch.unique(assignments))} clusters.")
if indices:
    paired = list(zip(indices, assignments.tolist()))
    print(f"First 5 assignments with indices: {paired[:5]}")
else:
    print(f"First 10 assignments: {assignments[:10]}")

PREDICTIONS_PATH = Path("usage_examples/predicted_clusters.jsonl")
print(f"Saving predictions to {PREDICTIONS_PATH}...")
PREDICTIONS_PATH.parent.mkdir(parents=True, exist_ok=True)
with PREDICTIONS_PATH.open('w', encoding='utf-8') as f:
    for sample_idx, cluster_id in zip(indices, assignments.tolist()):
        record = {"idx": sample_idx, "cluster": int(cluster_id)}
        f.write(json.dumps(record, ensure_ascii=False) + '\n')
print("Predictions saved.")


MODEL_PATH = "dect_model.pth"
print(f"\nSaving model to {MODEL_PATH}...")
dect_model.save_model(MODEL_PATH)

new_base_model = Autoencoder(input_dim=INPUT_DIM, latent_dim=LATENT_DIM)
new_base_model = new_base_model.to(DEVICE)
if torch.cuda.device_count() > 1:
    loaded_embedding_model = nn.DataParallel(new_base_model)
else:
    loaded_embedding_model = new_base_model

loaded_dect_model = DeepECT(embedding_model=loaded_embedding_model, latent_dim=LATENT_DIM, device=DEVICE)

print(f"Loading model from {MODEL_PATH}...")
loaded_dect_model.load_model(MODEL_PATH)

loaded_result = loaded_dect_model.predict(fulldataloader)
assert torch.equal(assignments, loaded_result["assignments"].cpu())
if indices:
    assert indices == loaded_result.get("idx", [])
print("Model loaded successfully and predictions match!")

```
