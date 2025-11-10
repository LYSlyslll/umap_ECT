import json
from pathlib import Path
from typing import Any, List, Tuple

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, Subset
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt

# Assuming dect.py is in the same directory
from dect import DeepECT

# 1. Define an embedding model (e.g., a simple Autoencoder)
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

# 2. Prepare Data
# Read embeddings from a JSONL file where each line is a JSON object with
# fields: "idx", "phrase", and "embedding" (a list of 768、1536 floats).


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
                if line_number==10000:
                    break

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


JSONL_PATH = Path("/home/liangyushan/DeepECT/concept_embeddings.jsonl")
if not JSONL_PATH.exists():
    print(f"Creating sample dataset at {JSONL_PATH}...")
    create_sample_jsonl(JSONL_PATH)

dataset = JsonlEmbeddingDataset(JSONL_PATH)

dataloader = DataLoader(dataset, batch_size=256, shuffle=False, collate_fn=jsonl_collate_fn)

# For prediction, we use the full dataset without shuffling to preserve order
fulldataloader = DataLoader(dataset, batch_size=512, shuffle=False, collate_fn=jsonl_collate_fn)


# 3. Initialize Models
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
INPUT_DIM = dataset.embedding_dim
LATENT_DIM = 128
print(f"INPUT_DIM={INPUT_DIM}")

base_embedding_model = Autoencoder(input_dim=INPUT_DIM, latent_dim=LATENT_DIM)
base_embedding_model = base_embedding_model.to(DEVICE)
if torch.cuda.device_count() > 1:
    print(f"Using {torch.cuda.device_count()} GPUs for training.")
    embedding_model = nn.DataParallel(base_embedding_model)
else:
    embedding_model = base_embedding_model

dect_model = DeepECT(embedding_model=embedding_model, latent_dim=LATENT_DIM, device=DEVICE)

# 4. Train the DeepECT model
print("Starting training...")
training_history = dect_model.train(
    dataloader=dataloader,
    iterations=500,
    lr=1e-3,
    max_leaves=3000,          # Stop growing when the tree has 10 leaves
    split_interval=2,     # Check for splits every 200 iterations
    pruning_threshold=0.05, # Prune nodes with weight < 0.05
    split_count_per_growth=5, # Split the 2 best candidate nodes each time
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

# 5. Get Cluster Assignments
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

PREDICTIONS_PATH = Path("./predicted_clusters.jsonl")
print(f"Saving predictions to {PREDICTIONS_PATH}...")
PREDICTIONS_PATH.parent.mkdir(parents=True, exist_ok=True)
with PREDICTIONS_PATH.open('w', encoding='utf-8') as f:
    for sample_idx, cluster_id in zip(indices, assignments.tolist()):
        record = {"idx": sample_idx, "cluster": int(cluster_id)}
        f.write(json.dumps(record, ensure_ascii=False) + '\n')
print("Predictions saved.")

# 6. Save and Load the Model
MODEL_PATH = "dect_model.pth"
print(f"\nSaving model to {MODEL_PATH}...")
dect_model.save_model(MODEL_PATH)

# Create a new instance to load the model into
new_base_model = Autoencoder(input_dim=INPUT_DIM, latent_dim=LATENT_DIM)
new_base_model = new_base_model.to(DEVICE)
if torch.cuda.device_count() > 1:
    loaded_embedding_model = nn.DataParallel(new_base_model)
else:
    loaded_embedding_model = new_base_model

loaded_dect_model = DeepECT(embedding_model=loaded_embedding_model, latent_dim=LATENT_DIM, device=DEVICE)

print(f"Loading model from {MODEL_PATH}...")
# The load_model method is not defined in the provided code snippet.
# Assuming it should be:
# loaded_dect_model.load_state_dict(torch.load(MODEL_PATH))
# Based on your implementation, it is:
loaded_dect_model.load_model(MODEL_PATH)

# Verify by predicting again
loaded_result = loaded_dect_model.predict(fulldataloader)
assert torch.equal(assignments, loaded_result["assignments"].cpu())
if indices:
    assert indices == loaded_result.get("idx", [])
print("Model loaded successfully and predictions match!")
