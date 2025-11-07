import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
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


# Example of Dataset for DECT
class ClusteringDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return self.data.shape[0]
    
    def __getitem__(self, idx):
        return self.data[idx]


# Create a DataLoader
dataset = ClusteringDataset(X)
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
