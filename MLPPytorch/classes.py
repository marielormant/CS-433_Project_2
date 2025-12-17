# classes.py


# PyTorch imports for dataset and neural network modules
import torch
from torch.utils.data import Dataset
import torch.nn as nn


# Custom dataset class for PyTorch DataLoader
class CustomTorchDataset(Dataset):
    def __init__(self, X, y):
        # Store features and labels as tensors
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        # Return the number of samples
        return len(self.X)

    def __getitem__(self, idx):
        # Retrieve a single sample and its label
        return self.X[idx], self.y[idx]



# Simple Multi-Layer Perceptron (MLP) model
class MLP(nn.Module):
    def __init__(self, input_dim, h1, h2, num_classes):
        super().__init__()
        # Define the network architecture
        self.net = nn.Sequential(
            nn.Linear(input_dim, h1),  # Input to first hidden layer
            nn.ReLU(),
            nn.Linear(h1, h2),         # First to second hidden layer
            nn.ReLU(),
            nn.Linear(h2, num_classes),# Second hidden to output layer
        )

    def forward(self, x):
        # Forward pass through the network
        return self.net(x)
