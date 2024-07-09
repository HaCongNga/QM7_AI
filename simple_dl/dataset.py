import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, Subset

# Create a custom Dataset class
class QM7Dataset(Dataset):
    def __init__(self, X, Z=None, R=None, T=None):
        self.X = X # Coulumb matrix
        self.T = T # labels
        self.Z = Z # atomic charges
        self.R = R # Cartesian coordinate

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.Z[idx], self.R[idx], self.T[idx]
