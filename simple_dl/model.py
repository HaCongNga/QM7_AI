import sys
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset

class SimpleNet(nn.Module):
    def __init__(self):
        super(SimpleNet, self).__init__()
        self.fc1 = nn.Linear(23 * 23 + 1 + 3, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 1)

    def flatten_sort(x) :
        if len(x.shape) > 1 :
            x = x.view(x.size(0), -1)  # Flatten the input
        x = torch.sort(x)
        x = x[0]
        return x
    
    def forward(self, x, z, r):
        x = self.flatten_sort(x)
        z = self.flatten_sort(z)
        r = self.flatten_sort(r)
        x = torch.cat((x,z,r), dim=1)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    
