import sys
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from dataset import QM7Dataset
from model import SimpleNet

import scipy.io

# Load the dataset
dataset = scipy.io.loadmat('qm7.mat')
X = dataset['X']
T = dataset['T'][0]
P = dataset['P']
Z = dataset['Z']
R = dataset['R']
num_splits = 5

# Convert to PyTorch tensors
X_tensor = torch.tensor(X, dtype=torch.float32)
T_tensor = torch.tensor(T, dtype=torch.float32)
Z_tensor = torch.tensor(Z, dtype=torch.float32)
R_tensor = torch.tensor(R, dtype=torch.float32)

# Normalize the target values
T_mean = T_tensor.mean()
T_std = T_tensor.std()
T_tensor = (T_tensor - T_mean) / T_std

dataset = QM7Dataset(X_tensor, Z_tensor, R_tensor, T_tensor)

# Instantiate the model
model = SimpleNet()

# Define the loss function and optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
def train_model(split, epochs = 1000):
    num_splits = 5
    train_indices = P[list(range(0,split)) + list(range(split+1, num_splits))].astype(int)
    train_dataset = Subset(dataset, train_indices)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

    num_epochs = epochs
    model.train()

    loss_hist = []
    for epoch in range(num_epochs):
        running_loss = 0.0
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels.unsqueeze(1))
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            loss_hist.append(running_loss)
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {running_loss / len(train_loader):.4f}')
        with open(f'loss_info.txt', 'a') as f:
            for loss in loss_hist:
                f.write(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {running_loss / len(train_loader):.4f}\n')

    # Save the model
    torch.save(model.state_dict(), f'model_split_{split}.pth')

    # Save the loss history


def evaluate_model(split, epochs):
    model.load_state_dict(torch.load(f'model_split_{split}.pth')) # model_split_0 has been trained on 1,2,3,4
    train_indices = P[list(range(0, split)) + list(range(split+1, num_splits))].astype(int)
    test_indices = np.setdiff1d(np.arange(len(dataset)), train_indices)

    train_dataset = Subset(dataset, train_indices)
    test_dataset = Subset(dataset, test_indices)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    model.eval()

    train_mae, train_rmse = 0.0, 0.0
    test_mae, test_rmse = 0.0, 0.0

    with torch.no_grad():
        for inputs, labels in train_loader:
            outputs = model(inputs)
            train_mae += torch.abs(outputs - labels.unsqueeze(1)).sum().item()
            train_rmse += torch.pow(outputs - labels.unsqueeze(1), 2).sum().item()

        for inputs, labels in test_loader:
            outputs = model(inputs)
            test_mae += torch.abs(outputs - labels.unsqueeze(1)).sum().item()
            test_rmse += torch.pow(outputs - labels.unsqueeze(1), 2).sum().item()

    train_mae /= len(train_dataset)
    train_rmse = (train_rmse / len(train_dataset)) ** 0.5
    test_mae /= len(test_dataset)
    test_rmse = (test_rmse / len(test_dataset)) ** 0.5

    print(f'Train MAE: {train_mae * T_std:.4f}, Train RMSE: {train_rmse * T_std:.4f}')
    print(f'Test MAE: {test_mae * T_std:.4f}, Test RMSE: {test_rmse * T_std:.4f}')

    with open(f'Error_info.txt', 'a') as f:
        f.write(f'Num_epochs {epochs}] Split {split}\n
                Train MAE: {train_mae * T_std:.4f}, Train RMSE: {train_rmse * T_std:.4f} \n
                Test MAE: {test_mae * T_std:.4f}, Test RMSE: {test_rmse * T_std:.4f}\n')

if __name__ == "__main__":
    # args 1 : train, test
    #### args 2 : split
    # args 2 : num_epochs

    run_type = str(sys.argv[1])
    epochs = int(sys.argv[2])
    
    for i in range(num_splits) :
        if run_type == "train" :
            split = i
            train_model(split, epochs)
        elif run_type == "test" :
            split = i
            evaluate_model(split, epochs)
        else :
            raise ValueError("Unsupported run. Please choose train or test")