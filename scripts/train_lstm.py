"""
Module 6: LSTM Orbit Prediction

This module builds, trains, evaluates, and saves an LSTM neural network
using PyTorch to predict future orbital trajectories based on past sequences.
"""

import os
import numpy as np
import logging
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


def load_processed_data(data_dir: str) -> tuple:
    logging.info("Loading preprocessed sequence data...")
    X = np.load(os.path.join(data_dir, "X.npy"))
    y = np.load(os.path.join(data_dir, "y.npy"))
    logging.info(f"Loaded X shape: {X.shape}")
    logging.info(f"Loaded y shape: {y.shape}")
    return X, y


class OrbitLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, num_layers=2, output_dim=7):
        super(OrbitLSTM, self).__init__()
        self.lstm = nn.LSTM(
            input_dim, hidden_dim, num_layers, batch_first=True, dropout=0.2
        )
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        out, _ = self.lstm(x)
        # We want the output from the last time step
        out = self.fc(out[:, -1, :])
        return out


def plot_training_history(train_losses, val_losses, save_path: str) -> None:
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label="Training Loss (MSE)")
    plt.plot(val_losses, label="Validation Loss (MSE)")
    plt.title("LSTM Training History")
    plt.xlabel("Epochs")
    plt.ylabel("Mean Squared Error Loss")
    plt.legend()
    plt.grid(True)
    plt.savefig(save_path)
    logging.info(f"Training history plot saved to {save_path}")


def train_and_evaluate() -> None:
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)
    data_dir = os.path.join(project_root, "data", "processed")
    models_dir = os.path.join(project_root, "models")
    vis_dir = os.path.join(project_root, "visualization")

    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(vis_dir, exist_ok=True)

    # 1. Load Data
    X, y = load_processed_data(data_dir)

    # 2. Split train/test data (80% train, 20% test)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Convert to PyTorch tensors
    X_train_t = torch.tensor(X_train, dtype=torch.float32)
    y_train_t = torch.tensor(y_train, dtype=torch.float32)
    X_test_t = torch.tensor(X_test, dtype=torch.float32)
    y_test_t = torch.tensor(y_test, dtype=torch.float32)

    train_dataset = TensorDataset(X_train_t, y_train_t)
    test_dataset = TensorDataset(X_test_t, y_test_t)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    # 3. Build LSTM
    features = X_train.shape[2]
    model = OrbitLSTM(input_dim=features, output_dim=features)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # 4. Train Model
    epochs = 10
    train_losses = []
    val_losses = []

    logging.info("Starting model training...")
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * batch_X.size(0)

        epoch_loss /= len(train_loader.dataset)
        train_losses.append(epoch_loss)

        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch_X, batch_y in test_loader:
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                val_loss += loss.item() * batch_X.size(0)
        val_loss /= len(test_loader.dataset)
        val_losses.append(val_loss)

        logging.info(
            f"Epoch [{epoch + 1}/{epochs}] - "
            f"Train Loss: {epoch_loss:.4f}, "
            f"Val Loss: {val_loss:.4f}"
        )

    # 5. Evaluate Performance
    logging.info(f"Final Test Loss (MSE): {val_losses[-1]:.4f}")

    # 6. Plot Training History
    plot_path = os.path.join(vis_dir, "lstm_training_history.png")
    plot_training_history(train_losses, val_losses, plot_path)

    # 7. Save the Model
    model_save_path = os.path.join(models_dir, "lstm_orbit_model.pth")
    torch.save(model.state_dict(), model_save_path)
    logging.info(f"Trained PyTorch LSTM model saved to {model_save_path}")


if __name__ == "__main__":
    logging.info("--- Starting LSTM Orbit Prediction Module ---")
    train_and_evaluate()
    logging.info("--- Module Completed ---")
