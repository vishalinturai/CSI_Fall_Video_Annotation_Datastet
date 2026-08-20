# python train_lstm_autoencoder.py --input algorithm1_dataset.npz --window 25 --epochs 50 --batch-size 64
"""
LSTM Autoencoder for CSI-based Fall Detection
=============================================
Trains on non-fall sequences, finds the optimal anomaly threshold,
and evaluates reconstruction performance on fall sequences.

Usage:
    python train_lstm_autoencoder.py --input algorithm1_dataset.npz --epochs 50 --batch-size 64
"""

import argparse
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


# ----------------------------------------------------------------------------
# 1. Dataset Preparation & Windowing
# ----------------------------------------------------------------------------
def create_sliding_windows(data: np.ndarray, seq_len: int, stride: int = 1):
    """
    Slices a continuous 2D array (N_samples, N_features) into
    overlapping 3D sequences (N_windows, seq_len, N_features).
    """
    windows = []
    L = len(data)
    for start in range(0, L - seq_len + 1, stride):
        windows.append(data[start: start + seq_len])
    if not windows:
        return np.empty((0, seq_len, data.shape[1]), dtype=np.float32)
    return np.array(windows, dtype=np.float32)


def load_and_preprocess_data(npz_path: str, seq_len: int = 25, val_split: float = 0.2):
    """
    Loads algorithm1_dataset.npz, fits a scaler exclusively on non-fall data,
    and returns PyTorch DataLoader objects.
    """
    data = np.load(npz_path)
    nofall_raw = data["nofall_data"]  # Shape: (N_nofall, N_features)
    fall_raw = data["fall_data"]  # Shape: (N_fall, N_features)

    print(f"[Data] Loaded {len(nofall_raw)} non-fall samples and {len(fall_raw)} fall samples.")

    # 1. Fit scaler ONLY on non-fall (train distribution) to prevent data leakage
    scaler = StandardScaler()
    nofall_scaled = scaler.fit_transform(nofall_raw)
    fall_scaled = scaler.transform(fall_raw)

    # 2. Slice into sequences of length `seq_len`
    nofall_windows = create_sliding_windows(nofall_scaled, seq_len=seq_len, stride=max(1, seq_len // 2))
    fall_windows = create_sliding_windows(fall_scaled, seq_len=seq_len, stride=max(1, seq_len // 4))

    print(f"[Windows] Created {len(nofall_windows)} non-fall windows and {len(fall_windows)} fall windows.")

    # 3. Train/Validation Split for Normal Data
    n_val = int(len(nofall_windows) * val_split)
    indices = np.random.permutation(len(nofall_windows))

    train_idx, val_idx = indices[n_val:], indices[:n_val]
    train_windows = nofall_windows[train_idx]
    val_windows = nofall_windows[val_idx]

    return train_windows, val_windows, fall_windows, scaler


# ----------------------------------------------------------------------------
# 2. LSTM Autoencoder Architecture
# ----------------------------------------------------------------------------
class LSTMEncoder(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, latent_dim: int, num_layers: int = 2, dropout: float = 0.2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0
        )
        self.fc = nn.Linear(hidden_dim, latent_dim)

    def forward(self, x):
        # x: (batch_size, seq_len, input_dim)
        lstm_out, (h_n, _) = self.lstm(x)
        # Latent representation from the last time-step hidden state
        latent = self.fc(h_n[-1])  # (batch_size, latent_dim)
        return latent


class LSTMDecoder(nn.Module):
    def __init__(self, latent_dim: int, hidden_dim: int, output_dim: int, seq_len: int, num_layers: int = 2,
                 dropout: float = 0.2):
        super().__init__()
        self.seq_len = seq_len
        self.fc_latent = nn.Linear(latent_dim, hidden_dim)
        self.lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0
        )
        self.fc_out = nn.Linear(hidden_dim, output_dim)

    def forward(self, latent):
        # latent: (batch_size, latent_dim)
        hidden_rep = self.fc_latent(latent)  # (batch_size, hidden_dim)
        # Repeat vector along time dimension: (batch_size, seq_len, hidden_dim)
        repeated = hidden_rep.unsqueeze(1).repeat(1, self.seq_len, 1)
        lstm_out, _ = self.lstm(repeated)
        recon = self.fc_out(lstm_out)  # (batch_size, seq_len, output_dim)
        return recon


class LSTMAutoencoder(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 64, latent_dim: int = 16, seq_len: int = 25,
                 num_layers: int = 2, dropout: float = 0.2):
        super().__init__()
        self.encoder = LSTMEncoder(input_dim, hidden_dim, latent_dim, num_layers, dropout)
        self.decoder = LSTMDecoder(latent_dim, hidden_dim, input_dim, seq_len, num_layers, dropout)

    def forward(self, x):
        latent = self.encoder(x)
        reconstruction = self.decoder(latent)
        return reconstruction


# ----------------------------------------------------------------------------
# 3. Model Training Function
# ----------------------------------------------------------------------------
def train_model(model, train_loader, val_loader, epochs: int, lr: float, device: torch.device, model_save_path: str):
    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

    best_val_loss = float("inf")

    print("\n--- Starting Training ---")
    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        for (batch_x,) in train_loader:
            batch_x = batch_x.to(device)

            optimizer.zero_grad()
            recon = model(batch_x)
            loss = criterion(recon, batch_x)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * batch_x.size(0)

        train_loss /= len(train_loader.dataset)

        # Validation phase
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for (batch_x,) in val_loader:
                batch_x = batch_x.to(device)
                recon = model(batch_x)
                loss = criterion(recon, batch_x)
                val_loss += loss.item() * batch_x.size(0)

        val_loss /= len(val_loader.dataset)
        scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), model_save_path)
            saved_marker = "[Model Saved]"
        else:
            saved_marker = ""

        if epoch % 5 == 0 or epoch == 1 or saved_marker:
            print(
                f"Epoch [{epoch:03d}/{epochs:03d}] | Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f} {saved_marker}")

    print(f"Training Complete. Best Validation Loss: {best_val_loss:.6f}\n")


# ----------------------------------------------------------------------------
# 4. Evaluation & Anomaly Thresholding
# ----------------------------------------------------------------------------
def compute_reconstruction_errors(model, data_windows, device):
    """Computes MSE per window between input sequence and reconstructed sequence."""
    model.eval()
    dataset = TensorDataset(torch.tensor(data_windows, dtype=torch.float32))
    loader = DataLoader(dataset, batch_size=64, shuffle=False)

    errors = []
    criterion = nn.MSELoss(reduction='none')

    with torch.no_grad():
        for (batch_x,) in loader:
            batch_x = batch_x.to(device)
            recon = model(batch_x)
            # MSE across feature & time dimensions: (batch_size,)
            mse = criterion(recon, batch_x).mean(dim=[1, 2]).cpu().numpy()
            errors.extend(mse)

    return np.array(errors)


def evaluate_and_plot(val_errors, fall_errors, threshold=None):
    # If no explicit threshold provided, use: Mean(Val_Loss) + 3 * Std(Val_Loss)
    if threshold is None:
        threshold = np.mean(val_errors) + 3 * np.std(val_errors)

    print(f"--- Fall Detection Anomaly Threshold: {threshold:.6f} ---")

    y_true = np.array([0] * len(val_errors) + [1] * len(fall_errors))
    y_scores = np.concatenate([val_errors, fall_errors])
    y_pred = (y_scores > threshold).astype(int)

    print("\nConfusion Matrix:")
    print(confusion_matrix(y_true, y_pred))

    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=["Non-Fall (Normal)", "Fall (Anomaly)"]))
    print(f"ROC-AUC Score: {roc_auc_score(y_true, y_scores):.4f}")

    # Plot Reconstruction Error Distributions
    plt.figure(figsize=(9, 4.5))
    plt.hist(val_errors, bins=40, alpha=0.6, color='blue', label='Non-Fall (Validation)', density=True)
    plt.hist(fall_errors, bins=40, alpha=0.6, color='red', label='Fall', density=True)
    plt.axvline(threshold, color='black', linestyle='--', linewidth=2, label=f'Threshold ({threshold:.4f})')
    plt.title('CSI Reconstruction Error Distribution')
    plt.xlabel('Reconstruction Error (MSE)')
    plt.ylabel('Density')
    plt.legend()
    plt.tight_layout()
    plt.savefig("reconstruction_error_distribution.png", dpi=300)
    print("Saved plot -> reconstruction_error_distribution.png")


# ----------------------------------------------------------------------------
# Main Execution
# ----------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Train LSTM Autoencoder for CSI Fall Detection")
    parser.add_argument("--input", default="algorithm1_dataset.npz", help="Path to algorithm1_dataset.npz")
    parser.add_argument("--window", type=int, default=25, help="Sequence window length (must match extraction window)")
    parser.add_argument("--hidden-dim", type=int, default=64, help="LSTM hidden layer dimension")
    parser.add_argument("--latent-dim", type=int, default=16, help="Latent bottleneck vector dimension")
    parser.add_argument("--num-layers", type=int, default=2, help="Number of stacked LSTM layers")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size for training")
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=1e-3, help="Initial learning rate")
    parser.add_argument("--output-model", default="csi_lstm_autoencoder.pth", help="Path to save best PyTorch weights")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using compute device: {device}")

    # 1. Load Data
    train_win, val_win, fall_win, _ = load_and_preprocess_data(args.input, seq_len=args.window)

    train_loader = DataLoader(TensorDataset(torch.tensor(train_win, dtype=torch.float32)), batch_size=args.batch_size,
                              shuffle=True)
    val_loader = DataLoader(TensorDataset(torch.tensor(val_win, dtype=torch.float32)), batch_size=args.batch_size,
                            shuffle=False)

    input_dim = train_win.shape[2]  # Subcarrier feature count

    # 2. Build Model
    model = LSTMAutoencoder(
        input_dim=input_dim,
        hidden_dim=args.hidden_dim,
        latent_dim=args.latent_dim,
        seq_len=args.window,
        num_layers=args.num_layers,
        dropout=0.2
    ).to(device)

    # 3. Train
    train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=args.epochs,
        lr=args.lr,
        device=device,
        model_save_path=args.output_model
    )

    # 4. Load Best Weights & Evaluate
    model.load_state_dict(torch.load(args.output_model))
    val_errors = compute_reconstruction_errors(model, val_win, device)
    fall_errors = compute_reconstruction_errors(model, fall_win, device)

    evaluate_and_plot(val_errors, fall_errors)


if __name__ == "__main__":
    main()