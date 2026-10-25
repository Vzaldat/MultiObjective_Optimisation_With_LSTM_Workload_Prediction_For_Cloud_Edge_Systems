import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import TimeSeriesSplit
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import optuna
import json, joblib

# ================================================
#  1.  GROUPED WINDOWS
# ================================================

FEATURES = ['cpu_util', 'avg_cpi', 'avg_mpki']
def prepare_single_split(csv_path, seq_length=12):
    df = pd.read_csv(csv_path)
    features = ['cpu_util', 'avg_cpi', 'avg_mpki']
    
    raw_matrix = df[features].values.astype(np.float32)
    total_rows = len(raw_matrix)
    
    # Clean 70% / 30% simple chronological split
    train_end = int(total_rows * 0.70)
    
    train_raw = raw_matrix[:train_end]
    val_raw = raw_matrix[train_end:]
    
    # Scale features carefully
    scaler = MinMaxScaler()
    train_scaled = scaler.fit_transform(train_raw)
    val_scaled = scaler.transform(val_raw)
    
    # Vectorized Sliding Window view
    X_tr_win = np.lib.stride_tricks.sliding_window_view(train_scaled[:-1, :], (seq_length, 3))
    X_train = torch.tensor(X_tr_win.squeeze(axis=1), dtype=torch.float32)
    y_train = torch.tensor(train_scaled[seq_length:, 0:1], dtype=torch.float32) # Target is cpu_util
    
    X_val_win = np.lib.stride_tricks.sliding_window_view(val_scaled[:-1, :], (seq_length, 3))
    X_eval = torch.tensor(X_val_win.squeeze(axis=1), dtype=torch.float32)
    y_eval = torch.tensor(val_scaled[seq_length:, 0:1], dtype=torch.float32)
    
    return TensorDataset(X_train, y_train), TensorDataset(X_eval, y_eval), scaler

def make_windows(scaled, seq_length=12):
    """
    Windows from rows [i : i + seq_length],
    target = cpu_util at row i + seq_length
    """
    win = np.lib.stride_tricks.sliding_window_view(scaled[:-1,:], (seq_length, scaled.shape[1]))
    X = torch.tensor(win.squeeze(axis=1), dtype=torch.float32)
    y = torch.tensor(scaled[seq_length:, 0:1], dtype=torch.float32)

    return TensorDataset(X, y)

def load_data(csv_path):
    df = pd.read_csv(csv_path)
    df = df.dropna(subset=FEATURES).reset_index(drop=True)
    raw = df[FEATURES].values.astype(np.float32)
    assert np.isfinite(raw).all(), "NaN/inf in raw features"
    return raw

# ================================================
# 2. LSTM Network Architecture
# ================================================

class WorkloadLSTM(nn.Module):

    def __init__(self, input_size, hidden_size, num_layers = 1, output_size = 1, dropout=0.0):

        super(WorkloadLSTM, self).__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.lstm = nn.LSTM(input_size, 
        hidden_size, num_layers, batch_first=True, dropout=dropout if num_layers > 1 else 0.0)

        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):

        # Initialize hidden state and cell state with zeros

        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)

        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)

        # Forward propagate LSTM
        out, _ = self.lstm(x, (h0, c0))

        # Decode the hidden state of the last time step

        out = self.fc(out[:, -1, :])

        return out

# =====================================================================
# 3. Stable Single-Model Training Loop 
# =====================================================================
def train_model(model, train_loader, val_loader, epochs=10, lr=0.001, weight_decay=0.0, verbose=True, save_path=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    criterion = nn.MSELoss()
    # Using a slightly lower learning rate (0.0005) or standard Adam
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    
    print(f"Launching plain training profile on target device: {device}")
    
    best_val, best_state = float("inf"), None

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for bx, by in train_loader:
            bx, by = bx.to(device), by.to(device)
            optimizer.zero_grad()
            loss = criterion(model(bx), by)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_loss += loss.item() * bx.size(0)
        avg_train = train_loss / len(train_loader.dataset)

        # Cross Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for bx, by in val_loader:
                bx, by = bx.to(device), by.to(device)
                val_loss += criterion(model(bx), by).item() * bx.size(0)
        avg_val = val_loss / len(val_loader.dataset)

        if avg_val < best_val:
            best_val = avg_val
            best_state = {k : v.detach().cpu().clone() for k, v in model.state_dict().items()}

        if verbose:
            print(f"Epoch {epoch+1:02d} | Train MSE: {avg_train:.6f} | Val MSE: {avg_val:.6f}") 
        
    if save_path and best_state is not None:
        torch.save(best_state, save_path)

    return best_val        



# =====================================================================

# 4. Bayesian Hyperparameter Optimization

# ====================================================================

def build_folds(raw_dev, n_splits):
    """
    Scale once per fold
    """
    folds = []
    for tr_idx, va_idx in TimeSeriesSplit(n_splits=n_splits).split(raw_dev):
        scaler = MinMaxScaler()
        folds.append((scaler.fit_transform(raw_dev[tr_idx]), scaler.transform(raw_dev[va_idx])))
        return folds
    
def objective(trial, folds, epochs):
    seq_length = trial.suggest_categorical("seq_length", [6, 12, 24, 48])
    hidden_size = trial.suggest_categorical("hidden_size", [16, 32, 64, 128])
    num_layers = trial.suggest_int("num_layers", 1, 3)
    dropout = trial.suggest_float("dropout", 0.0, 0.5)
    lr = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
    weight_decay = trial.suggest_float("weight_decay", 1e-8, 1e-3, log=True)

    scores = []
    for fold, (train_scaled, val_scaled) in enumerate(folds):
        train_loader = DataLoader(make_windows(train_scaled, seq_length), batch_size=512, shuffle=True)
        val_loader = DataLoader(make_windows(val_scaled, seq_length), batch_size=512, shuffle=False)

        model = WorkloadLSTM(3, hidden_size, num_layers=num_layers, dropout=dropout)

        best_val = train_model(model, train_loader, val_loader, epochs=epochs, lr=lr, weight_decay=weight_decay, verbose=False)

        scores.append(best_val)
        trial.report(float(np.mean(scores)), step=fold)
        if trial.should_prune():
            raise optuna.TrialPruned()
    
    return float(np.mean(scores))

# =====================================================================
# 5. Orchestration Entry Point
# =====================================================================
if __name__ == "__main__":
    csv_path = "workload_time_series.csv"
    raw = load_data(csv_path)

    split = int(len(raw) * 0.85)

    raw_dev, raw_test = raw[:split], raw[split:]

    folds = build_folds(raw_dev, n_splits=3)

    study = optuna.create_study(
        direction = "minimize",
        sampler = optuna.samplers.TPESampler(seed=32),
        pruner = optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=0),
    )
    study.optimize(lambda t: objective(t, folds, epochs=5), n_trials=20, show_progress_bar=True)

    print("Best CV MSE:", study.best_value)
    print("Best Hyperparameters:", study.best_params)

    p = study.best_params
    scaler = MinMaxScaler()
    dev_scaled = scaler.fit_transform(raw_dev)
    test_scaled = scaler.transform(raw_test)

    cut = int(len(dev_scaled) * 0.9)
    train_loader = DataLoader(make_windows(dev_scaled[:cut], p["seq_length"]),
                              batch_size=512, shuffle=True)
    val_loader = DataLoader(make_windows(dev_scaled[cut:], p["seq_length"]),
                            batch_size=512)
    test_loader = DataLoader(make_windows(test_scaled, p["seq_length"]),
                             batch_size=512)

    model = WorkloadLSTM(3, p["hidden_size"], p["num_layers"], dropout=p["dropout"])
    train_model(model, train_loader, val_loader, epochs=20, lr=p["lr"],
                weight_decay=p["weight_decay"], save_path="final_workload_lstm.pt")

    # Evaluate once on the test set
    device = next(model.parameters()).device
    model.load_state_dict(torch.load("final_workload_lstm.pt"))
    model.eval()
    mse, n = 0.0, 0
    with torch.no_grad():
        for bx, by in test_loader:
            bx, by = bx.to(device), by.to(device)
            mse += nn.functional.mse_loss(model(bx), by, reduction="sum").item()
            n += by.numel()
    print(f"Test MSE: {mse / n:.6f}")

    joblib.dump(scaler, "scaler.pkl")
    json.dump(study.best_params, open("best_params.json", "w"))
    np.save("raw_dev.npy", raw_dev)
    np.save("raw_test.npy", raw_test)