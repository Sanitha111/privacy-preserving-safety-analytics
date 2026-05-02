# models/train_ntu.py — Training pipeline on real NTU RGB+D data
"""
Trains all three Ghost-Vision models on real NTU RGB+D data.
Run from ghost_vision/ root:
    python models/train_ntu.py
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import numpy as np
import pickle
from torch.utils.data import DataLoader, WeightedRandomSampler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from collections import Counter

from models.stgcn import STGCN, FallSeverityCNN, PreFallLSTM
from utils.ntu_loader import NTUDataset, build_ntu_adjacency, NUM_NTU_JOINTS
from config import ACTIONS

# ── Paths ────────────────────────────────────────────────────
PKL_PATH      = "ntu120_3d.pkl"
STGCN_PATH    = "models/saved_models/stgcn_ntu60 (2).pth"
SEVERITY_PATH = "models/saved_models/severity_cnn_ntu60 (2).pth"
PREFALL_PATH  = "models/saved_models/prefall_lstm_ntu60 (2).pth"
RESULTS_PATH  = "models/saved_models/ntu_results.pkl"
os.makedirs("models/saved_models", exist_ok=True)

NUM_CLASSES   = len(ACTIONS)   # 5
CLASS_NAMES   = list(ACTIONS.values())


def make_weighted_sampler(dataset):
    """
    Creates a WeightedRandomSampler so minority classes (Fall, Motionless)
    get sampled as often as Normal — fixes class imbalance automatically.
    """
    labels = [dataset.samples[i][1] for i in range(len(dataset))]
    counts = Counter(labels)
    weights = [1.0 / counts[l] for l in labels]
    return WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)


def train_stgcn(epochs=50, batch_size=32, lr=0.001):
    print("\n" + "="*60)
    print("🧠 Training ST-GCN on NTU RGB+D")
    print("="*60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"   Device: {device}")

    # Load datasets
    train_ds = NTUDataset(PKL_PATH, split="xsub_train",
                          sequence_length=30, augment=True)
    val_ds   = NTUDataset(PKL_PATH, split="xsub_val",
                          sequence_length=30, augment=False)

    sampler      = make_weighted_sampler(train_ds)
    train_loader = DataLoader(train_ds, batch_size=batch_size,
                              sampler=sampler, num_workers=0)
    val_loader   = DataLoader(val_ds, batch_size=batch_size,
                              shuffle=False, num_workers=0)

    # Model — note NUM_NTU_JOINTS=25 not 33
    A     = build_ntu_adjacency().to(device)
    model = STGCN(num_joints=NUM_NTU_JOINTS,
                  num_classes=NUM_CLASSES).to(device)

    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                    optimizer, T_max=epochs)
    criterion = nn.CrossEntropyLoss()

    best_val_acc = 0.0
    train_losses, val_accs = [], []

    for epoch in range(epochs):
        # ── Train ──
        model.train()
        epoch_loss, correct, total = 0, 0, 0
        for X, y in train_loader:
            X, y = X.to(device), y.to(device)
            optimizer.zero_grad()
            out  = model(X, A)
            loss = criterion(out, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            epoch_loss += loss.item()
            correct    += (out.argmax(1) == y).sum().item()
            total      += y.size(0)

        train_losses.append(epoch_loss / len(train_loader))
        train_acc = correct / total

        # ── Validate ──
        model.eval()
        val_correct, val_total = 0, 0
        all_preds, all_true   = [], []
        with torch.no_grad():
            for X, y in val_loader:
                X, y   = X.to(device), y.to(device)
                out    = model(X, A)
                preds  = out.argmax(1)
                val_correct += (preds == y).sum().item()
                val_total   += y.size(0)
                all_preds.extend(preds.cpu().numpy())
                all_true.extend(y.cpu().numpy())

        val_acc = val_correct / val_total
        val_accs.append(val_acc)
        scheduler.step()

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), STGCN_PATH)

        if (epoch + 1) % 5 == 0:
            print(f"   Epoch [{epoch+1:3d}/{epochs}]  "
                  f"Loss: {train_losses[-1]:.4f}  "
                  f"Train: {train_acc:.2%}  "
                  f"Val: {val_acc:.2%}  "
                  f"Best: {best_val_acc:.2%}")

    print(f"\n✅ ST-GCN Training Complete! Best Val Acc: {best_val_acc:.2%}")
    print("\n📊 Classification Report:")
    print(classification_report(all_true, all_preds,
                                target_names=CLASS_NAMES))

    return {
        "best_val_acc": best_val_acc,
        "train_losses": train_losses,
        "val_accs": val_accs,
        "confusion_matrix": confusion_matrix(all_true, all_preds).tolist()
    }


def train_severity_cnn(epochs=30, batch_size=32, lr=0.001):
    """
    Train severity CNN using only Fall + Motionless sequences from NTU.
    Severity is estimated from how low the center of mass ends up.
    """
    print("\n" + "="*60)
    print("🧠 Training Severity CNN on NTU RGB+D")
    print("="*60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_ds = NTUDataset(PKL_PATH, split="xsub_train",
                          sequence_length=30, augment=True)

    # Filter to fall/motionless only and assign severity
    fall_X, fall_y = [], []
    for kp, label in train_ds.samples:
        if label not in [1, 3]:   # Fall or Motionless only
            continue
        kp_proc = train_ds._normalize(train_ds._pad_or_trim(kp))
        # Severity based on final average Y height (lower = more severe)
        final_height = np.mean(kp_proc[-1, :, 1])
        if final_height > -0.3:
            severity = 0   # Minor
        elif final_height > -0.6:
            severity = 1   # Moderate
        else:
            severity = 2   # Critical
        tensor = torch.FloatTensor(kp_proc.transpose(2, 0, 1))
        fall_X.append(tensor)
        fall_y.append(severity)

    if len(fall_X) == 0:
        print("⚠️  No fall sequences found — skipping severity training")
        return {}

    from collections import Counter
    print(f"   Fall/Motionless sequences: {len(fall_X)}")
    print(f"   Severity distribution: {Counter(fall_y)}")

    X = torch.stack(fall_X)
    y = torch.LongTensor(fall_y)

    # Simple 80/20 split
    n     = len(X)
    split = int(n * 0.8)
    idx   = torch.randperm(n)
    X_train, y_train = X[idx[:split]], y[idx[:split]]
    X_val,   y_val   = X[idx[split:]], y[idx[split:]]

    from torch.utils.data import TensorDataset
    train_loader = DataLoader(TensorDataset(X_train, y_train),
                              batch_size=batch_size, shuffle=True)

    model     = FallSeverityCNN(num_joints=NUM_NTU_JOINTS).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    best_val_acc = 0.0
    for epoch in range(epochs):
        model.train()
        for Xb, yb in train_loader:
            Xb, yb = Xb.to(device), yb.to(device)
            optimizer.zero_grad()
            loss = criterion(model(Xb), yb)
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            preds = model(X_val.to(device)).argmax(1).cpu()
        val_acc = accuracy_score(y_val, preds)
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), SEVERITY_PATH)

        if (epoch + 1) % 5 == 0:
            print(f"   Epoch [{epoch+1:3d}/{epochs}]  Val Acc: {val_acc:.2%}")

    print(f"\n✅ Severity CNN Complete! Best Val Acc: {best_val_acc:.2%}")
    return {"best_val_acc": best_val_acc}


def train_prefall_lstm(epochs=30, batch_size=32, lr=0.001):
    """
    Train PreFall LSTM using Pre-Fall Risk (label 2) vs Normal (label 0).
    """
    print("\n" + "="*60)
    print("🧠 Training Pre-Fall LSTM on NTU RGB+D")
    print("="*60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_ds = NTUDataset(PKL_PATH, split="xsub_train",
                          sequence_length=30, augment=True)

    lstm_X, lstm_y = [], []
    for kp, label in train_ds.samples:
        if label not in [0, 2]:   # Normal or Pre-Fall Risk only
            continue
        kp_proc = train_ds._normalize(train_ds._pad_or_trim(kp))
        # Flatten joints: (30, 25, 3) → (30, 75)
        flat = kp_proc.reshape(kp_proc.shape[0], -1).astype(np.float32)
        lstm_X.append(flat)
        lstm_y.append(1 if label == 2 else 0)

    print(f"   Sequences: {len(lstm_X)}  "
          f"(risk={lstm_y.count(1)}, safe={lstm_y.count(0)})")

    X = torch.FloatTensor(np.array(lstm_X))    # (N, 30, 75)
    y = torch.FloatTensor(lstm_y).unsqueeze(1) # (N, 1)

    n     = len(X)
    split = int(n * 0.8)
    idx   = torch.randperm(n)
    X_train, y_train = X[idx[:split]], y[idx[:split]]
    X_val,   y_val   = X[idx[split:]], y[idx[split:]]

    from torch.utils.data import TensorDataset
    train_loader = DataLoader(TensorDataset(X_train, y_train),
                              batch_size=batch_size, shuffle=True)

    input_size = X.shape[2]   # 25 * 3 = 75
    model      = PreFallLSTM(input_size=input_size).to(device)
    optimizer  = torch.optim.Adam(model.parameters(), lr=lr)
    criterion  = nn.BCELoss()

    best_val_acc = 0.0
    for epoch in range(epochs):
        model.train()
        for Xb, yb in train_loader:
            Xb, yb = Xb.to(device), yb.to(device)
            optimizer.zero_grad()
            loss = criterion(model(Xb), yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        model.eval()
        with torch.no_grad():
            preds = (model(X_val.to(device)).cpu() > 0.5).float()
        val_acc = accuracy_score(y_val.numpy(), preds.numpy())
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), PREFALL_PATH)

        if (epoch + 1) % 5 == 0:
            print(f"   Epoch [{epoch+1:3d}/{epochs}]  Val Acc: {val_acc:.2%}")

    print(f"\n✅ Pre-Fall LSTM Complete! Best Val Acc: {best_val_acc:.2%}")
    return {"best_val_acc": best_val_acc}


if __name__ == "__main__":
    print("\n" + "="*60)
    print("👁️  GHOST-VISION — NTU RGB+D Training Pipeline")
    print("="*60)

    stgcn_r    = train_stgcn(epochs=50, batch_size=32, lr=0.001)
    severity_r = train_severity_cnn(epochs=30, batch_size=32, lr=0.001)
    prefall_r  = train_prefall_lstm(epochs=30, batch_size=32, lr=0.001)

    results = {"stgcn": stgcn_r, "severity": severity_r, "prefall": prefall_r}
    with open(RESULTS_PATH, "wb") as f:
        pickle.dump(results, f)

    print("\n" + "="*60)
    print("✅ ALL MODELS TRAINED ON REAL DATA!")
    print(f"   ST-GCN accuracy    : {stgcn_r.get('best_val_acc', 0):.2%}")
    print(f"   Severity accuracy  : {severity_r.get('best_val_acc', 0):.2%}")
    print(f"   Pre-Fall accuracy  : {prefall_r.get('best_val_acc', 0):.2%}")
    print(f"   Models saved to    : models/saved_models/")
    print("="*60)