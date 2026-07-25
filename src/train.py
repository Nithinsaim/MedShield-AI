"""
train.py
Training pipeline for CNN-BiLSTM Intrusion Detection Model.

Loss:      Binary Cross-Entropy
Optimizer: Adam
Metrics:   Accuracy, Precision, Recall, F1-Score
"""

import torch, numpy as np, os, json
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score, precision_score, recall_score, \
                            f1_score, classification_report
from cnn_bilstm_ids import CNNBiLSTMIDS


def train(X_train: np.ndarray, y_train: np.ndarray,
          X_val:   np.ndarray, y_val:   np.ndarray,
          input_dim: int, seq_len: int = 10,
          epochs: int = 50, lr: float = 1e-3, batch_size: int = 256,
          out_dir: str = '../models/'):

    os.makedirs(out_dir, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Training on: {device}")

    # Expand features into sequence dimension
    def to_seq(X):
        return torch.tensor(X, dtype=torch.float32) \
                     .unsqueeze(1).expand(-1, seq_len, -1).to(device)

    Xt = to_seq(X_train)
    yt = torch.tensor(y_train, dtype=torch.float32).to(device)
    Xv = to_seq(X_val)
    yv = torch.tensor(y_val,   dtype=torch.float32).to(device)

    loader = DataLoader(TensorDataset(Xt, yt), batch_size=batch_size, shuffle=True)

    model     = CNNBiLSTMIDS(input_dim=input_dim, seq_len=seq_len).to(device)
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    best_f1, results = 0.0, {}

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        for xb, yb in loader:
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        if epoch % 5 == 0:
            model.eval()
            with torch.no_grad():
                probs = model(Xv).cpu().numpy()
            preds = (probs > 0.5).astype(int)
            yv_np = yv.cpu().numpy()
            acc   = accuracy_score(yv_np, preds)
            prec  = precision_score(yv_np, preds, zero_division=0)
            rec   = recall_score(yv_np, preds, zero_division=0)
            f1    = f1_score(yv_np, preds, zero_division=0)
            print(f"Epoch {epoch:3d}/{epochs} | loss={total_loss/len(loader):.4f} "
                  f"| acc={acc:.4f} | prec={prec:.4f} | rec={rec:.4f} | f1={f1:.4f}")

            if f1 > best_f1:
                best_f1 = f1
                torch.save(model.state_dict(),
                           os.path.join(out_dir, 'cnn_bilstm_ids_best.pth'))
                results = {'accuracy': acc, 'precision': prec,
                           'recall': rec, 'f1': f1, 'epoch': epoch}

    print(f"\nBest F1: {best_f1:.4f}")
    print("\nFinal Classification Report:")
    model.eval()
    with torch.no_grad():
        preds = (model(Xv).cpu().numpy() > 0.5).astype(int)
    print(classification_report(yv.cpu().numpy(), preds,
                                target_names=['Normal', 'Attack']))

    with open('../results/metrics.json', 'w') as f:
        json.dump(results, f, indent=2)


if __name__ == '__main__':
    print("Usage: from train import train")
    print("       train(X_train, y_train, X_val, y_val, input_dim=<n_features>)")
