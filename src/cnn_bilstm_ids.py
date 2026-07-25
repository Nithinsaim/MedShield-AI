"""
cnn_bilstm_ids.py
Hybrid CNN-BiLSTM Intrusion Detection Model for IoMT Security.

Architecture:
  CNN    → spatial feature extraction from traffic patterns
  BiLSTM → temporal modeling of sequential traffic (forward + backward)
  FC + Sigmoid → P_threat ∈ [0, 1]

Loss:   Binary cross-entropy
Optim:  Adam
Output: P_threat — probability that traffic instance is malicious
"""

import torch
import torch.nn as nn


class CNNBiLSTMIDS(nn.Module):
    def __init__(self, input_dim: int, seq_len: int = 10,
                 cnn_filters: int = 64, lstm_hidden: int = 128,
                 dropout: float = 0.3):
        super().__init__()

        # CNN: spatial feature extractor
        # Learns: abnormal traffic bursts, irregular packet structures
        self.cnn = nn.Sequential(
            nn.Conv1d(1, cnn_filters, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(cnn_filters, cnn_filters * 2, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(2)
        )

        cnn_out_len = seq_len // 4
        cnn_out_dim = (cnn_filters * 2) * cnn_out_len

        # BiLSTM: temporal sequence modeling
        # Forward LSTM:   →h_t = LSTM(P_t, →h_{t-1})
        # Backward LSTM:  ←h_t = LSTM(P_t, ←h_{t+1})
        # Output:         h_t  = [→h_t ; ←h_t]
        self.bilstm = nn.LSTM(
            input_size=cnn_out_dim,
            hidden_size=lstm_hidden,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=dropout
        )

        # Threat probability output: P_threat ∈ [0, 1]
        self.classifier = nn.Sequential(
            nn.Linear(lstm_hidden * 2, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        # x: (B, seq_len, input_dim)
        B, T, D = x.shape

        # CNN over each timestep
        x = x.view(B * T, 1, D)
        x = self.cnn(x)                       # (B*T, cnn_filters*2, reduced_len)
        x = x.view(B, T, -1)                  # (B, T, cnn_out_dim)

        # BiLSTM temporal modeling
        out, _ = self.bilstm(x)               # (B, T, 2*lstm_hidden)
        out = out[:, -1, :]                   # last timestep → (B, 2*lstm_hidden)

        return self.classifier(out).squeeze(-1)   # (B,) — P_threat


if __name__ == '__main__':
    model = CNNBiLSTMIDS(input_dim=41, seq_len=10)
    x = torch.randn(32, 10, 41)
    p = model(x)
    print(f"P_threat shape: {p.shape}")       # (32,)
    total = sum(param.numel() for param in model.parameters())
    print(f"Total parameters: {total:,}")
