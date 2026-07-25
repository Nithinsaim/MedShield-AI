# 🛡️ MedShield-AI: Intrusion-Aware Adaptive Encryption for IoMT

[![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org)
[![IoMT](https://img.shields.io/badge/IoMT-Security-red?style=for-the-badge)]()
[![CNN-BiLSTM](https://img.shields.io/badge/CNN--BiLSTM-Intrusion%20Detection-blue?style=for-the-badge)]()

---

## 🔴 Problem Statement

The Internet of Medical Things (IoMT) connects medical devices, wearable sensors, and cloud systems to deliver real-time healthcare. This ecosystem transmits highly sensitive patient data — ECG signals, EHR records, diagnostic images — continuously over networks.

**Current security solutions fail because:**
- Traditional encryption (AES, RSA) uses **static keys** that never adapt to changing threats
- Intrusion Detection Systems (IDS) **detect threats but don't respond** — encryption continues unchanged even after an attack is flagged
- IoMT devices are **resource-constrained** — heavy cryptographic methods are impractical
- Attacks like DDoS, MITM, replay, and data injection are increasingly sophisticated

**The result:** A detected intrusion does not stop ongoing data exposure. There is no closed-loop between detection and encryption response.

---

## 💡 Proposed Solution — MedShield-AI

A unified, closed-loop security framework that:
1. **Detects threats** using a hybrid CNN-BiLSTM model
2. **Classifies threat level** (Low / Medium / High) from a continuous probability score
3. **Adapts encryption** — key size, update frequency — based on threat severity
4. **Self-heals** — automatically re-keys, blocks, and isolates under attack

---

## 🏗️ System Architecture

```
IoMT Network Traffic
        │
        ▼
┌──────────────────────────────────────┐
│         PREPROCESSING                │
│  Normalize → One-hot encode →        │
│  Correlation-based feature selection │
└──────────────────────────────────────┘
        │
        ▼
┌──────────────────────────────────────┐
│     CNN (Spatial Feature Extractor)  │
│  Conv → ReLU → MaxPool               │
│  Learns: traffic bursts, patterns    │
└──────────────────────────────────────┘
        │
        ▼
┌──────────────────────────────────────┐
│   BiLSTM (Temporal Sequence Model)   │
│  Forward LSTM + Backward LSTM        │
│  Concatenated: h_t = [→h_t ; ←h_t]  │
│  Learns: multi-stage attack patterns │
└──────────────────────────────────────┘
        │
        ▼
   P_threat ∈ [0,1]  (sigmoid output)
        │
        ▼
┌──────────────────┬───────────────────┐
│ P < α → LOW      │ AES-128, slow rekey│
│ α ≤ P < β → MED  │ AES-256, rekey    │
│ P ≥ β → HIGH     │ AES-256, rekey +  │
│                  │ block + isolate   │
└──────────────────┴───────────────────┘
        │
        ▼
ECDH Key Exchange → KDF(K, context) → AES Encryption
        │
        ▼
Self-Healing: Isolate → Re-key → Restore channel
```

---

## 📁 Repository Structure

```
MedShield-AI/
├── README.md
├── requirements.txt
├── src/
│   ├── cnn_bilstm_ids.py        # Hybrid CNN-BiLSTM intrusion detection model
│   ├── threat_classifier.py     # Threshold-based threat level classification
│   ├── adaptive_encryption.py   # AES + ECDH + dynamic re-keying
│   ├── key_derivation.py        # KDF with contextual binding
│   ├── self_healing.py          # Automated response & recovery module
│   ├── preprocessing.py         # Feature normalization & selection
│   └── train.py                 # Training pipeline
├── notebooks/
│   └── MedShieldAI_Demo.ipynb
├── results/
│   └── metrics.json
├── images/
│   └── system_architecture.png
└── LICENSE
```

---

## ⚙️ Methods

### 1. CNN-BiLSTM Intrusion Detection

**CNN — Spatial Feature Extraction:**
```
F_k = ReLU(W_k * X' + b_k)     # Convolution
P_k = MaxPool(F_k)               # Dimensionality reduction
```
Captures discriminative patterns: abnormal traffic bursts, irregular packet structures.

**BiLSTM — Temporal Modeling:**
```
→h_t = LSTM(P_t, →h_{t-1})      # Forward pass
←h_t = LSTM(P_t, ←h_{t+1})      # Backward pass
h_t  = [→h_t ; ←h_t]            # Concatenated representation
```
Models past + future context simultaneously — effective against multi-stage attacks.

**Threat Probability Output:**
```
P_threat = sigmoid(W_o · h_t + b_o)    # ∈ [0, 1]
```

**Loss Function:**
```
L = -(1/N) Σ [y_i·log(P_i) + (1-y_i)·log(1-P_i)]   # Binary cross-entropy
```
Optimized with Adam optimizer.

---

### 2. Threat-Aware Encryption Policy

| Threat Level | Condition | AES Key Size | Re-key Interval | Response |
|---|---|---|---|---|
| **Low** | P_threat < α | AES-128 | Slow | Monitor |
| **Medium** | α ≤ P < β | AES-256 | Moderate | Re-key |
| **High** | P_threat ≥ β | AES-256 | Rapid | Re-key + Block + Isolate |

**Key update interval adapts automatically:**
```
t_update = 1 / (1 + P_threat)
```
Higher threat → shorter interval → more frequent key rotation.

---

### 3. Secure Key Exchange — ECDH

```
Q_A = d_A · G       # Device A public key
Q_B = d_B · G       # Device B public key
K   = d_A · Q_B     # Shared secret (never transmitted)
    = d_B · Q_A
```
**Key Derivation with Contextual Binding:**
```
K_enc = KDF(K, [timestamp, session_id, device_id, threat_level])
```
Every session uses a unique derived key — forward secrecy guaranteed.

---

### 4. Dynamic Re-Keying
```
K_new = KDF(K_old, P_threat, t)
```
Compromised keys are replaced automatically. Past sessions remain secure.

---

### 5. Self-Healing Mechanism

```
State Machine: Normal → Suspicious → Compromised
                                         │
                              Filter malicious traffic:
                              X_safe = X \ X_malicious
                                         │
                              Regenerate keys + re-establish channel:
                              C_new = AES(K_new, M)
```

---

## 📊 Evaluation

**Intrusion Detection Metrics:** Accuracy, Precision, Recall, F1-Score  
**System Metrics:** Latency, Computational Overhead  
**Datasets:** Benchmark network intrusion detection datasets (normal + malicious traffic)

---

## 🚀 Getting Started

```bash
pip install -r requirements.txt
```

### Train the IDS Model
```bash
python src/train.py --epochs 50 --lr 1e-3
```

### Run Adaptive Encryption Demo
```bash
python src/adaptive_encryption.py --threat_level high
```

### Full Pipeline
```bash
python src/self_healing.py --mode demo
```

---

## 📦 requirements.txt

See `requirements.txt` for full dependencies.

---

<div align="center">
📍 Amrita Vishwa Vidyapeetham, Coimbatore, Tamil Nadu
</div>
