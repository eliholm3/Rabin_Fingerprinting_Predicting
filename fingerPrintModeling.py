import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import confusion_matrix

# =========================
# Load Train + Test Files
# =========================

train_df = pd.read_csv("train_32.csv")
test_df  = pd.read_csv("test_32.csv")

# Convert "data" from 8-digit strings → floats → normalized
X_train = train_df["data"].astype("float32").values.reshape(-1, 1)
X_test  = test_df["data"].astype("float32").values.reshape(-1, 1)

# Normalize using TRAIN mean/std ONLY (correct ML practice)
mean = X_train.mean()
std = X_train.std()

X_train = (X_train - mean) / std
X_test  = (X_test - mean) / std

y_train = train_df["label"].astype("float32").values
y_test  = test_df["label"].astype("float32").values

# Convert to tensors
X_train_tensor = torch.tensor(X_train)
y_train_tensor = torch.tensor(y_train)

X_test_tensor = torch.tensor(X_test)
y_test_tensor = torch.tensor(y_test)

# Build TensorDatasets
train_ds = TensorDataset(X_train_tensor, y_train_tensor)
test_ds  = TensorDataset(X_test_tensor, y_test_tensor)

train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
test_loader  = DataLoader(test_ds, batch_size=64, shuffle=False)


# =========================
# Define Model
# =========================

class ChunkBoundaryNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
        )

    def forward(self, x):
        return self.net(x)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ChunkBoundaryNet().to(device)

# Class imbalance handling
neg = (y_train == 0).sum()
pos = (y_train == 1).sum()
pos_weight = torch.tensor([neg / pos], dtype=torch.float32).to(device)

loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)


# =========================
# Training Loop
# =========================

num_epochs = 10

for epoch in range(num_epochs):
    model.train()
    for xb, yb in train_loader:
        xb = xb.to(device)
        yb = yb.to(device).unsqueeze(1)

        logits = model(xb)
        loss = loss_fn(logits, yb)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # =========================
    # TEST SET EVALUATION
    # =========================
    model.eval()
    correct = 0
    total = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for xb, yb in test_loader:
            xb = xb.to(device)
            yb = yb.to(device).unsqueeze(1)

            logits = model(xb)
            probs = torch.sigmoid(logits)
            predicted = (probs > 0.5).float()

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(yb.cpu().numpy())

            correct += (predicted == yb).sum().item()
            total += yb.size(0)

    acc = correct / total
    cm = confusion_matrix(all_labels, all_preds, labels=[0, 1])

    if cm.size == 4:
        tn, fp, fn, tp = cm.ravel()
    else:
        tn, fp, fn, tp = (0, 0, 0, 0)

    print(f"Epoch {epoch + 1}: Test Accuracy = {acc:.4f}")
    print(f"Confusion Matrix: TP={tp}, FP={fp}, FN={fn}, TN={tn}")
