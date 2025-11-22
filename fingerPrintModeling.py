
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, random_split
from sklearn.metrics import confusion_matrix


# =========================
# Cell 1: Basic loading / tensors
# =========================

df = pd.read_csv("output_32.csv")

# Treat "data" as a single feature and normalize
X = df["data"].values.astype("float32").reshape(-1, 1)
X = (X - X.mean()) / X.std()
y = df["label"].values.astype("float32")

print(X.shape)
print(y.shape)

X_tensor = torch.tensor(X)
y_tensor = torch.tensor(y)


# =========================
# Cell 2: Model training with BCEWithLogitsLoss
# =========================

# Re-imports kept from the notebook for fidelity (optional / harmless)
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, random_split
from sklearn.metrics import confusion_matrix

# Load dataset
df = pd.read_csv("output_32.csv")

# If df["data"] is an array-like per row, this will stack into a 2D array.
# (e.g., each element is a vector of length 64 → shape (N, 64))
X = np.stack(df["data"].values).astype("float32")
y = df["label"].values.astype("float32")

# Normalize input
X = (X - X.mean()) / X.std()

# Reshape X for input shape (N, 1) if using only one feature.
# Adjust this if you want to use all features instead.
X_tensor = torch.tensor(X).view(-1, 1)
y_tensor = torch.tensor(y)

# Create Dataset and DataLoader
dataset = TensorDataset(X_tensor, y_tensor)
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_ds, val_ds = random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=64)


# Define neural network (no sigmoid at the output, handled by loss)
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


# Setup model and optimizer
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ChunkBoundaryNet().to(device)

# Handle class imbalance using pos_weight
neg = (y == 0).sum()
pos = (y == 1).sum()
pos_weight = torch.tensor([neg / pos], dtype=torch.float32).to(device)

# Use BCEWithLogitsLoss (expects raw logits, not sigmoid output)
loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Training loop
num_epochs = 10
for epoch in range(num_epochs):
    model.train()
    for xb, yb in train_loader:
        xb = xb.to(device)
        yb = yb.to(device).unsqueeze(1)

        logits = model(xb)  # raw logits
        loss = loss_fn(logits, yb)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # Evaluation
    model.eval()
    total = 0
    correct = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for xb, yb in val_loader:
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
        # Fallback if confusion_matrix doesn't return 2x2 (e.g., single class present)
        tn, fp, fn, tp = (0, 0, 0, 0)

    print(f"Epoch {epoch + 1}: Validation Accuracy = {acc:.4f}")
    print(f"Confusion Matrix: TP={tp}, FP={fp}, FN={fn}, TN={tn}")
