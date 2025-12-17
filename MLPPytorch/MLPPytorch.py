
import pickle
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    confusion_matrix, f1_score, accuracy_score, precision_score, recall_score
)

from classes import CustomTorchDataset, MLP



# 1) Load dataset 
with open("dataset/dataset.pkl", "rb") as f:
    dataset = pickle.load(f)

# Flatten nested dicts into flat columns like "plies.0.0.FI_ft"
data = pd.json_normalize(dataset)

print("Data shape:", data.shape)
print("Example columns:", data.columns[:25].tolist())



# 2) Build features x = eps1..eps6

if "eps_global" not in data.columns:
    raise KeyError("Column 'eps_global' not found. Check your dataset structure.")

eps_df = pd.DataFrame(
    data["eps_global"].tolist(),
    columns=[f"eps{i}" for i in range(1, 7)]
)

x = eps_df.to_numpy(dtype=np.float32)  # (N, 6)



# 3) Build labels y from plies FI columns
#    y=0 : no failure (all FI < 1)
#    y=1..16 : index of maximum FI among the 16 plies cols, +1

plies_cols = [
    "plies.0.0.FI_ft", "plies.0.0.FI_fc", "plies.0.0.FI_mt", "plies.0.0.FI_mc",
    "plies.45.0.FI_ft", "plies.45.0.FI_fc", "plies.45.0.FI_mt", "plies.45.0.FI_mc",
    "plies.90.0.FI_ft", "plies.90.0.FI_fc", "plies.90.0.FI_mt", "plies.90.0.FI_mc",
    "plies.-45.0.FI_ft", "plies.-45.0.FI_fc", "plies.-45.0.FI_mt", "plies.-45.0.FI_mc",
]

missing = [c for c in plies_cols if c not in data.columns]
if missing:
    raise KeyError(
        f"Missing {len(missing)} plies columns (example: {missing[:5]}). "
        f"Available 'plies' columns example: {[c for c in data.columns if 'plies' in c][:10]}"
    )

F = data[plies_cols].to_numpy(dtype=np.float32)  # (N, 16)

mask_failure = (F >= 1.0)
has_failure = mask_failure.any(axis=1)

F_valid = F.copy()
F_valid[F_valid < 1.0] = -np.inf

y = np.zeros(F.shape[0], dtype=np.int64)
y[has_failure] = F_valid[has_failure].argmax(axis=1) + 1  # 1..16


# 4) Train/test split and scaling

x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=0, stratify=y
)

scaler = StandardScaler()
x_train = scaler.fit_transform(x_train).astype(np.float32)
x_test = scaler.transform(x_test).astype(np.float32)



# 5) Torch datasets and loaders

train_dataset = CustomTorchDataset(x_train, y_train)
test_dataset = CustomTorchDataset(x_test, y_test)

batch_size = 64
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)



# 6) Device

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 7) Model and class weights on train

input_dim = x_train.shape[1]  # 6

#parameters
hidden1 = 308
hidden2 = 356
num_classes = 17
eta = 0.000685
weight_decay = 0.000026
epochs = 15

model = MLP(input_dim, hidden1, hidden2, num_classes).to(device)

# Class weights from y_train 
counts = np.bincount(y_train, minlength=num_classes).astype(np.float64)
Ntr = len(y_train)

# Avoid division by zero (in case a class is absent in y_train)
counts_safe = counts.copy()
counts_safe[counts_safe == 0] = 1.0

class_weights_np = Ntr / (num_classes * counts_safe)
class_weights_np = class_weights_np / class_weights_np.mean()

# If a class truly absent, setting weight=0 makes its loss irrelevant
class_weights_np[counts == 0] = 0.0

class_weights = torch.tensor(class_weights_np, dtype=torch.float32, device=device)
print("Class weights (train):", np.round(class_weights_np, 3))

criterion = nn.CrossEntropyLoss(weight=class_weights)
optimizer = torch.optim.Adam(model.parameters(), lr=eta, weight_decay=weight_decay)


# 8) Training loop

for epoch in range(epochs):
    model.train()
    total_loss = 0.0

    for xb, yb in train_loader:
        xb = xb.to(device)
        yb = yb.to(device)

        optimizer.zero_grad()
        logits = model(xb)
        loss = criterion(logits, yb)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * xb.size(0)

    epoch_loss = total_loss / len(train_loader.dataset)
    #print(f"Epoch {epoch+1}/{epochs} — Loss = {epoch_loss:.4f}")



# 9) Evaluation on Test set 

model.eval()
all_preds = []
all_targets = []

with torch.no_grad():
    for xb, yb in test_loader:
        xb = xb.to(device)
        logits = model(xb)
        preds = logits.argmax(dim=1).cpu().numpy()

        all_preds.append(preds)
        all_targets.append(yb.numpy())

y_pred = np.concatenate(all_preds)
y_true = np.concatenate(all_targets)

acc = accuracy_score(y_true, y_pred)
f1 = f1_score(y_true, y_pred, average="macro")
rec = recall_score(y_true, y_pred, average="macro", zero_division=0)
prec = precision_score(y_true, y_pred, average="macro", zero_division=0)

print("\nGlobal Metrics on Test Set")
print(f"Accuracy:        {acc:.4f}")
print(f"Macro F1:        {f1:.4f}")
print(f"Macro Recall:    {rec:.4f}")
print(f"Macro Precision: {prec:.4f}")


# 10) Confusion matrix 
cm = confusion_matrix(y_true, y_pred, labels=np.arange(num_classes))

print("\nDetailed Metrics Per Class")
print(f"{'Class':<6} {'Support':<10} {'FN':<8} {'FP':<8} {'Precision':<12} {'Recall':<10} {'F1':<10}")

f1s = []
accs = []
for c in range(num_classes):
    # True Positives, False Positives, False Negatives, Support
    TP = cm[c, c]
    FP = cm[:, c].sum() - TP
    FN = cm[c, :].sum() - TP
    support = cm[c, :].sum()

    p = TP / (TP + FP) if (TP + FP) > 0 else 0.0
    r = TP / (TP + FN) if (TP + FN) > 0 else 0.0
    f1_c = (2 * p * r / (p + r)) if (p + r) > 0 else 0.0
    acc_c = (TP / support) if support > 0 else 0.0

    print(f"{c:<6} {support:<10} {FN:<8} {FP:<8} {p:<12.4f} {r:<10.4f} {f1_c:<10.4f}")

    f1s.append(f1_c)
    accs.append(acc_c)


print(f"Average per-class F1:       {np.mean(f1s):.4f}")
print(f"Average per-class accuracy: {np.mean(accs):.4f}")
