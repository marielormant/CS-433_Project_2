import optuna
from classes import NumpyDataset as npdata
import pickle
import csv
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split



with open("dataset/dataset.pkl", "rb") as f:
    dataset = pickle.load(f)

data = pd.json_normalize(dataset)
print(np.array(data).shape) # (1 000 000, 17)

print(data.columns.tolist())
"""['eps_global', 'plies.0.0.FI_ft', 'plies.0.0.FI_fc', 
'plies.0.0.FI_mt', 'plies.0.0.FI_mc', 'plies.45.0.FI_ft', 
'plies.45.0.FI_fc', 'plies.45.0.FI_mt', 'plies.45.0.FI_mc',
 'plies.90.0.FI_ft', 'plies.90.0.FI_fc', 'plies.90.0.FI_mt',
 'plies.90.0.FI_mc', 'plies.-45.0.FI_ft', 'plies.-45.0.FI_fc', 
 'plies.-45.0.FI_mt', 'plies.-45.0.FI_mc']"""

eps_global = data['eps_global']
print(np.array(eps_global).shape) # output (1 000 000,)

eps_df = pd.DataFrame(
    eps_global.tolist(), 
    columns=[f"eps{i}" for i in range(1, 7)]
)

print(eps_df.shape)  # output (1_000_000, 6)
print(eps_df.head())

plies_cols = ['plies.0.0.FI_ft', 'plies.0.0.FI_fc', 
'plies.0.0.FI_mt', 'plies.0.0.FI_mc', 'plies.45.0.FI_ft', 
'plies.45.0.FI_fc', 'plies.45.0.FI_mt', 'plies.45.0.FI_mc',
 'plies.90.0.FI_ft', 'plies.90.0.FI_fc', 'plies.90.0.FI_mt',
 'plies.90.0.FI_mc', 'plies.-45.0.FI_ft', 'plies.-45.0.FI_fc', 
 'plies.-45.0.FI_mt', 'plies.-45.0.FI_mc']

plies_df = data[plies_cols]

data_df = pd.concat([eps_df, plies_df], axis=1)

print(data_df.shape)    # (1_000_000, 22)
print(data_df.head())

x = data_df.values

# Build label for the all dataset

# List of the 16 FI feature columns (in a fixed order)
fi_cols = [
    'plies.0.0.FI_ft', 'plies.0.0.FI_fc', 'plies.0.0.FI_mt', 'plies.0.0.FI_mc',
    'plies.45.0.FI_ft', 'plies.45.0.FI_fc', 'plies.45.0.FI_mt', 'plies.45.0.FI_mc',
    'plies.90.0.FI_ft', 'plies.90.0.FI_fc', 'plies.90.0.FI_mt', 'plies.90.0.FI_mc',
    'plies.-45.0.FI_ft', 'plies.-45.0.FI_fc', 'plies.-45.0.FI_mt', 'plies.-45.0.FI_mc',
]

# Extract FI values, so extract the 16 FI columns and their corresponding rows
F = data_df[fi_cols].values  # (1_000_000, 16)

# Define label: 0 if no FI > 1, otherwise argmax + 1
mask_failure = (F >= 1) 
has_failure = mask_failure.any(axis=1)  #  (N,), True if at least one failure for each line

F_valid = F.copy()
F_valid[F_valid < 1] = -np.inf

y = np.zeros(len(F), dtype=int)  # class 0 by default
max_FI = F_valid[has_failure].argmax(axis=1) + 1  # class 1..16
y[has_failure] = max_FI #replacing at failure row index the correspodning class


print(y[:10])
print(y.shape) #(1_000_000,)

#count of each class
numb_0 = len(np.where(y == 0)[0]) # no failure : 0 by default
# failure class 1..16 :
numb_1 = len(np.where(y == 1)[0]) 
numb_2 = len(np.where(y == 2)[0])
numb_3 = len(np.where(y == 3)[0])
numb_4 = len(np.where(y == 4)[0])
numb_5 = len(np.where(y == 5)[0])
numb_6 = len(np.where(y == 6)[0])
numb_7 = len(np.where(y == 7)[0])
numb_8 = len(np.where(y == 8)[0])
numb_9 = len(np.where(y == 9)[0])
numb_10 = len(np.where(y == 10)[0])
numb_11= len(np.where(y == 11)[0])
numb_12 = len(np.where(y == 12)[0])
numb_13 = len(np.where(y == 13)[0])
numb_14= len(np.where(y == 14)[0])
numb_15= len(np.where(y == 15)[0])
numb_16 = len(np.where(y == 16)[0])

print(numb_0)
print(numb_1)
print(numb_2)
print(numb_3)
print(numb_4)
print(numb_5)
print(numb_6)
print(numb_7)
print(numb_8)
print(numb_9)
print(numb_10)
print(numb_11)
print(numb_12)
print(numb_13)
print(numb_14)
print(numb_15)
print(numb_16)



#class weighting to balance 


counts = {
    0: 209953,
    1: 87948,
    2: 111527,
    3: 51882,
    4: 57206,
    5: 29708,
    6: 23705,
    7: 20207,
    8: 12997,
    9: 87721,
    10: 111797,
    11: 51575,
    12: 56978,
    13: 29464,
    14: 24010,
    15: 20295,
    16: 13027
}

N = data_df.shape[0]
K = data_df[fi_cols].shape[1] + 1 #number of features without deformation epsilon

class_weights = {c: N / (K * n) for c, n in counts.items()}


#extract data into 80% train/ 20%test
# x = eps_df.values only epsilon ? 
x_train, x_test, y_train, y_test = train_test_split(
    x, y,
    test_size=0.2,
    random_state=0,
    stratify=y
)

# Scaling
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test  = scaler.transform(x_test)

# Build sample weights for TRAIN
sample_weight_train = np.array([class_weights[y_i] for y_i in y_train])

# NNOpti.py
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score
import optuna

from classes import MLP, NumpyDataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Training for 1 epoch
def train_one_epoch(model, loader, optimizer, criterion):
    model.train()
    total_loss = 0.0

    for Xb, yb in loader:
        Xb = Xb.to(device)
        yb = yb.to(device)

        optimizer.zero_grad()
        logits = model(Xb)
        loss = criterion(logits, yb)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * Xb.size(0)

    return total_loss / len(loader.dataset)



# Evaluation: macro F1

def eval_f1_macro(model, loader):
    model.eval()
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device)
            logits = model(xb)
            preds = logits.argmax(dim=1)

            all_preds.append(preds.cpu().numpy())
            all_targets.append(yb.numpy())

    y_pred = np.concatenate(all_preds)
    y_true = np.concatenate(all_targets)
    return f1_score(y_true, y_pred, average="macro")



# Optuna objective factory

def make_objective(train_dataset, test_dataset, class_weights, input_dim, num_classes):

    def objective(trial):
        # Hyperparameters
        hidden1 = trial.suggest_int("hidden1", 32, 256)
        hidden2 = trial.suggest_int("hidden2", 32, 256)
        lr      = trial.suggest_loguniform("learning_rate", 1e-4, 1e-2)
        wd      = trial.suggest_loguniform("weight_decay", 1e-6, 1e-2)
        batch_size = trial.suggest_categorical("batch_size", [128, 256, 512])
        num_epochs = trial.suggest_int("epochs", 5, 15)

        # Dataloaders
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_loader  = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

        # Model
        model = MLP(input_dim, hidden1, hidden2, num_classes).to(device)
        criterion = nn.CrossEntropyLoss(weight=class_weights)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)

        # Training loop
        best_f1 = 0.0
        for epoch in range(num_epochs):
            train_one_epoch(model, train_loader, optimizer, criterion)
            f1 = eval_f1_macro(model, test_loader)

            trial.report(f1, epoch)
            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()

            best_f1 = max(best_f1, f1)

        return best_f1

    return objective


# Run Optuna study

def run_optuna(train_dataset, test_dataset, class_weights, input_dim, num_classes, n_trials=30):
    objective = make_objective(train_dataset, test_dataset, class_weights, input_dim, num_classes)

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials)

    print("Best:", study.best_params)
    return study.best_params

import NNOpti as opti





# 4. Train-test split + scaling

x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=0, stratify=y
)

scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test  = scaler.transform(x_test)


# 5. PyTorch Dataset using classes.py & DataLoader

train_dataset = npdata(x_train, y_train)
test_dataset  = npdata(x_test, y_test)

batch_size = opti.batch_size
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader  = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


best = opti.run_optuna(
    train_dataset,
    test_dataset,
    class_weights,
    input_dim=x_train.shape[1],
    num_classes=17,
    n_trials=50
)

hidden1     = best["hidden1"]
hidden2     = best["hidden2"]
eta         = best["learning_rate"]
weightdecay = best["weight_decay"]
batch_size  = best["batch_size"]
epochs      = best["epochs"]

print("=== Best hyperparameters ===")
print(best)
