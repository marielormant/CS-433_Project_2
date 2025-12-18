import pickle
import numpy as np
import pandas as pd

from sklearn.metrics import accuracy_score, f1_score, recall_score
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.svm import LinearSVC
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.pipeline import Pipeline



SEED = 0
np.random.seed(SEED)

# 1. Load dataset x

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


x = eps_df.values # Use only the first 6 features (eps1-eps6)

# 2. Build label for the all dataset

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

y = np.zeros(len(F), dtype=int)  # class 0 by default, safe class 
max_FI = F_valid[has_failure].argmax(axis=1) + 1  # class 1..16
y[has_failure] = max_FI #replacing at failure row index the correspodning class


print(y[:10])
print(y.shape) #(1_000_000,)

# 3. Class weights to handle imbalanced dataset

# Class counts (computed from y)
counts = {c: int((y == c).sum()) for c in range(17)}
print("Class counts:", counts)

N = len(y)
K = 17  # number of classes
class_weights = {c: N / (K * counts[c]) for c in range(K)}

# Normalize weights so mean = 1 (recommended)
w_arr = np.array(list(class_weights.values()), dtype=np.float32)
w_arr /= w_arr.mean()
class_weights = {c: float(w_arr[c]) for c in range(K)}

print("Mean weight:", np.mean(list(class_weights.values())))
print("Min/Max weight:", min(class_weights.values()), max(class_weights.values()))


# 4. Cross-validation and Grid Search with Pipeline
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)

pipe = Pipeline([
    ("scaler", StandardScaler()),
    ("poly", PolynomialFeatures(degree=2, include_bias=False)),  # Expansion polynomiale
    ("svc", LinearSVC(dual=False, max_iter=4000))
])

C_values = [0.0001, 0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000]
param_grid = {"svc__C": C_values}

grid = GridSearchCV(
    estimator=pipe,
    param_grid=param_grid,
    scoring="recall_macro",   # minimize FN across classes
    cv=cv,
    n_jobs=-1,
    verbose=1,
    refit=True
)

# 5. sample weights 
sample_weight = np.array([class_weights[y_i] for y_i in y])
grid.fit(x, y, svc__sample_weight=sample_weight)

print("\nBest params:", grid.best_params_)
print("Best CV macro recall:", grid.best_score_)

# 6. Test on the whole dataset
y_pred = grid.predict(x)

recall = recall_score(y, y_pred, average="macro", zero_division=0)
f1 = f1_score(y, y_pred, average="macro")
acc = accuracy_score(y, y_pred)


print(f"Accuracy: {acc:.4f}")
print(f"Macro F1: {f1:.4f}")
print(f"Macro Recall: {recall:.4f}")
