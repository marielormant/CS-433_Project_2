"""
XGBoost Baseline Model (no hyperparameter optimization)
- Inputs: 6 epsilon features
- Labels: 17 classes (0=no failure, 1-16 failure classes derived from FI)
- Split: 80/20 train/test
- Metrics: Accuracy, F1 (macro), FN rate, classification report
- Saves: xgboost_baseline_model.pkl
"""

import pickle
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, classification_report

print("XGBoost Baseline Model — baseline (no tuning)")

# 1) Load and prepare data
with open("dataset/dataset.pkl", "rb") as f:
    dataset = pickle.load(f)

data = pd.json_normalize(dataset)
print(f"Samples: {len(data):,}")

# Features: 6 epsilons only (no FI in inputs)
eps_df = pd.DataFrame(
    data["eps_global"].tolist(),
    columns=[f"eps{i}" for i in range(1, 7)]
)
X = eps_df.values

# Labels: derive class from FI >= 1; take max FI index when multiple failures
plies_cols = [
    'plies.0.0.FI_ft', 'plies.0.0.FI_fc', 'plies.0.0.FI_mt', 'plies.0.0.FI_mc',
    'plies.45.0.FI_ft', 'plies.45.0.FI_fc', 'plies.45.0.FI_mt', 'plies.45.0.FI_mc',
    'plies.90.0.FI_ft', 'plies.90.0.FI_fc', 'plies.90.0.FI_mt', 'plies.90.0.FI_mc',
    'plies.-45.0.FI_ft', 'plies.-45.0.FI_fc', 'plies.-45.0.FI_mt', 'plies.-45.0.FI_mc',
]
F = data[plies_cols].values
mask_failure = (F >= 1.0)
has_failure = mask_failure.any(axis=1)
F_valid = F.copy()
F_valid[F_valid < 1.0] = -np.inf
y = np.zeros(len(F), dtype=int)
y[has_failure] = F_valid[has_failure].argmax(axis=1) + 1

print(f"Class distribution: {np.bincount(y)}")

# Split stratified 80/20
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 2) Baseline hyperparameters (typical starting values)
baseline_params = {
    'objective': 'multi:softmax',
    'num_class': 17,
    'max_depth': 6,
    'learning_rate': 0.10,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'colsample_bylevel': 1.0,
    'min_child_weight': 3,
    'gamma': 0.0,
    'reg_alpha': 0.0,
    'reg_lambda': 1.0,
    'n_estimators': 300,
    'random_state': 42,
    'eval_metric': 'mlogloss',
    'verbosity': 1
}

print("\nBaseline hyperparameters:")
for k, v in baseline_params.items():
    print(f"  {k}: {v}")

# 3) Train baseline model
model = xgb.XGBClassifier(**baseline_params)
model.fit(X_train, y_train)

# 4) Evaluate on test set
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
f1_macro = f1_score(y_test, y_pred, average='macro', zero_division=0)
f1_weighted = f1_score(y_test, y_pred, average='weighted', zero_division=0)

# False negatives analysis (missed failures)
is_actual_failure = (y_test > 0)
is_predicted_safe = (y_pred == 0)
false_negatives = np.sum(is_actual_failure & is_predicted_safe)
total_failures = np.sum(is_actual_failure)
fn_rate = false_negatives / total_failures if total_failures > 0 else 0

print(f"Accuracy: {acc:.4f}")
print(f"F1 Score (macro): {f1_macro:.4f}")
print(f"F1 Score (weighted): {f1_weighted:.4f}")
print(f"False Negatives: {false_negatives} / {total_failures}")
print(f"FN Rate: {fn_rate:.4f}")
print("\nClassification report:")
print(classification_report(y_test, y_pred, zero_division=0))

# 5) Save model
with open('xgboost_baseline_model.pkl', 'wb') as f:
    pickle.dump(model, f)
print("Saved: xgboost_baseline_model.pkl")
print("Baseline training complete")
