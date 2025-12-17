"""
XGBoost model using best hyperparameters from Optuna
Loads study results and trains/evaluates a final model.
"""

import pickle
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, classification_report

print("XGBoost Model with optimized hyperparameters")

# 1) Load and preprocess data
with open("dataset/dataset.pkl", "rb") as f:
    dataset = pickle.load(f)

data = pd.json_normalize(dataset)
print(f"Loaded {len(dataset)} samples")

# Features: 6 epsilons
eps_df = pd.DataFrame(
    data["eps_global"].tolist(),
    columns=[f"eps{i}" for i in range(1, 7)]
)
x = eps_df.values  # Only 6 epsilon features

# Labels: derive class from FI >= 1; take max FI index when multiple failures
plies_cols = [
    'plies.0.0.FI_ft', 'plies.0.0.FI_fc', 'plies.0.0.FI_mt', 'plies.0.0.FI_mc',
    'plies.45.0.FI_ft', 'plies.45.0.FI_fc', 'plies.45.0.FI_mt', 'plies.45.0.FI_mc',
    'plies.90.0.FI_ft', 'plies.90.0.FI_fc', 'plies.90.0.FI_mt', 'plies.90.0.FI_mc',
    'plies.-45.0.FI_ft', 'plies.-45.0.FI_fc', 'plies.-45.0.FI_mt', 'plies.-45.0.FI_mc',
]

# Build labels
F = data[plies_cols].values  # FI used ONLY for labels, not features
mask_failure = (F >= 1)
has_failure = mask_failure.any(axis=1)

F_valid = F.copy()
F_valid[F_valid < 1] = -np.inf
y = np.zeros(len(F), dtype=int)
max_FI = F_valid[has_failure].argmax(axis=1) + 1
y[has_failure] = max_FI

print(f"Class distribution: {np.bincount(y)}")

# Split data: 80/20 train/test (stratified)
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=42, stratify=y
)

# Standardize features
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

print(f"Training samples: {len(x_train):,}")
print(f"Test samples: {len(x_test):,}")

# 2) Load best hyperparameters from optimization
try:
    with open('xgb/optuna_xgboost_study.pkl', 'rb') as f:
        study = pickle.load(f)
    
    best_trial = study.best_trial
    best_params = best_trial.params
    
    print(f"Best trial: {best_trial.number}")
    print(f"Best F1 Score from optimization: {best_trial.value:.6f}")
    print("\nBest Hyperparameters:")
    for key, value in best_params.items():
        print(f"  {key}: {value}")
    
except FileNotFoundError:
    print("ERROR: optuna_xgboost_study.pkl not found!")
    print("Please run xgboost_optuna.py first to optimize hyperparameters.")
    exit(1)

# 3) Train final model with best hyperparameters

# Use best parameters
params = {
    'objective': 'multi:softmax',
    'num_class': 17,
    'max_depth': best_params['max_depth'],
    'learning_rate': best_params['learning_rate'],
    'subsample': best_params['subsample'],
    'colsample_bytree': best_params['colsample_bytree'],
    'colsample_bylevel': best_params['colsample_bylevel'],
    'min_child_weight': best_params['min_child_weight'],
    'lambda': best_params['lambda'],
    'alpha': best_params['alpha'],
    'gamma': best_params['gamma'],
    'n_estimators': best_params['n_estimators'],
    'random_state': 42,
    'eval_metric': 'mlogloss',
    'verbosity': 1
}

model = xgb.XGBClassifier(**params)
model.fit(x_train, y_train)

# 4) Evaluate on test set
print("\nFinal evaluation on test set")

y_test_pred = model.predict(x_test)

accuracy = accuracy_score(y_test, y_test_pred)
f1 = f1_score(y_test, y_test_pred, average='macro', zero_division=0)

# False negatives (missed failures)
is_actual_failure = (y_test > 0)
is_predicted_safe = (y_test_pred == 0)
false_negatives = np.sum(is_actual_failure & is_predicted_safe)
total_failures = np.sum(is_actual_failure)
fn_rate = false_negatives / total_failures if total_failures > 0 else 0

print(f"\nAccuracy: {accuracy:.4f}")
print(f"F1 Score (macro): {f1:.4f}")
print(f"False Negatives: {false_negatives} / {total_failures}")
print(f"FN Rate: {fn_rate:.4f}")

print("\nClassification report:")
print(classification_report(y_test, y_test_pred))

# 5) Save best model and scaler
with open('xgboost_best_model.pkl', 'wb') as f:
    pickle.dump(model, f)

with open('xgboost_best_scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)

print("Saved:")
print("  - xgboost_best_model.pkl (trained model)")
print("  - xgboost_best_scaler.pkl (feature scaler)")
print("Training complete")
