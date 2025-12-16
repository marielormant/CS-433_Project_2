"""
XGBoost hyperparameter optimization with Optuna (5-fold CV)
Goal: maximize macro F1.
"""

import pickle
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, accuracy_score
from sklearn.utils import class_weight
import optuna
from optuna.samplers import TPESampler
import time

print("XGBoost Optuna tuning (5-fold CV, macro F1)")

# 1) Load and preprocess data
with open("dataset/dataset.pkl", "rb") as f:
    dataset = pickle.load(f)

data = pd.json_normalize(dataset)
print(f"Loaded {len(dataset)} samples")

# Features: 6 epsilons only (no FI in inputs)
eps_df = pd.DataFrame(
    data["eps_global"].tolist(),
    columns=[f"eps{i}" for i in range(1, 7)]
)
x = eps_df.values  # Only 6 epsilon features

# Get FI columns for LABELS ONLY (not features!)
plies_cols = [
    'plies.0.0.FI_ft', 'plies.0.0.FI_fc', 'plies.0.0.FI_mt', 'plies.0.0.FI_mc',
    'plies.45.0.FI_ft', 'plies.45.0.FI_fc', 'plies.45.0.FI_mt', 'plies.45.0.FI_mc',
    'plies.90.0.FI_ft', 'plies.90.0.FI_fc', 'plies.90.0.FI_mt', 'plies.90.0.FI_mc',
    'plies.-45.0.FI_ft', 'plies.-45.0.FI_fc', 'plies.-45.0.FI_mt', 'plies.-45.0.FI_mc',
]

# Labels: derive class from FI >= 1; take max FI index when multiple failures
F = data[plies_cols].values  # FI used ONLY for labels, not features
mask_failure = (F >= 1)
has_failure = mask_failure.any(axis=1)

F_valid = F.copy()
F_valid[F_valid < 1] = -np.inf
y = np.zeros(len(F), dtype=int)
max_FI = F_valid[has_failure].argmax(axis=1) + 1
y[has_failure] = max_FI

print(f"Class distribution: {np.bincount(y)}")

# Split data: 80/20 train/test only (K-Fold on train)
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=42, stratify=y
)

# Setup K-Fold cross-validation
kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

print(f"Total training samples: {len(x_train):,}")
print(f"Test samples (held out): {len(x_test):,}")
print(f"K-Fold folds: 5")

# Optuna objective: maximize mean fold F1
def objective(trial):
    """
    Objective function for Optuna - maximize F1 score using K-Fold CV
    """
    
    # Suggest hyperparameters (including optional scaling)
    use_scaler = trial.suggest_categorical('use_scaler', [True, False])
    
    params = {
        'objective': 'multi:softmax',
        'num_class': 17,
        'max_depth': trial.suggest_int('max_depth', 4, 9),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2, log=True),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
        'colsample_bylevel': trial.suggest_float('colsample_bylevel', 0.6, 1.0),
        'min_child_weight': trial.suggest_int('min_child_weight', 2, 8),
        'lambda': trial.suggest_float('lambda', 0.0, 5.0),  # L2 regularization
        'alpha': trial.suggest_float('alpha', 0.0, 5.0),    # L1 regularization
        'gamma': trial.suggest_float('gamma', 0.0, 3.0),    # min loss reduction
        'n_estimators': trial.suggest_int('n_estimators', 100, 500, step=50),
        'random_state': 42,
        'eval_metric': 'mlogloss',
        'verbosity': 0
    }
    
    # Perform K-Fold cross-validation
    f1_scores = []
    
    for fold, (train_idx, val_idx) in enumerate(kfold.split(x_train, y_train)):
        # Split into train/val for this fold
        x_train_fold = x_train[train_idx]
        y_train_fold = y_train[train_idx]
        x_val_fold = x_train[val_idx]
        y_val_fold = y_train[val_idx]
        
        # Optionally standardize this fold
        if use_scaler:
            fold_scaler = StandardScaler()
            x_train_fold = fold_scaler.fit_transform(x_train_fold)
            x_val_fold = fold_scaler.transform(x_val_fold)
        
        # Train model
        model = xgb.XGBClassifier(**params)
        model.fit(
            x_train_fold, y_train_fold,
            eval_set=[(x_val_fold, y_val_fold)],
            verbose=False
        )
        
        # Evaluate on validation fold
        y_val_pred = model.predict(x_val_fold)
        f1 = f1_score(y_val_fold, y_val_pred, average='macro', zero_division=0)
        f1_scores.append(f1)
    
    # Return mean F1 score across all folds
    mean_f1 = np.mean(f1_scores)
    return mean_f1

# 2) Run Optuna study
sampler = TPESampler(seed=42)
study = optuna.create_study(
    direction='maximize',  # Maximize F1 score
    sampler=sampler
)

start_time = time.time()

# Callback to save study after each trial
def save_study_callback(study, trial):
    """Save study after each trial for checkpointing"""
    with open('optuna_xgboost_study.pkl', 'wb') as f:
        pickle.dump(study, f)


study.optimize(objective, n_trials=50, show_progress_bar=True, callbacks=[save_study_callback])

elapsed_time = time.time() - start_time
print(f"\nOptimization completed in {elapsed_time/3600:.1f} hours")

best_trial = study.best_trial
# 3) Report best trial
best_trial = study.best_trial

print("Best trial:")
print(f"  Trial number: {best_trial.number}")
print(f"  Best F1 Score: {best_trial.value:.6f}")
print("  Hyperparameters:")
for key, value in best_trial.params.items():
    print(f"    {key}: {value}")

# 4) Train final model with best hyperparameters
print("\nTraining final model with best hyperparameters (full train set)")

best_params = best_trial.params
use_scaler_best = best_params['use_scaler']

print(f"Best trial uses StandardScaler: {use_scaler_best}")

# Apply scaler if needed for final training
x_train_final = x_train.copy()
x_test_final = x_test.copy()

if use_scaler_best:
    final_scaler = StandardScaler()
    x_train_final = final_scaler.fit_transform(x_train_final)
    x_test_final = final_scaler.transform(x_test_final)
else:
    final_scaler = None

# Train final model on ALL training data
model = xgb.XGBClassifier(**{k: v for k, v in best_params.items() if k != 'use_scaler'})
model.fit(x_train_final, y_train, verbose=False)

# 5) Evaluate on test set
print("\nFinal evaluation on held-out test set")

y_test_pred = model.predict(x_test_final)
accuracy = accuracy_score(y_test, y_test_pred)
f1 = f1_score(y_test, y_test_pred, average='macro', zero_division=0)

# Count false negatives
is_actual_failure = (y_test > 0)
is_predicted_safe = (y_test_pred == 0)
false_negatives = np.sum(is_actual_failure & is_predicted_safe)
total_failures = np.sum(is_actual_failure)
fn_rate = false_negatives / total_failures if total_failures > 0 else 0

print(f"Accuracy: {accuracy:.4f}")
print(f"F1 Score (macro): {f1:.4f}")
print(f"False Negatives: {false_negatives} / {total_failures}")
print(f"FN Rate: {fn_rate:.4f}")

# 6) Save best model and scaler
with open('xgboost_model_optuna.pkl', 'wb') as f:
    pickle.dump(model, f)

if final_scaler is not None:
    with open('xgboost_scaler_optuna.pkl', 'wb') as f:
        pickle.dump(final_scaler, f)
    print("Saved:")
    print("  - xgboost_model_optuna.pkl")
    print("  - xgboost_scaler_optuna.pkl (StandardScaler used)")
else:
    print("Saved:")
    print("  - xgboost_model_optuna.pkl")
    print("  - No scaler saved (raw features used)")
print("Optimization complete")

# Save study results to file
with open('optuna_xgboost_study.pkl', 'wb') as f:
    pickle.dump(study, f)
print("Study saved to: optuna_xgboost_study.pkl")
