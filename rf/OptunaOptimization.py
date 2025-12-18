"""
Random Forest Optuna Optimization (hyperparameter search with safety constraint)
- Inputs: 6 epsilon strain features
- Labels: 17 classes (16 ply-mode failure classes + "safe")
- Split: Internal train/validation split per trial + final 80/20 test evaluation
- Objective: Maximize weighted F1 with penalty if safety recall < threshold
- Outputs: Best model + JSON summaries + Optuna SQLite study + trial snapshots
- Saves: random_forest_final.pkl (default), best_params.json, best_metrics.json, snapshots/*
"""

import argparse
import json
import os
import signal
import sys
import time
import pickle
from datetime import datetime
from typing import Dict, Any, Optional

import numpy as np
import optuna
from optuna.trial import TrialState
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
import joblib


# Safety metrics
def compute_safety_false_negatives(y_true, y_pred) -> int:
    y_true = np.asarray(y_true, dtype=object)
    y_pred = np.asarray(y_pred, dtype=object)
    unsafe_mask = (y_true != "safe")
    return int(np.sum(y_pred[unsafe_mask] == "safe"))


def compute_safety_recall(y_true, y_pred) -> float:
    y_true = np.asarray(y_true, dtype=object)
    y_pred = np.asarray(y_pred, dtype=object)
    unsafe_mask = (y_true != "safe")
    if unsafe_mask.sum() == 0:
        return 1.0
    return float(np.mean(y_pred[unsafe_mask] != "safe"))


# Helper functions (data and label construction)
def format_ply_angle(ply):
    try:
        ply_f = float(ply)
        if ply_f.is_integer():
            return str(int(ply_f))
        return str(ply_f)
    except Exception:
        return str(ply)


def get_eps_features(sample):
    eps = sample.get("eps", sample.get("eps_global", None))
    if isinstance(eps, np.ndarray):
        eps = eps.tolist()
    return np.array(eps[:6], dtype=float)


def iter_failure_indices(ply_dict):
    src = ply_dict.get("criteria", ply_dict)
    for key, value in src.items():
        if isinstance(key, str) and (key.startswith("FI_") or key.startswith("F_")):
            mode = key.split("_", 1)[1]
            if hasattr(value, "item"):
                value = value.item()
            yield mode, float(value)


def build_label(sample, threshold=1.0):
    best_value = -np.inf
    best_label = "safe"

    for ply, ply_dict in sample["plies"].items():
        ply_label = format_ply_angle(ply)
        for mode, val in iter_failure_indices(ply_dict):
            if val > best_value:
                best_value = val
                best_label = f"{ply_label}_{mode}"

    return "safe" if best_value <= threshold else best_label


def build_class_weights(y, safe_weight: float, unsafe_mult: float) -> Dict[str, float]:
    classes = np.unique(y)
    return {"safe": safe_weight, **{cls: unsafe_mult for cls in classes if cls != "safe"}}


# Optuna search space
def suggest_hparams(trial: optuna.trial.Trial, mode: str, y) -> Dict[str, Any]:
    if mode == "broad":
        n_estimators = trial.suggest_int("n_estimators", 300, 1200, step=100)
        max_depth = trial.suggest_categorical("max_depth", [None] + list(range(12, 61, 4)))
        min_samples_split = trial.suggest_int("min_samples_split", 2, 20)
        min_samples_leaf = trial.suggest_int("min_samples_leaf", 1, 12)
        bootstrap = trial.suggest_categorical("bootstrap", [True, False])
        criterion = trial.suggest_categorical("criterion", ["gini", "entropy", "log_loss"])
        max_features = trial.suggest_categorical("max_features", ["sqrt", "log2", 1.0, 0.8, 0.6, 0.5, 0.4])
        safe_weight = trial.suggest_float("safe_weight", 0.3, 1.2)
        unsafe_mult = trial.suggest_float("unsafe_mult", 1.0, 4.0)
    else:
        n_estimators = trial.suggest_int("n_estimators", 600, 900, step=100)
        max_depth = trial.suggest_categorical("max_depth", [None] + list(range(44, 61, 4)))
        min_samples_split = trial.suggest_int("min_samples_split", 2, 6)
        min_samples_leaf = trial.suggest_int("min_samples_leaf", 1, 3)
        bootstrap = trial.suggest_categorical("bootstrap", [False, True])
        criterion = trial.suggest_categorical("criterion", ["gini", "log_loss"])
        max_features = trial.suggest_categorical("max_features", [0.5, 0.6, "sqrt"])
        safe_weight = trial.suggest_float("safe_weight", 0.9, 1.2)
        unsafe_mult = trial.suggest_float("unsafe_mult", 2.5, 3.5)

    params = dict(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        max_features=max_features,
        bootstrap=bootstrap,
        criterion=criterion,
        class_weight=build_class_weights(y, safe_weight, unsafe_mult),
    )
    return params


# Objective function (maximize weighted F1 with safety penalty)
def make_objective(X, y, safety_threshold, val_size, seed, train_subsample, mode):
    X_train, X_valid, y_train, y_valid = train_test_split(
        X, y, test_size=val_size, stratify=y, random_state=seed
    )

    if 0 < train_subsample < 1.0:
        X_train, _, y_train, _ = train_test_split(
            X_train, y_train,
            train_size=train_subsample,
            stratify=y_train,
            random_state=seed
        )

    def objective(trial: optuna.trial.Trial) -> float:
        params = suggest_hparams(trial, mode, y)

        model = RandomForestClassifier(
            **params,
            n_jobs=-1,
            random_state=seed
        )

        model.fit(X_train, y_train)
        y_pred = model.predict(X_valid)

        weighted_f1 = f1_score(y_valid, y_pred, average="weighted")
        safety_rec = compute_safety_recall(y_valid, y_pred)
        safety_fn = compute_safety_false_negatives(y_valid, y_pred)

        trial.set_user_attr("weighted_f1", float(weighted_f1))
        trial.set_user_attr("safety_recall", float(safety_rec))
        trial.set_user_attr("safety_false_negatives", int(safety_fn))

        if safety_rec < safety_threshold:
            penalty = max(1e-6, (safety_rec / safety_threshold) ** 4)
            return weighted_f1 * penalty

        return weighted_f1

    return objective


# Train final best model and evaluate
def train_eval_best(X, y, params, seed, test_size, model_out):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=seed
    )

    model = RandomForestClassifier(**params, n_jobs=-1, random_state=seed)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    metrics = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "f1_weighted": float(f1_score(y_test, y_pred, average="weighted")),
        "safety_recall": float(compute_safety_recall(y_test, y_pred)),
        "safety_false_negatives": int(compute_safety_false_negatives(y_test, y_pred)),
        "report": classification_report(y_test, y_pred, digits=4),
        "confusion_matrix_labels": list(sorted(np.unique(y_test))),
        "confusion_matrix": confusion_matrix(y_test, y_pred, labels=sorted(np.unique(y_test))).tolist(),
    }

    joblib.dump(model, model_out)
    return metrics


# Progress snapshots (optional)
class TrialProgress:
    def __init__(self, snapshot_dir: str):
        self.snapshot_dir = snapshot_dir
        os.makedirs(self.snapshot_dir, exist_ok=True)
        self._last_best = -float("inf")

    def save_trials_csv(self, study: optuna.Study):
        try:
            df = study.trials_dataframe(attrs=("number", "value", "state", "params", "user_attrs"))
            df.to_csv(os.path.join(self.snapshot_dir, "study_trials.csv"), index=False)
        except Exception:
            pass

    def save_best_snapshot(self, study: optuna.Study):
        if not study.best_trial:
            return
        best = study.best_trial
        snap = {
            "timestamp": datetime.utcnow().replace(microsecond=0).isoformat() + "Z",
            "trial_number": best.number,
            "value": best.value,
            "params": best.params,
            "user_attrs": best.user_attrs,
        }
        with open(os.path.join(self.snapshot_dir, "best_so_far.json"), "w") as f:
            json.dump(snap, f, indent=2)

    def __call__(self, study: optuna.Study, trial: optuna.trial.FrozenTrial):
        self.save_trials_csv(study)
        if study.best_value > self._last_best:
            self._last_best = study.best_value
            self.save_best_snapshot(study)


def main():
    parser = argparse.ArgumentParser(description="Optuna hyperparameter optimization for Random Forest")
    parser.add_argument("--pickle", default="dataset/dataset.pkl")
    parser.add_argument("--trials", type=int, default=300)
    parser.add_argument("--timeout", type=int, default=None)
    parser.add_argument("--study-name", default="rf_optuna_focus")
    parser.add_argument("--new-study", action="store_true")
    parser.add_argument("--mode", choices=["focused", "broad"], default="focused")
    parser.add_argument("--storage", default="sqlite:///optuna_rf.db")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val-size", type=float, default=0.2)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--safety-threshold", type=float, default=0.98)
    parser.add_argument("--train-subsample", type=float, default=0.6)
    parser.add_argument("--snapshot-dir", default="snapshots")
    parser.add_argument("--model-out", default="random_forest_final.pkl")
    parser.add_argument("--params-out", default="best_params.json")
    parser.add_argument("--metrics-out", default="best_metrics.json")
    args = parser.parse_args()

    progress_cb = TrialProgress(args.snapshot_dir)

    # 1) Load and prepare data
    with open(args.pickle, "rb") as f:
        dataset = pickle.load(f)

    X = np.vstack([get_eps_features(s) for s in dataset])
    y = np.array([build_label(s) for s in dataset], dtype=object)

    # 2) Create or resume study
    study_name = args.study_name
    load_if_exists = True
    if args.new_study:
        study_name = f"{study_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        load_if_exists = False

    study = optuna.create_study(
        study_name=study_name,
        storage=args.storage,
        load_if_exists=load_if_exists,
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=args.seed, multivariate=True, group=True),
    )

    objective = make_objective(
        X, y,
        safety_threshold=args.safety_threshold,
        val_size=args.val_size,
        seed=args.seed,
        train_subsample=args.train_subsample,
        mode=args.mode,
    )

    # Graceful Ctrl+C: snapshot and exit
    def handle_sigint(signum, frame):
        progress_cb.save_trials_csv(study)
        progress_cb.save_best_snapshot(study)
        sys.exit(130)

    signal.signal(signal.SIGINT, handle_sigint)

    # 3) Run optimization
    study.optimize(
        objective,
        n_trials=args.trials if args.timeout is None else None,
        timeout=args.timeout,
        callbacks=[progress_cb],
        gc_after_trial=True,
        show_progress_bar=False,
    )

    # 4) Train best model + evaluate
    best_params = study.best_trial.params.copy()
    safe_weight = best_params.pop("safe_weight")
    unsafe_mult = best_params.pop("unsafe_mult")
    best_params["class_weight"] = build_class_weights(y, safe_weight, unsafe_mult)
    best_params["random_state"] = args.seed
    best_params["n_jobs"] = -1

    metrics = train_eval_best(
        X, y,
        params=best_params,
        seed=args.seed,
        test_size=args.test_size,
        model_out=args.model_out,
    )

    with open(args.params_out, "w") as f:
        json.dump(best_params, f, indent=2)
    with open(args.metrics_out, "w") as f:
        json.dump(metrics, f, indent=2)

    print("\nBest parameters:")
    print(json.dumps(best_params, indent=2))

    print("\nFinal evaluation metrics:")
    print(json.dumps(metrics, indent=2))

    print(f"\nSaved model: {args.model_out}")
    print(f"Saved params: {args.params_out}")
    print(f"Saved metrics: {args.metrics_out}")


if __name__ == "__main__":
    main()
