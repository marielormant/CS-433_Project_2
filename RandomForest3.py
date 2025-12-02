import time
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib

"""
RandomForest_safety.py
----------------------
Task 4 classifier with safety-focused metrics.

This script assumes you have already created:
    dataset/dataset_with_labels.csv
which contains:
    - eps_0..eps_5 : global laminate strain (R^6)
    - FI_ply{ply}_{mode} : ply-level failure indices (not used as features)
    - label_17 : 17-class label ("safe" OR "<ply>_<mode>")

Model:
    - Features X = eps_0..eps_5 (epsilon only, Task-4 compliant)
    - Target y  = label_17
    - Stratified 80/20 train/test
    - RandomForestClassifier

Safety metrics (per TA):
    - Safety False Negatives  = count of (true != "safe" AND pred == "safe")
    - Safety Recall           = fraction of unsafe correctly predicted as unsafe
"""

# ---------- Safety metrics (per TA) ----------
def compute_safety_false_negatives(y_true, y_pred):
    """
    Count catastrophic errors:
      true label = unsafe ( != "safe")
      predicted  = "safe"
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    unsafe_mask = (y_true != "safe")
    return int(np.sum(y_pred[unsafe_mask] == "safe"))

def compute_safety_recall(y_true, y_pred):
    """
    Safety Recall = (# unsafe predicted as unsafe) / (# unsafe)
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    unsafe_mask = (y_true != "safe")
    if unsafe_mask.sum() == 0:
        return 1.0
    return float(np.mean(y_pred[unsafe_mask] != "safe"))


def train_random_forest(
    csv_path="dataset/dataset_with_labels.csv",
    model_out="random_forest_task4.pkl",
    use_weighting=False,
):
    t0 = time.time()
    print("\nLoading labeled dataset...")
    df = pd.read_csv(csv_path)
    print(f"[TIMER] Loaded labeled dataset — {time.time() - t0:.3f}s")

    # ε-only features (Task 4 requirement)
    feature_cols = [c for c in df.columns if c.startswith("eps_")]
    X = df[feature_cols]
    y = df["label_17"]

    print(f"Total features: {len(feature_cols)}")
    print(f"Total samples:  {len(df)}")

    t1 = time.time()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    print(f"[TIMER] Train/test split — {time.time() - t1:.3f}s")

    # Random Forest (optionally safety-weighted)
    if use_weighting:
        # down-weight "safe", up-weight all unsafe classes
        class_w = {"safe": 0.5, **{cls: 2.0 for cls in y.unique() if cls != "safe"}}
    else:
        class_w = None

    clf = RandomForestClassifier(
        n_estimators=300 if not use_weighting else 600,
        class_weight=class_w,
        min_samples_leaf=1 if not use_weighting else 2,
        min_samples_split=2 if not use_weighting else 4,
        n_jobs=-1,
        random_state=42,
    )

    t2 = time.time()
    print("\nTraining Random Forest...")
    clf.fit(X_train, y_train)
    print(f"[TIMER] Fit model — {time.time() - t2:.3f}s")

    # Standard predictions (no threshold moving here)
    t3 = time.time()
    y_pred = clf.predict(X_test)
    print(f"[TIMER] Generated predictions — {time.time() - t3:.3f}s\n")

    # ----- Standard metrics -----
    acc = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {acc:.4f}\n")

    print("Classification Report:")
    print(classification_report(y_test, y_pred))

    # Show confusion matrix with explicit class order
    classes = np.array(sorted(y_test.unique()))
    cm = confusion_matrix(y_test, y_pred, labels=classes)
    print("Confusion Matrix (rows = true, cols = pred) with label order:")
    print(list(classes))
    print(cm)

    # ----- Safety metrics (TA focus) -----
    total_safety_fn = compute_safety_false_negatives(y_test, y_pred)
    safety_rec = compute_safety_recall(y_test, y_pred)
    print("\nSafety False Negatives (true!=safe, pred==safe):", total_safety_fn)
    print("Safety Recall (unsafe predicted unsafe):        {:.4f}".format(safety_rec))

    joblib.dump(clf, model_out)
    print(f"\nModel saved as: {model_out}")

    return clf


if __name__ == "__main__":
    # If you want the weighted version, set use_weighting=True
    # Baseline (no weighting):
    # train_random_forest(use_weighting=False)

    # Safety-weighted:
    train_random_forest(use_weighting=True)
