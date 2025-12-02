import time
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
    f1_score,
)
import joblib
"""
DIFFERENCE WITH 3: THRESHOLD
"""

"""
RandomForest_safety_thresholds.py
---------------------------------
Task 4 classifier with safety-focused evaluation.

Assumes you already have:
    dataset/dataset_with_labels.csv

Columns:
    - eps_0..eps_5               (features used)
    - FI_ply{ply}_{mode}         (not used as features)
    - label_17                   (target: "safe" OR "<ply>_<mode>")

What this script does:
    1) Train a RandomForest on epsilon-only features (once)
    2) Compute predict_proba (once)
    3) Sweep decision thresholds for P(safe) from 0.50 to 0.90 (step 0.01)
       For each threshold, compute:
           - Accuracy
           - Safety False Negatives (true!=safe AND pred==safe)
           - Safety Recall = (# unsafe predicted unsafe) / (# unsafe)
           - Macro-F1 (all 17 classes)
    4) Print a table of metrics for all thresholds
    5) Pick the BEST threshold (max Safety Recall, then Macro-F1, then Accuracy)
       and print a classification report + confusion matrix at that threshold.
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


def predict_with_safe_threshold(classes, proba, safe_threshold):
    """
    Convert class probabilities into predicted labels with a custom 'safe' threshold.

    Rule:
      if P(safe) >= safe_threshold -> predict "safe"
      else -> choose the most probable *unsafe* class (argmax over non-safe classes)
    """
    classes = np.array(classes)
    safe_idx = list(classes).index("safe")

    # build predictions
    y_pred = []
    for p in proba:
        if p[safe_idx] >= safe_threshold:
            y_pred.append("safe")
        else:
            up = p.copy()
            up[safe_idx] = -1.0  # mask out "safe" so an unsafe class is chosen
            y_pred.append(classes[np.argmax(up)])
    return np.array(y_pred)


def train_and_evaluate_with_threshold_sweep(
    csv_path="dataset/dataset_with_labels.csv",
    model_out="random_forest_task4.pkl",
    use_weighting=True,
    thresholds=np.arange(0.50, 0.901, 0.01),  # 0.50..0.90 step 0.01
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

    # Weighted vs baseline RF settings
    if use_weighting:
        class_w = {"safe": 0.5, **{cls: 2.0 for cls in y.unique() if cls != "safe"}}
        n_estimators = 600
        min_leaf = 2
        min_split = 4
    else:
        class_w = None
        n_estimators = 300
        min_leaf = 1
        min_split = 2

    clf = RandomForestClassifier(
        n_estimators=n_estimators,
        class_weight=class_w,
        min_samples_leaf=min_leaf,
        min_samples_split=min_split,
        n_jobs=-1,
        random_state=42,
    )

    t2 = time.time()
    print("\nTraining Random Forest...")
    clf.fit(X_train, y_train)
    print(f"[TIMER] Fit model — {time.time() - t2:.3f}s")

    # One proba pass for all thresholds
    t3 = time.time()
    proba = clf.predict_proba(X_test)
    classes = clf.classes_
    print(f"[TIMER] Computed predict_proba — {time.time() - t3:.3f}s\n")

    # Evaluate all thresholds
    rows = []
    for thr in thresholds:
        y_pred_thr = predict_with_safe_threshold(classes, proba, thr)

        acc = accuracy_score(y_test, y_pred_thr)
        sfn = compute_safety_false_negatives(y_test, y_pred_thr)
        srec = compute_safety_recall(y_test, y_pred_thr)
        mf1 = f1_score(y_test, y_pred_thr, average="macro")

        rows.append(
            {
                "threshold": round(float(thr), 2),
                "accuracy": acc,
                "safety_FN": sfn,
                "safety_recall": srec,
                "macro_f1": mf1,
            }
        )

    results = pd.DataFrame(rows)

    # Pretty print the table
    print("=== Threshold Sweep (P(safe) >= threshold → predict 'safe') ===")
    print(f"{'thr':>5} | {'acc':>7} | {'safety_FN':>10} | {'safety_recall':>13} | {'macro_f1':>8}")
    print("-" * 58)
    for r in results.itertuples(index=False):
        print(f"{r.threshold:5.2f} | {r.accuracy:7.4f} | {r.safety_FN:10d} | {r.safety_recall:13.4f} | {r.macro_f1:8.4f}")

    # Choose best threshold:
    #   1) maximize safety_recall
    #   2) tie-break by macro_f1
    #   3) tie-break by accuracy
    best = results.sort_values(
        by=["safety_recall", "macro_f1", "accuracy"], ascending=[False, False, False]
    ).iloc[0]

    print("\n>>> Best threshold selected:", f"{best.threshold:.2f}")
    print("Metrics at best threshold → "
          f"Accuracy: {best.accuracy:.4f} | Safety FN: {int(best.safety_FN)} | "
          f"Safety Recall: {best.safety_recall:.4f} | Macro-F1: {best.macro_f1:.4f}\n")

    # Detailed report at best threshold
    y_pred_best = predict_with_safe_threshold(classes, proba, best.threshold)

    print("Classification Report @ best threshold:")
    print(classification_report(y_test, y_pred_best))

    cls_order = np.array(sorted(y_test.unique()))
    cm = confusion_matrix(y_test, y_pred_best, labels=cls_order)
    print("Confusion Matrix (rows=true, cols=pred) with label order:")
    print(list(cls_order))
    print(cm)

    # Save model
    joblib.dump(clf, model_out)
    print(f"\nModel saved as: {model_out}")

    return clf, results


if __name__ == "__main__":
    # Toggle weighting here if you want the baseline:
    # clf, results = train_and_evaluate_with_threshold_sweep(use_weighting=False)

    clf, results = train_and_evaluate_with_threshold_sweep(use_weighting=True)
