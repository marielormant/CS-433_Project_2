import time
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib

"""
This script performs two main steps:

1. Build 17-class labels from the dataset
   --------------------------------------
   The input CSV (dataset.csv) contains:
     - eps_0..eps_5 : global laminate strain vector (R^6)
     - FI_ply{ply}_{mode} : ply-level failure indices produced by the simulator
                            for ply angles (-45, 0, 45, 90) and failure modes
                            (ft, fc, mt, mc).

   For each sample, the max FI across all plies and modes is found.
     * If max FI <= 1 → label = "safe"
     * Otherwise      → label = "<ply>_<mode>"   (e.g., "45_mt", "-45_fc")

    17-class label: {16 ply–mode failure classes} + {"safe"}.

   The labeled dataset is saved as dataset_with_labels.csv.

2. Train a Random Forest classifier
   --------------------------------
   Feature matrix X uses:
       eps_0, eps_1, eps_2, eps_3, eps_4, eps_5

   The failure labels y = label_17.

   A stratified 80/20 train–test split is applied.
   A RandomForestClassifier is trained and evaluated using:
       - overall accuracy
       - per-class precision/recall/F1-score
       - confusion matrix
       - total false negatives (across all classes)

   The trained model is saved to: random_forest_task4.pkl

Usage:
   python RandomForest.py

Outputs:
   - dataset/dataset_with_labels.csv    (with the 17-class labels)
   - random_forest_task4.pkl            (trained ML model)
   - printed performance metrics and step-by-step timings
"""

class StepTimer:
    """Lightweight timer to log per-step and total elapsed time."""
    def __init__(self):
        self._t0 = time.perf_counter()
        self._last = self._t0

    def tick(self, label: str):
        now = time.perf_counter()
        step = now - self._last
        total = now - self._t0
        print(f"[TIMER] {label} — step: {step:.3f}s | total: {total:.3f}s")
        self._last = now


def add_labels(csv_in="dataset/dataset.csv",
               csv_out="dataset/dataset_with_labels.csv"):
    timer = StepTimer()

    print("Loading dataset...")
    df = pd.read_csv(csv_in)
    timer.tick("Loaded dataset")

    fi_cols = [c for c in df.columns if c.startswith("FI_ply")]
    print(f"Detected {len(fi_cols)} FI columns.")

    labels = []

    # Build labels
    for _, row in df.iterrows():
        fi_values = row[fi_cols]
        max_fi = fi_values.max()
        max_col = fi_values.idxmax()

        if max_fi <= 1.0:
            labels.append("safe")
        else:
            _, ply_mode = max_col.split("FI_ply")
            ply, mode = ply_mode.split("_")
            labels.append(f"{ply}_{mode}")

    df["label_17"] = labels
    timer.tick("Built labels")

    print("Saving dataset with labels...")
    df.to_csv(csv_out, index=False)
    print(f"Saved: {csv_out}")
    timer.tick("Saved labeled dataset")

    return df


def train_random_forest(csv_path="dataset/dataset_with_labels.csv",
                        model_out="random_forest_task4.pkl"):
    timer = StepTimer()

    print("\nLoading labeled dataset...")
    df = pd.read_csv(csv_path)
    timer.tick("Loaded labeled dataset")

    feature_cols = [c for c in df.columns if c.startswith("eps_")]
    X = df[feature_cols]
    y = df["label_17"]

    print(f"Total features: {len(feature_cols)}")
    print(f"Total samples: {len(df)}")

    # 80/20 train-test split with stratification
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )
    timer.tick("Performed train/test split")

    print("\nTraining Random Forest...")
    clf = RandomForestClassifier(
        n_estimators=600,
        class_weight={"safe": 0.5, **{cls: 2.0 for cls in y.unique() if cls != "safe"}},
        min_samples_leaf=2,
        min_samples_split=4,
        n_jobs=-1,
        random_state=42
    )
    clf.fit(X_train, y_train)
    timer.tick("Fit model")

    print("\nEvaluating model...")
    y_pred = clf.predict(X_test)
    timer.tick("Generated predictions")

    acc = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {acc:.4f}\n")
    timer.tick("Computed accuracy")

    print("Classification Report:")
    print(classification_report(y_test, y_pred))
    timer.tick("Printed classification report")

    # Confusion matrix with explicit, consistent label ordering
    ordered_labels = sorted(y.unique())
    cm = confusion_matrix(y_test, y_pred, labels=ordered_labels)
    print("Confusion Matrix (rows = true, cols = pred) with label order:")
    print(ordered_labels)
    print(cm)
    timer.tick("Computed confusion matrix")

    # ---- Total False Negatives (across all classes) ----
    # For multiclass, total FN equals total misclassified samples:
    # sum of all entries minus sum of diagonal (true positives).
    total_false_negatives = int(cm.sum() - np.trace(cm))
    print(f"\nTotal False Negatives (across all classes): {total_false_negatives}")
    timer.tick("Computed total false negatives")

    joblib.dump(clf, model_out)
    print(f"\nModel saved as: {model_out}")
    timer.tick("Saved model")

    return clf


if __name__ == "__main__":
    print("=== Building 17-class labels ===")
    add_labels()

    print("\n=== Training Random Forest Classifier ===")
    train_random_forest()
