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

    print("Loading dataset...")  # running info
    df = pd.read_csv(csv_in)
    timer.tick("Loaded dataset")

    fi_cols = [c for c in df.columns if c.startswith("FI_ply")]  # Define FI columns
    print(f"Detected {len(fi_cols)} FI columns.")

    labels = []

    for _, row in df.iterrows():
        fi_values = row[fi_cols]
        max_fi = fi_values.max()
        max_col = fi_values.idxmax()

        if max_fi <= 1.0:
            labels.append("safe")
        else:
            _, ply_mode = max_col.split("FI_ply")  # extracts ply and mode
            ply, mode = ply_mode.split("_")
            labels.append(f"{ply}_{mode}")  # Column format: FI_ply{ply}_{mode}

    df["label_17"] = labels
    timer.tick("Built labels")

    print("Saving dataset with labels...")  # running info
    df.to_csv(csv_out, index=False)
    print(f"Saved: {csv_out}")
    timer.tick("Saved labeled dataset")

    return df


def train_random_forest(csv_path="dataset/dataset_with_labels.csv",
                        model_out="random_forest_task4.pkl"):

    timer = StepTimer()

    print("\nLoading labeled dataset...")  # running info
    df = pd.read_csv(csv_path)
    timer.tick("Loaded labeled dataset")

    feature_cols = [c for c in df.columns if c.startswith("eps_")]  # Features = eps columns
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

    print("\nTraining Random Forest...")  # running info
    clf = RandomForestClassifier(
        n_estimators=300,
        n_jobs=-1,
        random_state=42
    )
    clf.fit(X_train, y_train)
    timer.tick("Fit model")

    print("\nEvaluating model...")  # running info
    y_pred = clf.predict(X_test)
    timer.tick("Generated predictions")

    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}\n")
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
    # For multiclass, the sum of false negatives over all classes equals the number of misclassified samples:
    # total_off_diagonal = cm.sum() - np.trace(cm)
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


"""
Results of this first, unoptimized code (before adding time and flase neg count):


Accuracy: 0.9379

Classification Report:
              precision    recall  f1-score   support

      -45_fc       0.98      0.93      0.96      4802
      -45_ft       0.88      0.79      0.84      5893
      -45_mc       0.90      0.68      0.78      2605
      -45_mt       0.91      0.77      0.84      4059
        0_fc       0.95      0.99      0.97     22305
        0_ft       0.94      0.96      0.95     17590
        0_mc       0.94      0.94      0.94     11441
        0_mt       0.93      0.92      0.93     10376
       45_fc       0.98      0.92      0.95      4741
       45_ft       0.89      0.80      0.84      5942
       45_mc       0.91      0.67      0.77      2599
       45_mt       0.92      0.77      0.84      4041
       90_fc       0.95      0.99      0.97     22360
       90_ft       0.94      0.96      0.95     17544
       90_mc       0.94      0.94      0.94     11396
       90_mt       0.93      0.92      0.93     10315
        safe       0.93      0.98      0.96     41991

    accuracy                           0.94    200000
   macro avg       0.93      0.88      0.90    200000
weighted avg       0.94      0.94      0.94    200000

Confusion Matrix:
[[ 4482     0     0     0    80     0     0     0     0     0     1     0
     79     0     0     0   160]
 [    0  4682    11    16    70   276    55    37     0     5    14    44
     84   262    50    29   258]
 [    0    12  1778    61   133    22    86    19     7    22    11     2
    130     8    76    17   221]
 [    0    66    42  3127    80    79     1   114     0    71     6    19
     66    82     2    89   215]
 [   44     2     3     3 22044     0    55     0    37     1     3     2
      0    58     0     0    53]
 [    0   137     4     3     0 16830     0     3     0   142     8     8
     58     2   131   198    66]
 [    0    14    28     2   119     0 10811   108     0     9    20     1
      9    68     1     0   251]
 [    0    47     5    46     9     5    85  9566     0    37     6    60
     32   219     0     3   256]
 [    0     0     2     0    97     0     0     0  4367     0     0     0
    103     0     0     0   172]
 [    0     6    11    50    92   258    40    44     0  4748    16    15
     79   246    41    23   273]
 [    2    45     3     1   131    11    76    21     0     5  1743    50
    145    19    89    22   236]
 [    0    93     4    28    69    77     3    86     0    50    33  3106
     71    75     3   126   217]
 [   39     1     5     6     1    69     0     0    29     1     6     3
 [   39     1     5     6     1    69     0     0    29     1     6     3
  22074     0    49     1    76]
 [    0   109     3    12    72     4   122   217     0   128     6     7
      0 16799     0     2    63]
      0 16799     0     2    63]
 [    0    11    28     2    12    54     1     1     1    12    17     3
    145     0 10708   127   274]
 [    0    41    19    56    27   202     0     3     0    43     8    55
     15    10    75  9494   267]
 [    9    38    22    10    49    45   145    52     7    34    15    17
     52    69   168    41 41218]]

Model saved as: random_forest_task4.pkl
"""
