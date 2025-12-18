"""
Random Forest Weighted Model (class weighting, no hyperparameter optimization)
- Inputs: 6 epsilon strain features
- Labels: 17 classes (16 ply-mode failure classes + "safe")
- Split: 80/20 train/test (stratified)
- Metrics: Classification report (including macro-F1, weighted F1, Accuracy), False Negatives (FN)
- Saves: random_forest_weighted.pkl
"""

import pickle
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import joblib


# Helper functions (data + label construction)
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


# 1) Load and prepare data
with open("dataset/dataset.pkl", "rb") as f:
    dataset = pickle.load(f)

X = np.vstack([get_eps_features(s) for s in dataset])
y = np.array([build_label(s) for s in dataset], dtype=object)

# 2) Train / test split (80/20 stratified)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 3) Train weighted Random Forest
classes = np.unique(y_train)
class_weights = {"safe": 0.5, **{cls: 2.0 for cls in classes if cls != "safe"}}

model = RandomForestClassifier(
    n_estimators=600,
    class_weight=class_weights,
    min_samples_leaf=2,
    min_samples_split=4,
    n_jobs=-1,
    random_state=42
)

model.fit(X_train, y_train)

# 4) Evaluation
y_pred = model.predict(X_test)

# Safety false negatives (FN): unsafe predicted as safe
unsafe_mask = (y_test != "safe")
false_negatives = np.sum(y_pred[unsafe_mask] == "safe")

print("\nClassification report:")
print(classification_report(y_test, y_pred, digits=4))

print(f"Safety false negatives (unsafe predicted safe): {false_negatives}")

# 5) Save model
joblib.dump(model, "random_forest_weighted.pkl")
print("Saved: random_forest_weighted.pkl")
