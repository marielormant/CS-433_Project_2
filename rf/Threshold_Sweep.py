"""
Safe-Threshold Sweep for Random Forest Final Model
- Inputs: 6 epsilon strain features
- Labels: 17 classes (16 ply–mode failure classes + "safe")
- Split: 80/20 train/test (stratified)
- Method: Predict "safe" only if P(safe) >= τ, otherwise predict best non-safe class
- Metrics: Accuracy, Macro-F1, Safety recall, False Negatives (FN)
- Saves: accuracy_macroF1_safetyRecall_vs_threshold.png
"""

import pickle
import numpy as np
import matplotlib.pyplot as plt
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score


# Safety metrics
def compute_safety_recall(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=object)
    y_pred = np.asarray(y_pred, dtype=object)
    unsafe_mask = (y_true != "safe")
    if unsafe_mask.sum() == 0:
        return 1.0
    return float(np.mean(y_pred[unsafe_mask] != "safe"))


def compute_safety_false_negatives(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=object)
    y_pred = np.asarray(y_pred, dtype=object)
    unsafe_mask = (y_true != "safe")
    return int(np.sum(y_pred[unsafe_mask] == "safe"))


def predict_with_safe_threshold(classes, proba, tau):
    classes = np.asarray(classes, dtype=object)
    safe_idx = list(classes).index("safe")

    preds = []
    for p in proba:
        if p[safe_idx] >= tau:
            preds.append("safe")
        else:
            q = p.copy()
            q[safe_idx] = -1.0
            preds.append(classes[np.argmax(q)])
    return np.asarray(preds, dtype=object)


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


# 1) Load model and data
model = joblib.load("random_forest_final.pkl")

with open("dataset/dataset.pkl", "rb") as f:
    dataset = pickle.load(f)

X = np.vstack([get_eps_features(s) for s in dataset])
y = np.array([build_label(s) for s in dataset], dtype=object)

# 2) Train / test split (80/20 stratified)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 3) Predict probabilities once
proba = model.predict_proba(X_test)
classes = model.classes_

# 4) Sweep thresholds
thresholds = np.arange(0.50, 0.951, 0.01)

acc_list = []
f1_list = []
safety_recall_list = []
fn_list = []

print("\ntau  |  Acc     MacroF1  SafetyRec   FN")
print("----------------------------------------------")

for tau in thresholds:
    y_pred_tau = predict_with_safe_threshold(classes, proba, tau)

    acc = accuracy_score(y_test, y_pred_tau)
    mf1 = f1_score(y_test, y_pred_tau, average="macro")
    srec = compute_safety_recall(y_test, y_pred_tau)
    sfn = compute_safety_false_negatives(y_test, y_pred_tau)

    acc_list.append(acc)
    f1_list.append(mf1)
    safety_recall_list.append(srec)
    fn_list.append(sfn)

    print(f"{tau:>4.2f} |  {acc:.4f}   {mf1:.4f}   {srec:.4f}   {sfn}")

# 5) Plot metrics
fig, ax1 = plt.subplots(figsize=(12, 7))

ax1.plot(thresholds, acc_list, marker="o", label="Accuracy", linewidth=2)
ax1.plot(thresholds, f1_list, marker="s", label="Macro-F1", linewidth=2)

ax1.set_xlabel("Threshold τ", fontsize=16)
ax1.set_ylabel("Accuracy / Macro-F1", fontsize=16)
ax1.tick_params(axis="both", labelsize=13)
ax1.grid(True)

ax2 = ax1.twinx()
ax2.plot(thresholds, safety_recall_list, marker="^", color="red", label="Safety Recall", linewidth=2)
ax2.set_ylabel("Safety Recall", fontsize=16)
ax2.tick_params(axis="y", labelsize=13)

plt.title("Accuracy, Macro-F1, and Safety Recall vs Threshold τ", fontsize=18, pad=12)

lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(
    lines1 + lines2,
    labels1 + labels2,
    loc="lower left",
    bbox_to_anchor=(0.12, 0.02),
    fontsize=13,
    frameon=True
)

plt.tight_layout()
plt.savefig("accuracy_macroF1_safetyRecall_vs_threshold.png", dpi=150)
print("Saved: accuracy_macroF1_safetyRecall_vs_threshold.png")

plt.show()
