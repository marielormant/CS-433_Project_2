import pickle
import numpy as np
from pathlib import Path
from collections import Counter

# Load dataset
pkl_path = Path("dataset/dataset.pkl")

with open(pkl_path, "rb") as f:
    data = pickle.load(f)

print("Loaded", len(data), "samples")

# Prepare storage
eps_array = np.zeros((len(data), 6))
labels = []

# Analyze samples quickly
for i, entry in enumerate(data):
    eps = entry["eps_global"]
    plies = entry["plies"]

    # Store eps_global vector
    eps_array[i, :] = eps

    # Compute max FI and failing mode
    max_val = -1
    max_label = "no_failure"

    for angle, vals in plies.items():
        for mode, val in vals.items():
            v = float(val)
            if v > max_val:
                max_val = v
                max_label = f"{angle}_{mode}"

    if max_val <= 1:
        labels.append("no_failure")
    else:
        labels.append(max_label)

# Convert to numpy for fast numeric ops
labels = np.array(labels)

# === Summary Statistics ===

print("\n--- EPSILON STATS ---")
for i in range(6):
    col = eps_array[:, i]
    print(
        f"eps_global_{i+1}: min={col.min():.4f}, "
        f"mean={col.mean():.4f}, max={col.max():.4f}"
    )

print("\n--- FAILURE DISTRIBUTION ---")
counts = Counter(labels)

total = len(data)
failures = total - counts["no_failure"]
fail_pct = 100 * failures / total

print("Total samples:", total)
print("Failures:", failures, f"({fail_pct:.2f}%)")
print("No failure:", counts['no_failure'])

print("\n--- FAILING MODE COUNTS ---")
for label, count in counts.items():
    if label != "no_failure":
        print(f"{label}: {count}")
