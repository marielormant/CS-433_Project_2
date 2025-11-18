import pandas as pd
import numpy as np
import pickle
from pathlib import Path

# Path setup
pkl_path = Path("dataset/dataset.pkl")
output_path = Path("dataset/sample100.csv")

# Load pickle
with open(pkl_path, "rb") as f:
    data = pickle.load(f)

# Take first 1000 samples (or fewer if dataset smaller)
data = data[:1000]

# Flatten structure into a list of rows
rows = []

for i, entry in enumerate(data):
    row = {}
    eps = entry.get("eps_global", [])
    plies = entry.get("plies", {})

    # Add global strain components
    for j, val in enumerate(eps):
        row[f"eps_global_{j+1}"] = float(val)

    # Add all ply failure indices
    for angle, vals in plies.items():
        for key, val in vals.items():
            row[f"{key}_{angle}deg"] = float(val)

    rows.append(row)

# Convert to DataFrame
df = pd.DataFrame(rows)

# Save to CSV
df.to_csv(output_path, index=False)
print(f"Saved first {len(df)} entries to {output_path.resolve()}")
