
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler

# %% Import data
with open("dataset/dataset.pkl", "rb") as f:
    data = pickle.load(f)

n = len(data)
nbins = int(1 + 3.322 * np.log10(n))
# %% Define strain and plies FI values

eps = pd.DataFrame(
    [d["eps_global"] for d in data],
    columns=["11", "22", "33", "23", "13", "12"],
)

print("Strain values")
print(eps.head().round(4))

# %%
scaler = MinMaxScaler(feature_range=(-1, 1))
eps_scaled = eps.copy()
eps_scaled.iloc[:, :] = scaler.fit_transform(eps.values)

print("Strain values scaled")
print(eps_scaled.head().round(2))
# %%
plies = {}
angles = [0.0, 45.0, 90.0, -45.0]

for angle in angles:
    plies[angle] = pd.DataFrame(
        [d["plies"][angle] for d in data],
    )

print("Plies")
print(list(plies.keys()))
# %%
print("Failure index values for ply 0")
print(list(plies.values())[0].head().round(2))

# %% Faiure summary

stacked = pd.concat(plies, axis=1)
stacked.columns = [f"{angle}_{mode}" for angle, mode in stacked.columns]

max_val = stacked.max(axis=1)
max_col = stacked.idxmax(axis=1)
split = max_col.str.split("_", expand=True)
max_angle = split[0].astype(float)
short_mode = split[2]

values = stacked.to_numpy()
cols = stacked.columns.get_indexer(max_col)
FI = values[np.arange(len(values)), cols]

fail_threshold = 1.0
ffp = np.where(max_val >= fail_threshold, max_angle, np.nan)
mode = np.where(max_val >= fail_threshold, short_mode, "nf")

fail_summary = pd.DataFrame(
    {
        "ffp": np.where(np.isnan(ffp), "none", ffp),
        "mode": mode,
        "FI": FI,
    }
)

print("Failure summary")
print(fail_summary.sample(10, random_state=21))
print(len())