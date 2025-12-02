import pickle
import pandas as pd
import numpy as np

def pickle_to_csv(pickle_path: str, csv_path: str = "dataset.csv"):
    """
    Convert dataset pickle -> CSV using keys:
      - eps_global: numpy array of length 6
      - plies: dict with ply angles (e.g., 0.0, 45.0, -45.0, 90.0)
               each mapping to a dict like {"FI_ft": val, "FI_fc": val, "FI_mt": val, "FI_mc": val}
    Columns produced:
      eps_0..eps_5 and FI_ply{ply}_{ft|fc|mt|mc}
    """
    print(f"Loading pickle: {pickle_path}")
    with open(pickle_path, "rb") as f:
        data = pickle.load(f)

    print("Pickle loaded. Flattening to rows...")
    rows = []

    for sample in data:
        row = {}

        # --- global strain vector ---
        eps = sample.get("eps_global", None)
        if isinstance(eps, np.ndarray):
            eps = eps.tolist()
        if isinstance(eps, (list, tuple)) and len(eps) >= 6:
            for i in range(6):
                row[f"eps_{i}"] = eps[i]
        else:
            # ensure columns exist even if something's off
            for i in range(6):
                row[f"eps_{i}"] = None

        # --- ply × failure modes ---
        plies = sample.get("plies", {})
        if isinstance(plies, dict):
            for ply, modes in plies.items():
                # normalize ply label (e.g., 0.0 -> "0", 45.0 -> "45", -45.0 -> "-45")
                if isinstance(ply, (int, float)) and float(ply).is_integer():
                    ply_label = str(int(ply))
                else:
                    ply_label = str(ply)

                if isinstance(modes, dict):
                    for mode_key, value in modes.items():
                        # mode_key is like "FI_ft", "FI_fc", ...
                        mode_short = mode_key.split("_", 1)[1] if mode_key.startswith("FI_") else mode_key
                        col = f"FI_ply{ply_label}_{mode_short}"
                        # unwrap numpy scalar if present
                        if hasattr(value, "item"):
                            try:
                                value = value.item()
                            except Exception:
                                pass
                        row[col] = value

        rows.append(row)

    print("Creating DataFrame...")
    df = pd.DataFrame(rows)

    # Optional: order columns (eps first, then FI_*)
    eps_cols = [f"eps_{i}" for i in range(6)]
    fi_cols = sorted([c for c in df.columns if c.startswith("FI_ply")])
    other = [c for c in df.columns if c not in eps_cols + fi_cols]
    df = df[eps_cols + fi_cols + other]

    print(f"Saving CSV to: {csv_path} (this may take a while for 1,000,000 rows)...")
    df.to_csv(csv_path, index=False)
    print("Done.")
    print("Head of epsilon columns:\n", df[[f'eps_{i}' for i in range(6)]].head())

    return df

if __name__ == "__main__":
    df = pickle_to_csv("dataset/dataset.pkl", "dataset/dataset.csv")
