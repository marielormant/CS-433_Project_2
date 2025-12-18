import matplotlib.pyplot as plt
import numpy as np
from adjustText import adjust_text

# Data
FI = [
    "NOFAIL","FT0","FC0","MT0","MC0",
    "FT45","FC45","MT45","MC45",
    "FT90","FC90","MT90","MC90",
    "FT-45","FC-45","MT-45","MC-45"
]

f1 = np.array([
    0.96,0.95,0.98,0.95,0.95,
    0.87,0.99,0.87,0.82,
    0.95,0.99,0.95,0.95,
    0.87,0.99,0.89,0.81
])

support = np.array([
    41991,17590,22305,10376,11441,
    5942,4741,4041,2599,
    17544,22360,10315,11396,
    5893,4802,4059,2605
])

# ---- Trendline (log-log regression) ----
log_f1 = np.log10(f1)
log_support = np.log10(support)

a, b = np.polyfit(log_f1, log_support, 1)

# Smooth x for trendline
f1_fit = np.linspace(0.8, 1.0, 200)
support_fit = 10 ** (a * np.log10(f1_fit) + b)

# ---- Plot ----
plt.figure()
plt.loglog(f1, support, 'o', markersize=4, label="Failure classes")

# Trendline
plt.loglog(f1_fit, support_fit, '-', linewidth=1.5,
           label=f"Trendline")

# Axis limits
plt.xlim(0.8, 1.0)
plt.ylim(2000, 50000)

# Ticks
plt.xticks([0.80, 0.85, 0.90, 0.95, 1.00])
plt.yticks([2000, 5000, 10000, 20000, 50000],
           ["2k", "5k", "10k", "20k", "50k"])

# Labels
plt.xlabel("F1 score")
plt.ylabel("Support")

# Data labels
texts = [plt.text(xi, yi, label, fontsize=8)
         for xi, yi, label in zip(f1, support, FI)]

adjust_text(
    texts,
    expand_points=(1.3, 1.3),
    expand_text=(1.2, 1.2)
)

# Grid + legend
plt.grid(True, which="both")
plt.legend()

plt.tight_layout()
plt.show()
