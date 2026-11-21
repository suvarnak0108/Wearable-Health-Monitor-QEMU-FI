import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 10,
    "axes.titlesize": 10.5,
    "axes.labelsize": 9.5,
    "figure.dpi": 300
})

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.6, 3.6))

# Panel (a): Compound Multi-Fault Stress Performance
scenarios = ["Noise +\nMissing", "Burst +\nStale", "Noise +\nDrift", "Triple\nCompound"]
f1_vals = [100.00, 98.68, 100.00, 100.00]
rec_vals = [100.00, 100.00, 100.00, 100.00]

x = np.arange(len(scenarios))
w = 0.34
ax1.bar(x - w/2, rec_vals, width=w, color="#2b5c8f", label="Recall (%)")
ax1.bar(x + w/2, f1_vals, width=w, color="#1b9e77", label="F1-Score (%)")
for i, v in enumerate(f1_vals):
    ax1.text(i + w/2 + 0.05, v + 0.3, f"{v:.2f}%", ha="center", va="bottom", fontsize=7.6)
ax1.set_xticks(x)
ax1.set_xticklabels(scenarios)
ax1.set_ylim(90, 103.5)
ax1.set_ylabel("Classification Score (%)")
ax1.set_title("(a) Concurrent Compound Multi-Fault Stress Results")
ax1.grid(True, axis="y", linestyle="--", alpha=0.45)
ax1.legend(loc="lower right", fontsize=8.2)

# Panel (b): Exact Cortex-M4 Compiled Footprint (Total = 2,312 B)
modules = [
    "ECG Biquad Filter",
    "Windowed HR Est.",
    "Adaptive R-Peak",
    "5-Stage Classifier",
    "Active HR Recovery",
    "Control & Semihost"
]
sizes = [128, 168, 368, 376, 440, 832]
colors = ["#4e79a7", "#59a14f", "#f28e2b", "#e15759", "#76b7b2", "#bab0ac"]

y = np.arange(len(modules))
bars = ax2.barh(y, sizes, color=colors, height=0.58)
for bar, sz in zip(bars, sizes):
    ax2.text(sz + 18, bar.get_y() + bar.get_height()/2, f"{sz} B", va="center", fontsize=8.2)
ax2.set_yticks(y)
ax2.set_yticklabels(modules)
ax2.set_xlim(0, 1020)
ax2.set_xlabel("ARM Thumb-2 Code Size (Bytes) — Total: 2,312 B")
ax2.set_title("(b) Cortex-M4 Compiled Module Footprint (2,312 B)")
ax2.grid(True, axis="x", linestyle="--", alpha=0.45)

plt.tight_layout()
plt.savefig("results/fig5_compound_and_footprint.png", dpi=300)
plt.close()
print("Saved updated results/fig5_compound_and_footprint.png (Total = 2,312 B)!")
