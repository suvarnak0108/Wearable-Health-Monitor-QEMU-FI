import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 10,
    "axes.titlesize": 10.5,
    "axes.labelsize": 9.5,
    "figure.dpi": 300
})

scen_df = pd.read_csv("results/qemu_clinical_scenario_summary.csv")
order = ["clean_clinical", "burst_stale_clinical", "noise_020_clinical", "triple_compound_clinical"]
scen_df = scen_df.set_index("Scenario").loc[order].reset_index()
labels = ["Clean\nClinical", "Burst +\nStale", "0.20 mV\nNoise", "Triple\nCompound"]

full_df = pd.read_csv("results/qemu_40_clinical_trials.csv")
trip_df = full_df[full_df["Scenario"] == "triple_compound_clinical"].reset_index(drop=True)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.6, 3.6))

# Panel (a): Raw vs Mitigated HR MAE across 4 Clinical Scenarios
x = np.arange(len(labels))
w = 0.34
b1 = ax1.bar(x - w/2, scen_df["Raw_HR_MAE"], width=w, color="#d95f02", label="Unmitigated HR MAE (BPM)")
b2 = ax1.bar(x + w/2, scen_df["Mitigated_HR_MAE"], width=w, color="#1b9e77", label="Fault-Tolerant Mitigated MAE (BPM)")
for bar in b2:
    h = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2, h + 0.15, f"{h:.2f}", ha="center", va="bottom", fontsize=8)
ax1.set_xticks(x)
ax1.set_xticklabels(labels)
ax1.set_ylim(0, 7.4)
ax1.set_ylabel("Mean Absolute Error (BPM)")
ax1.set_title("(a) Clinical HR MAE: Unmitigated vs. Active Recovery")
ax1.grid(True, axis="y", linestyle="--", alpha=0.45)
ax1.legend(loc="upper left", fontsize=8.2)

# Panel (b): Per-Patient Triple Compound Recovery across 10 MIT-BIH Records
x2 = np.arange(len(trip_df))
ax2.plot(x2, trip_df["Raw_HR_MAE"], marker="o", color="#d95f02", lw=1.4, label="Unmitigated HR MAE (BPM)")
ax2.plot(x2, trip_df["Mitigated_HR_MAE"], marker="s", color="#1b9e77", lw=1.5, label="Mitigated HR MAE (BPM)")
ax2.set_xticks(x2)
ax2.set_xticklabels([str(r) for r in trip_df["Record"]], rotation=30)
ax2.set_xlabel("MIT-BIH Clinical Patient Record")
ax2.set_ylabel("HR Estimation MAE (BPM)")
ax2.set_title("(b) Per-Patient Recovery Under Triple Compound Faults")
ax2.grid(True, linestyle="--", alpha=0.45)
ax2.legend(loc="upper left", fontsize=8.2)

plt.tight_layout()
plt.savefig("results/fig9_clinical_recovery.png", dpi=300)
plt.close()
print("Saved results/fig9_clinical_recovery.png successfully!")
