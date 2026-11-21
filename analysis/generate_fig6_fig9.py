import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 10,
    "axes.titlesize": 10.5,
    "axes.labelsize": 9.5,
    "figure.dpi": 300
})

# ==============================================================================
# FIGURE 6: (a) NSTDB Continuous-HR MAE (Table 7) & (b) 1%-20% Density Sweep (Table 10)
# ==============================================================================
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.8, 3.7))

# Panel (a): Non-Circular Continuous-HR (65-135 BPM) + PhysioNet NSTDB MAE
nstdb_labels = ["Clean\n(65-135 BPM)", "NSTDB BW\n(0.25 mV)", "NSTDB MA\n(0.20 mV)", "NSTDB EM\n(0.25 mV)", "NSTDB EM\n(0.35 mV)"]
raw_mae_nstdb = [0.97, 1.10, 1.50, 1.72, 2.59]
mit_mae_nstdb = [0.97, 1.10, 1.48, 1.70, 2.53]

x1 = np.arange(len(nstdb_labels))
w1 = 0.36
ax1.bar(x1 - w1/2, raw_mae_nstdb, width=w1, color="#5c768d", label="Raw ECG MAE")
ax1.bar(x1 + w1/2, mit_mae_nstdb, width=w1, color="#1b9e77", label="Mitigated MAE")
for i, v in enumerate(mit_mae_nstdb):
    ax1.text(i + w1/2 + 0.03, v + 0.08, f"{v:.2f}", ha="center", va="bottom", fontsize=7.6)

ax1.axhline(5.0, color="#d95f02", linestyle="--", linewidth=1.1, label="IEC 60601-2-27 Limit (5 BPM)")
ax1.set_xticks(x1)
ax1.set_xticklabels(nstdb_labels, fontsize=8.3)
ax1.set_ylim(0, 5.8)
ax1.set_ylabel("Windowed HR MAE (BPM)")
ax1.set_title("(a) Continuous-HR + PhysioNet NSTDB Error")
ax1.grid(True, axis="y", linestyle="--", alpha=0.45)
ax1.legend(loc="upper left", fontsize=7.8)

# Panel (b): Fault-Density Sensitivity Sweep (1% to 20%)
densities = ["1.0%", "2.5%", "5.0%", "10.0%", "20.0%"]
f1_sweep = [100.0, 100.0, 100.0, 100.0, 100.0]
ax2.plot(densities, f1_sweep, marker="o", linewidth=2.0, markersize=6, color="#2b5c8f", label="Missing / Range / Drift (F1)")
for i, v in enumerate(f1_sweep):
    ax2.text(i, v + 0.45, "100.0%", ha="center", va="bottom", fontsize=8.0)
ax2.set_ylim(94.0, 102.5)
ax2.set_xlabel("Injected Fault Density (% of 6,000 Samples)")
ax2.set_ylabel("Sample-Level F1-Score (%)")
ax2.set_title("(b) Fault-Density Sensitivity Sweep (1%–20%)")
ax2.grid(True, linestyle="--", alpha=0.45)
ax2.legend(loc="lower right", fontsize=8.2)

plt.tight_layout()
plt.savefig("results/fig6_hr_error_and_density.png", dpi=300)
plt.close()

# ==============================================================================
# FIGURE 9: (a) 4 Clinical Scenarios (Table 11) & (b) 10-Patient Ablation (Table 12)
# ==============================================================================
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.2, 3.8))

# Panel (a): Scenario-Level Raw vs. Mitigated MAE
scen_names = ["Clean\nBaseline", "Burst +\nStale", "Noise (0.2mV)\n+ Drift", "Triple\nCompound"]
raw_scen = [2.074, 2.765, 6.282, 4.613]
mit_scen = [1.845, 2.220, 2.087, 4.400]

x_s = np.arange(len(scen_names))
w_s = 0.35
ax1.bar(x_s - w_s/2, raw_scen, width=w_s, color="#d95f02", label="Raw ECG MAE")
ax1.bar(x_s + w_s/2, mit_scen, width=w_s, color="#1b9e77", label="Mitigated MAE")
for i, (r, m) in enumerate(zip(raw_scen, mit_scen)):
    ax1.text(i - w_s/2, r + 0.14, f"{r:.2f}", ha="center", va="bottom", fontsize=7.4)
    ax1.text(i + w_s/2 + 0.03, m + 0.14, f"{m:.2f}", ha="center", va="bottom", fontsize=7.4)

ax1.set_xticks(x_s)
ax1.set_xticklabels(scen_names, fontsize=8.5)
ax1.set_ylim(0, 7.4)
ax1.set_ylabel("Mean HR MAE Across 10 Patients (BPM)")
ax1.set_title("(a) Clinical MIT-BIH Scenario Error Reduction")
ax1.grid(True, axis="y", linestyle="--", alpha=0.45)
ax1.legend(loc="upper left", fontsize=8.0)

# Panel (b): Per-Patient Triple Compound Comparison (Full-Wave + 300ms Refractory)
recs = ["100", "101", "103", "105", "106", "108", "113", "115", "119", "200"]
med_base = [9.86, 2.57, 1.05, 1.00, 7.47, 12.68, 20.07, 2.38, 14.65, 4.87]
raw_pat  = [6.75, 0.27, 0.40, 0.77, 3.59, 9.36, 13.17, 0.64, 7.85, 3.33]
mit_pat  = [5.73, 0.27, 0.40, 0.77, 3.52, 8.83, 13.17, 0.64, 7.85, 2.82]

x_p = np.arange(len(recs))
w_p = 0.27
ax2.bar(x_p - w_p, med_base, width=w_p, color="#999999", label="Cum. Median Base.")
ax2.bar(x_p,       raw_pat,  width=w_p, color="#7570b3", label="Raw ECG MAE")
ax2.bar(x_p + w_p, mit_pat,  width=w_p, color="#1b9e77", label="Mitigated MAE")

ax2.set_xticks(x_p)
ax2.set_xticklabels(recs, fontsize=8.2)
ax2.set_ylim(0, 22.5)
ax2.set_xlabel("MIT-BIH Patient Record ID")
ax2.set_ylabel("Windowed HR MAE (BPM)")
ax2.set_title("(b) Per-Patient Ablation (Triple Compound Fault)")
ax2.grid(True, axis="y", linestyle="--", alpha=0.45)
ax2.legend(loc="upper left", fontsize=7.6)

plt.tight_layout()
plt.savefig("results/fig9_clinical_recovery.png", dpi=300)
plt.close()
print("Saved updated results/fig6_hr_error_and_density.png and results/fig9_clinical_recovery.png!")
