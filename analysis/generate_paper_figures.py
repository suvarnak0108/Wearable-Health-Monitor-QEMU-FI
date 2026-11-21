import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from scipy.signal import butter, lfilter

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "legend.fontsize": 8.5,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "figure.dpi": 300
})

# ---------------------------------------------------------
# FIGURE 1: Host-Target QEMU Fault-Injection Architecture
# ---------------------------------------------------------
fig, ax = plt.subplots(figsize=(9.5, 4.2))
ax.set_xlim(0, 10)
ax.set_ylim(0, 4.6)
ax.axis("off")

# Host container
host_box = patches.FancyBboxPatch((0.2, 0.35), 4.2, 3.85, boxstyle="round,pad=0.12",
                                  facecolor="#F4F7FB", edgecolor="#2B5B84", linewidth=1.6)
ax.add_patch(host_box)
ax.text(2.3, 3.95, "HOST ENVIRONMENT (Python / WSL Ubuntu)", ha="center", va="center",
        fontweight="bold", fontsize=10, color="#1B3B5A")

# Target container
target_box = patches.FancyBboxPatch((5.2, 0.35), 4.6, 3.85, boxstyle="round,pad=0.12",
                                    facecolor="#F6FAF5", edgecolor="#2E6B3E", linewidth=1.6)
ax.add_patch(target_box)
ax.text(7.5, 3.95, "TARGET EMULATION (QEMU ARM Cortex-M4 MPS2-AN386)", ha="center", va="center",
        fontweight="bold", fontsize=10, color="#1E4B2A")

def draw_node(x, y, w, h, title, sub, color="#FFFFFF", edge="#333333"):
    box = patches.FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.08",
                                 facecolor=color, edgecolor=edge, linewidth=1.2)
    ax.add_patch(box)
    ax.text(x + w/2, y + h*0.63, title, ha="center", va="center", fontweight="bold", fontsize=9)
    ax.text(x + w/2, y + h*0.28, sub, ha="center", va="center", fontsize=8, color="#333333")

draw_node(0.45, 2.75, 3.7, 0.85, "1. Synthetic Physiological Generator", "100 Hz ECG (P-QRS-T), HR (70-95 BPM), SpO2", "#FFFFFF", "#2B5B84")
draw_node(0.45, 1.60, 3.7, 0.85, "2. Multi-Class Fault Injector", "Noise, Dropout, Burst Loss, Range, Stale, Drift", "#FFFFFF", "#2B5B84")
draw_node(0.45, 0.50, 3.7, 0.80, "5. Oracle & Statistical Evaluator", "Confusion Matrix (TP/FP/FN/TN), F1, FPR, MAE", "#EBF2FA", "#2B5B84")

draw_node(5.45, 2.75, 4.1, 0.85, "3. Digital Bandpass & Quadratic R-Peak DSP", "3-Stage Biquad (5-15 Hz) + Adaptive Threshold", "#FFFFFF", "#2E6B3E")
draw_node(5.45, 1.60, 4.1, 0.85, "4. Windowed HR & 5-Stage Fault Classifier", "10s Median RR HR + Range/Stale/Consistency", "#FFFFFF", "#2E6B3E")
draw_node(5.45, 0.50, 4.1, 0.80, "ARM Semihosting Telemetry (BKPT 0xAB)", "Deterministic Cycle-Free Execution Log Output", "#E9F5EB", "#2E6B3E")

arrow_kw = dict(arrowstyle="-|>", color="#333333", lw=1.3, mutation_scale=12)
ax.annotate("", xy=(2.3, 2.50), xytext=(2.3, 2.73), arrowprops=arrow_kw)
ax.annotate("", xy=(5.40, 3.18), xytext=(4.20, 2.05), arrowprops=arrow_kw)
ax.text(4.80, 2.82, "C Header\nVectors", ha="center", va="center", fontsize=7.8, fontweight="bold")
ax.annotate("", xy=(7.5, 2.50), xytext=(7.5, 2.73), arrowprops=arrow_kw)
ax.annotate("", xy=(7.5, 1.35), xytext=(7.5, 1.58), arrowprops=arrow_kw)
ax.annotate("", xy=(4.20, 0.90), xytext=(5.40, 0.90), arrowprops=arrow_kw)
ax.text(4.80, 1.12, "Semihost\nMetrics", ha="center", va="center", fontsize=7.8, fontweight="bold")

plt.tight_layout()
plt.savefig("results/fig1_architecture.png", dpi=300)
plt.close()

# ---------------------------------------------------------
# FIGURE 2: Clean vs High-Noise (0.35 mV) Filtered ECG & Peaks
# ---------------------------------------------------------
df = pd.read_csv("datasets/independent_validation_data.csv")
t_clean = df[(df["noise_std"] == 0.08) & (df["trial"] == 1)].sort_values("time_s").reset_index(drop=True)
t_noisy = df[(df["noise_std"] == 0.35) & (df["trial"] == 1)].sort_values("time_s").reset_index(drop=True)

b, a = butter(3, [5.0 / 50.0, 15.0 / 50.0], btype="band")
filt_clean = lfilter(b, a, t_clean["ecg_mV"].to_numpy()[:1000])
filt_noisy = lfilter(b, a, t_noisy["ecg_mV"].to_numpy()[:1000])
t_sec = t_clean["time_s"].to_numpy()[:1000]

fig, axes = plt.subplots(2, 1, figsize=(8.5, 4.6), sharex=True)
axes[0].plot(t_sec, t_clean["ecg_mV"].to_numpy()[:1000], color="#1f77b4", lw=1.0, label="Low-Noise ECG (0.08 mV)")
axes[0].plot(t_sec, filt_clean, color="#2ca02c", lw=1.2, label="Bandpass Filtered (5-15 Hz)")
axes[0].set_ylabel("Amplitude (mV)")
axes[0].set_title("(a) Low-Noise ECG Regime (0.08 mV) — 100% R-Peak & Fault Detection F1")
axes[0].grid(True, linestyle="--", alpha=0.45)
axes[0].legend(loc="upper right")

axes[1].plot(t_sec, t_noisy["ecg_mV"].to_numpy()[:1000], color="#7f7f7f", lw=0.8, alpha=0.85, label="High-Noise ECG (0.35 mV)")
axes[1].plot(t_sec, filt_noisy, color="#d62728", lw=1.2, label="Bandpass Filtered (5-15 Hz)")
axes[1].set_xlabel("Time (seconds)")
axes[1].set_ylabel("Amplitude (mV)")
axes[1].set_title("(b) High-Noise ECG Regime (0.35 mV) — Suppressing False Spikes via Quadratic Adaptive Thresholding")
axes[1].grid(True, linestyle="--", alpha=0.45)
axes[1].legend(loc="upper right")

plt.tight_layout()
plt.savefig("results/fig2_waveform_comparison.png", dpi=300)
plt.close()

# ---------------------------------------------------------
# FIGURE 3: Fixed vs Adaptive Firmware Across Noise Levels
# ---------------------------------------------------------
base_sum = pd.read_csv("results/qemu_independent_summary.csv")
adap_sum = pd.read_csv("results/qemu_adaptive_summary.csv")

x = np.arange(len(base_sum))
w = 0.35

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.2, 3.8))
ax1.bar(x - w/2, base_sum["F1"] * 100, width=w, color="#d95f02", label="Baseline Fixed-Threshold")
ax1.bar(x + w/2, adap_sum["F1"] * 100, width=w, color="#1b9e77", label="Proposed Adaptive Quadratic")
ax1.set_xticks(x)
ax1.set_xticklabels([f"{v:.2f}" for v in base_sum["noise_std"]])
ax1.set_xlabel("Gaussian Noise Std. Dev. (mV)")
ax1.set_ylabel("Mean F1-Score (%)")
ax1.set_ylim(0, 112)
ax1.set_title("(a) Fault-Detection F1-Score vs. Noise")
ax1.grid(True, axis="y", linestyle="--", alpha=0.45)
ax1.legend(loc="lower left")

ax2.bar(x - w/2, base_sum["FPR"] * 100, width=w, color="#d95f02", label="Baseline Fixed-Threshold")
ax2.bar(x + w/2, adap_sum["FPR"] * 100, width=w, color="#1b9e77", label="Proposed Adaptive Quadratic")
ax2.set_xticks(x)
ax2.set_xticklabels([f"{v:.2f}" for v in base_sum["noise_std"]])
ax2.set_xlabel("Gaussian Noise Std. Dev. (mV)")
ax2.set_ylabel("Mean False Positive Rate (%)")
ax2.set_ylim(0, 95)
ax2.set_title("(b) False Positive Rate (FPR) Reduction")
ax2.grid(True, axis="y", linestyle="--", alpha=0.45)
ax2.legend(loc="upper left")

plt.tight_layout()
plt.savefig("results/fig3_fixed_vs_adaptive.png", dpi=300)
plt.close()

# ---------------------------------------------------------
# FIGURE 4: Multi-Fault Classification Across All 6 Scenarios
# ---------------------------------------------------------
fault_df = pd.read_csv("results/qemu_all_faults_summary.csv")
labels_clean = [
    "Missing\nSamples",
    "Comm.\nBurst Loss",
    "Out-of-Range\nCorruption",
    "Stale\nSensor Data",
    "In-Range\nHR Drift",
    "ECG Noise\n(0.15 mV)"
]
f1_vals = list(fault_df["F1_pct"]) + [round(adap_sum.loc[adap_sum["noise_std"] == 0.15, "F1"].values[0] * 100, 2)]
rec_vals = list(fault_df["Recall_pct"]) + [round(adap_sum.loc[adap_sum["noise_std"] == 0.15, "recall"].values[0] * 100, 2)]

x4 = np.arange(len(labels_clean))
fig, ax = plt.subplots(figsize=(8.8, 3.8))
bars1 = ax.bar(x4 - 0.18, rec_vals, width=0.35, color="#386cb0", label="Mean Recall (%)")
bars2 = ax.bar(x4 + 0.18, f1_vals, width=0.35, color="#1b9e77", label="Mean F1-Score (%)")

for bar in bars2:
    h = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2, h + 1.2, f"{h:.1f}%", ha="center", va="bottom", fontsize=8)

ax.set_xticks(x4)
ax.set_xticklabels(labels_clean)
ax.set_ylabel("Performance (%)")
ax.set_ylim(0, 112)
ax.set_title("QEMU Cortex-M4 Multi-Fault Detection Reliability Across All 6 Fault Taxonomies (20 Trials Each)")
ax.grid(True, axis="y", linestyle="--", alpha=0.45)
ax.legend(loc="lower right")

plt.tight_layout()
plt.savefig("results/fig4_multifault_performance.png", dpi=300)
plt.close()

print("Generated all 4 publication figures in results/ successfully!")
