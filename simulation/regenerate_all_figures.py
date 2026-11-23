import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pathlib import Path
from scipy.signal import butter, lfilter

ROOT = Path.cwd()
RESULTS_DIR = ROOT / "results"
RESULTS_DIR.mkdir(exist_ok=True)

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 10,
    "axes.titlesize": 10.5,
    "axes.labelsize": 9.5,
    "figure.dpi": 300
})

# ==============================================================================
# FIG 1: Host-Target Functional Emulation Architecture
# ==============================================================================
fig, ax = plt.subplots(figsize=(10.5, 4.6))
ax.set_xlim(0, 10.5)
ax.set_ylim(0, 4.6)
ax.axis("off")

host_box = patches.FancyBboxPatch((0.2, 0.25), 4.4, 4.1, boxstyle="round,pad=0.12,rounding_size=0.18",
                                  linewidth=1.5, edgecolor="#2b5c8f", facecolor="#f0f5fa")
target_box = patches.FancyBboxPatch((5.9, 0.25), 4.4, 4.1, boxstyle="round,pad=0.12,rounding_size=0.18",
                                    linewidth=1.5, edgecolor="#1b7856", facecolor="#eff8f4")
ax.add_patch(host_box)
ax.add_patch(target_box)

ax.text(2.4, 4.12, "HOST ENVIRONMENT (Python 3 / WSL Ubuntu)", ha="center", va="center",
        fontsize=10, fontweight="bold", color="#1e4166")
ax.text(8.1, 4.12, "TARGET EMULATION (QEMU ARM Cortex-M4)", ha="center", va="center",
        fontsize=10, fontweight="bold", color="#13573e")

def draw_step(x, y, title, sub, edge):
    b = patches.FancyBboxPatch((x, y), 3.9, 0.85, boxstyle="round,pad=0.06,rounding_size=0.12",
                               linewidth=1.1, edgecolor=edge, facecolor="white")
    ax.add_patch(b)
    ax.text(x + 1.95, y + 0.54, title, ha="center", va="center", fontsize=8.8, fontweight="bold", color="#222222")
    ax.text(x + 1.95, y + 0.22, sub, ha="center", va="center", fontsize=7.6, color="#444444")

draw_step(0.45, 2.95, "1. Multi-Source Physiological Dataset Generator",
          "56–134 BPM ECG + 1 Hz Aux + 10 MIT-BIH Records", "#2b5c8f")
draw_step(0.45, 1.72, "2. 8-Class & Compound Fault Injector",
          "Noise, Dropouts, Bursts, Range, Stale, Drift (1–20%)", "#2b5c8f")
draw_step(0.45, 0.48, "6. Verification Oracle & Statistical Evaluator",
          "Safety Properties (P1–P4), AAMI EC57, MAE & SQI", "#2b5c8f")

draw_step(6.15, 2.95, "3. Cascaded Biquad & Adaptive R-Peak DSP",
          "3-Stage 5–15 Hz Butterworth + Quadratic 300 ms R-Peak", "#1b7856")
draw_step(6.15, 1.72, "4. 5-Stage Classifier & Active HR Recovery",
          "Fault Isolation + Physiological RR Filter + SQI Gate", "#1b7856")
draw_step(6.15, 0.48, "5. ARM Semihosting & Stack Telemetry",
          "Bare-Metal mps2-an386 (2,312 B Code / 3,080 B HWM)", "#1b7856")

# Vertical arrows
for x_mid in [2.4, 8.1]:
    ax.annotate("", xy=(x_mid, 2.63), xytext=(x_mid, 2.92), arrowprops=dict(arrowstyle="->", lw=1.4, color="#333333"))
    ax.annotate("", xy=(x_mid, 1.39), xytext=(x_mid, 1.69), arrowprops=dict(arrowstyle="->", lw=1.4, color="#333333"))

# Cross-tier arrows
ax.annotate("", xy=(6.10, 3.38), xytext=(4.40, 2.15),
            arrowprops=dict(arrowstyle="->", lw=1.6, color="#111111"))
ax.text(5.25, 2.86, "Static C Header\nTest Vectors\n(.h Arrays)", ha="center", va="center",
        fontsize=7.6, fontweight="bold", bbox=dict(boxstyle="round,pad=0.22", fc="white", ec="#666666"))

ax.annotate("", xy=(4.40, 0.90), xytext=(6.10, 0.90),
            arrowprops=dict(arrowstyle="->", lw=1.6, color="#111111"))
ax.text(5.25, 0.90, "Semihosted\nMetrics & Logs", ha="center", va="center",
        fontsize=7.6, fontweight="bold", bbox=dict(boxstyle="round,pad=0.22", fc="white", ec="#666666"))

plt.tight_layout()
plt.savefig(RESULTS_DIR / "fig1_architecture.png", dpi=300)
plt.close()

# ==============================================================================
# FIG 2: Raw vs 5-15 Hz Biquad-Filtered ECG (0.08 mV vs 0.35 mV)
# ==============================================================================
fs = 100.0
t = np.arange(1000) / fs
ecg_clean = np.zeros(1000)
for bt in np.arange(0.05, 9.9, 60.0 / 70.0):
    dt = t - bt
    ecg_clean += (0.12*np.exp(-((dt+0.16)/0.025)**2) - 0.15*np.exp(-((dt+0.035)/0.010)**2)
                  + 1.00*np.exp(-(dt/0.012)**2) - 0.22*np.exp(-((dt-0.035)/0.010)**2)
                  + 0.28*np.exp(-((dt-0.22)/0.045)**2))

b_bp, a_bp = butter(3, [5.0 / 50.0, 15.0 / 50.0], btype="bandpass")
rng = np.random.default_rng(42)
raw_low = ecg_clean + rng.normal(0, 0.08, size=1000)
filt_low = lfilter(b_bp, a_bp, raw_low)
raw_high = ecg_clean + rng.normal(0, 0.35, size=1000)
filt_high = lfilter(b_bp, a_bp, raw_high)

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9.4, 4.6), sharex=True)
ax1.plot(t, raw_low, color="#377eb8", lw=0.95, alpha=0.85, label="Low-Noise ECG (0.08 mV)")
ax1.plot(t, filt_low, color="#238b45", lw=1.25, label="Bandpass Filtered (5–15 Hz)")
ax1.set_ylabel("Amplitude (mV)")
ax1.set_title("(a) Low-Noise ECG Regime (0.08 mV) — 0.209 BPM Mitigated HR MAE")
ax1.grid(True, linestyle="--", alpha=0.45)
ax1.legend(loc="upper right", fontsize=8.2)

ax2.plot(t, raw_high, color="#888888", lw=0.85, alpha=0.75, label="High-Noise ECG (0.35 mV)")
ax2.plot(t, filt_high, color="#cb181d", lw=1.25, label="Bandpass Filtered (5–15 Hz)")
ax2.set_xlabel("Time (seconds)")
ax2.set_ylabel("Amplitude (mV)")
ax2.set_title("(b) Severe Broadband Noise Regime (0.35 mV) — 300 ms Refractory Noise Admission Limit")
ax2.grid(True, linestyle="--", alpha=0.45)
ax2.legend(loc="upper right", fontsize=8.2)

plt.tight_layout()
plt.savefig(RESULTS_DIR / "fig2_waveform_comparison.png", dpi=300)
plt.close()

# ==============================================================================
# FIG 3: Table 5 Gaussian Noise Ablation (Mean R-Peaks & Mitigated HR MAE)
# ==============================================================================
sigmas = ["0.08", "0.15", "0.25", "0.35"]
rpeaks_t5 = [79.10, 81.20, 94.70, 111.20]
mae_t5 = [0.209, 1.087, 16.294, 39.007]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.4, 3.5))
x3 = np.arange(len(sigmas))
bars1 = ax1.bar(x3, rpeaks_t5, width=0.48, color="#2b5c8f", label="Detected R-Peaks (Mean)")
ax1.axhline(79.0, color="#d95f02", linestyle="--", lw=1.2, label="True Beat Count (79)")
for b, v in zip(bars1, rpeaks_t5):
    ax1.text(b.get_x() + b.get_width()/2, v + 1.5, f"{v:.1f}", ha="center", va="bottom", fontsize=8.0)
ax1.set_xticks(x3); ax1.set_xticklabels(sigmas)
ax1.set_ylim(0, 128); ax1.set_xlabel("Gaussian Noise Std. Dev. (mV)")
ax1.set_ylabel("Mean Detected R-Peaks"); ax1.set_title("(a) Detected Peaks vs. True Beats (300 ms Refractory)")
ax1.grid(True, axis="y", linestyle="--", alpha=0.45); ax1.legend(loc="upper left", fontsize=8.0)

bars2 = ax2.bar(x3, mae_t5, width=0.48, color="#1b9e77", label="Mitigated HR MAE (BPM)")
for b, v in zip(bars2, mae_t5):
    ax2.text(b.get_x() + b.get_width()/2, v + 0.8, f"{v:.2f}", ha="center", va="bottom", fontsize=8.0)
ax2.set_xticks(x3); ax2.set_xticklabels(sigmas)
ax2.set_ylim(0, 45); ax2.set_xlabel("Gaussian Noise Std. Dev. (mV)")
ax2.set_ylabel("Mitigated HR MAE (BPM)"); ax2.set_title("(b) Windowed HR Error Under White Noise")
ax2.grid(True, axis="y", linestyle="--", alpha=0.45); ax2.legend(loc="upper left", fontsize=8.0)

plt.tight_layout()
plt.savefig(RESULTS_DIR / "fig3_fixed_vs_adaptive.png", dpi=300)
plt.close()

# ==============================================================================
# FIG 4: Table 8 Five Primary Fault Classes (Realistic 1 Hz Aux)
# ==============================================================================
scens4 = ["Missing\nSamples (F2)", "Comm.\nBurst (F3)", "Out-of-\nRange (F4)", "ECG Stale\n1.6 s (F5)", "In-Range Drift\n+22 BPM (F6)"]
recs4 = [100.0, 100.0, 100.0, 11.0, 93.87]
f1s4  = [43.76, 43.64, 43.76, 6.07, 41.65]

fig, ax = plt.subplots(figsize=(8.0, 3.4))
x4 = np.arange(len(scens4)); w4 = 0.34
ax.bar(x4 - w4/2, recs4, width=w4, color="#2b5c8f", label="Sample Recall (%)")
ax.bar(x4 + w4/2, f1s4,  width=w4, color="#1b9e77", label="Sample F1-Score (%)")
for i, (r, f) in enumerate(zip(recs4, f1s4)):
    ax.text(i - w4/2, r + 1.5, f"{r:.1f}%", ha="center", va="bottom", fontsize=7.6)
    ax.text(i + w4/2, f + 1.5, f"{f:.1f}%", ha="center", va="bottom", fontsize=7.6)
ax.set_xticks(x4); ax.set_xticklabels(scens4, fontsize=8.4)
ax.set_ylim(0, 115); ax.set_ylabel("Classification Score (%)")
ax.set_title("QEMU Cortex-M4 Fault Classifier Performance Across Five Fault Classes (1 Hz Quantized Aux)")
ax.grid(True, axis="y", linestyle="--", alpha=0.45); ax.legend(loc="upper right", fontsize=8.2)
plt.tight_layout()
plt.savefig(RESULTS_DIR / "fig4_multifault_performance.png", dpi=300)
plt.close()

# ==============================================================================
# FIG 5: Table 11 Compound Multi-Fault Stress + Table 14 Footprint (2,312 B)
# ==============================================================================
cnames8 = ["Noise +\nMissing", "Burst +\n1.6 s Stale", "Noise +\nDrift", "Triple\nCompound"]
recs8 = [100.0, 34.43, 88.09, 94.80]
f1s8  = [20.31, 19.16, 18.13, 32.44]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.6, 3.6))
x8 = np.arange(len(cnames8)); w8 = 0.34
ax1.bar(x8 - w8/2, recs8, width=w8, color="#2b5c8f", label="Recall (%)")
ax1.bar(x8 + w8/2, f1s8,  width=w8, color="#1b9e77", label="F1-Score (%)")
for i, (r, f) in enumerate(zip(recs8, f1s8)):
    ax1.text(i - w8/2, r + 1.5, f"{r:.1f}%", ha="center", va="bottom", fontsize=7.5)
    ax1.text(i + w8/2, f + 1.5, f"{f:.1f}%", ha="center", va="bottom", fontsize=7.5)
ax1.set_xticks(x8); ax1.set_xticklabels(cnames8, fontsize=8.2); ax1.set_ylim(0, 115)
ax1.set_ylabel("Classification Score (%)"); ax1.set_title("(a) Concurrent Compound Stress (1 Hz Aux)")
ax1.grid(True, axis="y", linestyle="--", alpha=0.45); ax1.legend(loc="upper right", fontsize=8.2)

mods = ["ECG Biquad Filter", "Windowed HR Est.", "Adaptive R-Peak", "5-Stage Classifier", "Active HR Recovery", "Control & Semihost"]
szs = [128, 168, 368, 376, 440, 832]
cols = ["#4e79a7", "#59a14f", "#f28e2b", "#e15759", "#76b7b2", "#bab0ac"]
ym = np.arange(len(mods))
bars = ax2.barh(ym, szs, color=cols, height=0.58)
for bar, sz in zip(bars, szs):
    ax2.text(sz + 18, bar.get_y() + bar.get_height()/2, f"{sz} B", va="center", fontsize=8.2)
ax2.set_yticks(ym); ax2.set_yticklabels(mods); ax2.set_xlim(0, 1020)
ax2.set_xlabel("ARM Thumb-2 Code Size (Bytes) — Total: 2,312 B")
ax2.set_title("(b) Cortex-M4 Compiled Module Footprint (2,312 B)")
ax2.grid(True, axis="x", linestyle="--", alpha=0.45)

plt.tight_layout()
plt.savefig(RESULTS_DIR / "fig5_compound_and_footprint.png", dpi=300)
plt.close()

# ==============================================================================
# FIG 6: Table 6 NSTDB MAE + Table 10 Density Sweep Sensitivity
# ==============================================================================
nlabs = ["Clean\n(56–134 BPM)", "NSTDB BW\n(0.25 mV)", "NSTDB MA\n(0.20 mV)", "NSTDB EM\n(0.25 mV)", "NSTDB EM\n(0.35 mV)"]
raw6 = [0.97, 1.10, 1.50, 1.72, 2.59]
mit6 = [0.97, 1.10, 1.48, 1.70, 2.53]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.8, 3.6))
x6 = np.arange(len(nlabs)); w6 = 0.36
ax1.bar(x6 - w6/2, raw6, width=w6, color="#5c768d", label="Raw ECG MAE")
ax1.bar(x6 + w6/2, mit6, width=w6, color="#1b9e77", label="RR-Filtered MAE")
for i, v in enumerate(mit6):
    ax1.text(i + w6/2 + 0.03, v + 0.08, f"{v:.2f}", ha="center", va="bottom", fontsize=7.6)
ax1.axhline(5.0, color="#d95f02", linestyle="--", lw=1.1, label="5.0 BPM Reference")
ax1.set_xticks(x6); ax1.set_xticklabels(nlabs, fontsize=8.1); ax1.set_ylim(0, 5.8)
ax1.set_ylabel("Windowed HR MAE (BPM)"); ax1.set_title("(a) Continuous-HR + PhysioNet NSTDB Error")
ax1.grid(True, axis="y", linestyle="--", alpha=0.45); ax1.legend(loc="upper left", fontsize=7.8)

dens = ["1.0%", "2.5%", "5.0%", "10.0%", "20.0%"]
ax2.plot(dens, [100.0]*5, marker="o", lw=2.0, color="#2b5c8f", label="Missing / Range / Drift (Recall)")
for i in range(5):
    ax2.text(i, 100.45, "100.0%", ha="center", va="bottom", fontsize=8.0)
ax2.set_ylim(94.0, 102.5); ax2.set_xlabel("Injected Fault Density (% of 6,000 Samples)")
ax2.set_ylabel("Rule Detection Recall (%)"); ax2.set_title("(b) Fault-Density Sensitivity Sweep (1%–20%)")
ax2.grid(True, linestyle="--", alpha=0.45); ax2.legend(loc="lower right", fontsize=8.2)

plt.tight_layout()
plt.savefig(RESULTS_DIR / "fig6_hr_error_and_density.png", dpi=300)
plt.close()

# ==============================================================================
# FIG 7: Front-End Wander + 50 Hz Hum Suppression & Gradual Drift Detection
# ==============================================================================
raw_hum = ecg_clean + 0.35 * np.sin(2 * np.pi * 0.35 * t) + 0.18 * np.sin(2 * np.pi * 50.0 * t)
filt_hum = lfilter(b_bp, a_bp, raw_hum)

t60 = np.arange(6000) / fs
ref_hr = np.repeat([70.0, 84.0, 75.0, 95.0, 80.0, 70.0], 1000)
sens_hr = ref_hr.copy()
ramp = np.linspace(0, 27.0, 1800)  # +1.5 BPM/s ramp from t=20s to t=38s
sens_hr[2000:3800] += ramp

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9.4, 4.6))
ax1.plot(t, raw_hum, color="#888888", lw=0.85, alpha=0.8, label="ECG + 0.35 Hz Wander + 50 Hz Mains Hum")
ax1.plot(t, filt_hum, color="#1b9e77", lw=1.25, label="Cortex-M4 5–15 Hz Biquad Output")
ax1.set_xlabel("Time (seconds)"); ax1.set_ylabel("Amplitude (mV)")
ax1.set_title("(a) Suppression of Baseline Wander and 50 Hz Power-Line Interference")
ax1.grid(True, linestyle="--", alpha=0.45); ax1.legend(loc="upper right", fontsize=8.2)

ax2.plot(t60, ref_hr, color="#2b5c8f", lw=1.5, label="ECG-Derived Reference HR (BPM)")
ax2.plot(t60, sens_hr, color="#d95f02", linestyle="--", lw=1.4, label="Sensor HR with Gradual Calibration Drift (+1.5 BPM/s)")
ax2.axvspan(28.0, 38.0, color="#fbb4ae", alpha=0.45, label="Cross-Modal Inconsistency Triggered (>12 BPM)")
ax2.set_xlabel("Time (seconds)"); ax2.set_ylabel("Heart Rate (BPM)")
ax2.set_title("(b) Detection of Gradual Sensor Calibration Drift Inside Valid Bounds")
ax2.grid(True, linestyle="--", alpha=0.45); ax2.legend(loc="upper right", fontsize=8.0)

plt.tight_layout()
plt.savefig(RESULTS_DIR / "fig7_frontend_and_drift.png", dpi=300)
plt.close()

# ==============================================================================
# FIG 9: Table 12 & 13 Clinical MIT-BIH Error Reduction & Per-Patient Ablation
# ==============================================================================
snames = ["Clean\nBaseline", "Burst +\nStale", "NSTDB EM\n+ Drift", "Triple\nCompound"]
raw12 = [2.243, 2.243, 1.805, 1.401]
mit12 = [2.149, 2.149, 1.751, 1.289]

recs13 = ["100", "101", "103", "105", "106", "108", "113", "115", "119", "200"]
med13  = [0.387, 2.469, 0.983, 1.050, 3.652, 11.498, 0.690, 1.882, 3.343, 5.724]
raw13  = [0.610, 0.002, 0.339, 0.478, 0.143, 7.970, 0.364, 0.508, 0.599, 2.994]
mit13  = [0.610, 0.002, 0.339, 0.478, 0.143, 6.857, 0.364, 0.508, 0.599, 2.994]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.2, 3.8))
xs = np.arange(len(snames)); ws = 0.35
ax1.bar(xs - ws/2, raw12, width=ws, color="#d95f02", label="Raw ECG MAE")
ax1.bar(xs + ws/2, mit12, width=ws, color="#1b9e77", label="RR-Filtered MAE")
for i, (r, m) in enumerate(zip(raw12, mit12)):
    ax1.text(i - ws/2, r + 0.05, f"{r:.2f}", ha="center", va="bottom", fontsize=7.5)
    ax1.text(i + ws/2 + 0.03, m + 0.05, f"{m:.2f}", ha="center", va="bottom", fontsize=7.5)
ax1.set_xticks(xs); ax1.set_xticklabels(snames, fontsize=8.4); ax1.set_ylim(0, 2.8)
ax1.set_ylabel("Mean HR MAE Across 10 Patients (BPM)"); ax1.set_title("(a) Clinical MIT-BIH Scenario Error Reduction")
ax1.grid(True, axis="y", linestyle="--", alpha=0.45); ax1.legend(loc="upper right", fontsize=8.0)

xp = np.arange(len(recs13)); wp = 0.27
ax2.bar(xp - wp, med13, width=wp, color="#999999", label="Cum. Median Base.")
ax2.bar(xp,      raw13, width=wp, color="#7570b3", label="Raw ECG MAE")
ax2.bar(xp + wp, mit13, width=wp, color="#1b9e77", label="RR-Filtered MAE")
ax2.set_xticks(xp); ax2.set_xticklabels(recs13, fontsize=8.2)
ax2.set_ylim(0, 13.0); ax2.set_xlabel("MIT-BIH Patient Record ID")
ax2.set_ylabel("Windowed HR MAE (BPM)"); ax2.set_title("(b) Per-Patient Ablation (Triple Compound Fault)")
ax2.grid(True, axis="y", linestyle="--", alpha=0.45); ax2.legend(loc="upper left", fontsize=7.6)

plt.tight_layout()
plt.savefig(RESULTS_DIR / "fig9_clinical_recovery.png", dpi=300)
plt.close()

print("Saved all 8 figures cleanly to:", RESULTS_DIR)
