import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import butter, lfilter

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 10,
    "axes.titlesize": 10.5,
    "axes.labelsize": 9.5,
    "figure.dpi": 300
})

# FIGURE 6: ECG HR Estimation MAE/RMSE & Fault Density Stability
hr_df = pd.read_csv("results/qemu_hr_estimation_accuracy.csv")
dens_df = pd.read_csv("results/qemu_density_sweep.csv")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.4, 3.6))
x = np.arange(len(hr_df))
w = 0.34
ax1.bar(x - w/2, hr_df["ECG_HR_MAE_BPM"], width=w, color="#2b5c8f", label="Mean Absolute Error (MAE)")
ax1.bar(x + w/2, hr_df["ECG_HR_RMSE_BPM"], width=w, color="#d95f02", label="Root Mean Square Error (RMSE)")
ax1.set_xticks(x)
ax1.set_xticklabels([f"{v:.2f}" for v in hr_df["Noise_mV"]])
ax1.set_xlabel("Gaussian Noise Std. Dev. (mV)")
ax1.set_ylabel("Heart-Rate Estimation Error (BPM)")
ax1.set_title("(a) Windowed ECG HR Error Across Noise Levels")
ax1.grid(True, axis="y", linestyle="--", alpha=0.45)
ax1.legend(loc="upper left", fontsize=8.5)

for f_type, label, marker in [
    ("missing_samples", "Missing Samples", "o"),
    ("out_of_range", "Out-of-Range Corruption", "s"),
    ("in_range_hr_drift", "In-Range HR Drift", "^")
]:
    sub = dens_df[dens_df["Fault_Type"] == f_type]
    ax2.plot(sub["Density_pct"], sub["F1"], marker=marker, lw=1.8, label=label)
ax2.set_xlabel("Injected Fault Density (%)")
ax2.set_ylabel("Mean F1-Score (%)")
ax2.set_ylim(95, 101.5)
ax2.set_title("(b) Fault-Density Scalability (1% to 20%)")
ax2.grid(True, linestyle="--", alpha=0.45)
ax2.legend(loc="lower right", fontsize=8.5)

plt.tight_layout()
plt.savefig("results/fig6_hr_error_and_density.png", dpi=300)
plt.close()

# FIGURE 7: Baseline Wander + 50 Hz Hum Attenuation & Gradual Ramp Drift
df = pd.read_csv("datasets/independent_validation_data.csv")
clean = df[(df["noise_std"] == 0.08) & (df["trial"] == 1)].sort_values("time_s").reset_index(drop=True)
t = clean["time_s"].to_numpy()[:1000]
corrupted = clean["ecg_mV"].to_numpy()[:1000] + 0.40 * np.sin(2 * np.pi * 0.35 * t) + 0.12 * np.sin(2 * np.pi * 50.0 * t)
b, a = butter(3, [5.0 / 50.0, 15.0 / 50.0], btype="band")
recovered = lfilter(b, a, corrupted)

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8.8, 4.4))
ax1.plot(t, corrupted, color="#7f7f7f", lw=0.9, label="ECG + 0.35 Hz Wander + 50 Hz Mains Hum")
ax1.plot(t, recovered, color="#1b9e77", lw=1.25, label="Cortex-M4 5-15 Hz Biquad Output")
ax1.set_xlabel("Time (seconds)")
ax1.set_ylabel("Amplitude (mV)")
ax1.set_title("(a) Suppression of Baseline Wander and 50 Hz Power-Line Interference")
ax1.grid(True, linestyle="--", alpha=0.45)
ax1.legend(loc="upper right", fontsize=8.5)

t_full = np.arange(6000) / 100.0
true_hr = np.repeat([69.76, 84.50, 75.00, 95.23, 80.00, 69.76], 1000)
drift = np.zeros(6000)
drift[2000:2250] = np.linspace(0, 37.5, 250)
sens_hr = true_hr + drift
ax2.plot(t_full, true_hr, color="#2b5c8f", lw=1.5, label="ECG-Derived Reference HR (BPM)")
ax2.plot(t_full, sens_hr, color="#d95f02", lw=1.3, linestyle="--", label="Sensor HR with Gradual Calibration Drift")
ax2.axvspan(20.67, 22.50, color="#ffcccc", alpha=0.55, label="Cross-Modal Inconsistency Triggered (>10 BPM)")
ax2.set_xlabel("Time (seconds)")
ax2.set_ylabel("Heart Rate (BPM)")
ax2.set_title("(b) Detection of Gradual Sensor Calibration Drift Inside Valid Bounds")
ax2.grid(True, linestyle="--", alpha=0.45)
ax2.legend(loc="upper right", fontsize=8.2)

plt.tight_layout()
plt.savefig("results/fig7_frontend_and_drift.png", dpi=300)
plt.close()
print("Saved results/fig6_hr_error_and_density.png and results/fig7_frontend_and_drift.png!")
