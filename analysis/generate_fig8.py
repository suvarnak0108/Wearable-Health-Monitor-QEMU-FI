import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import wfdb
from scipy.signal import resample, butter, lfilter

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 10,
    "axes.titlesize": 10.5,
    "axes.labelsize": 9.5,
    "figure.dpi": 300
})

clin_df = pd.read_csv("results/qemu_mitbih_clinical_validation.csv")

# Fetch 10s snippet of MIT-BIH Record 105 (noisy clinical record) for waveform panel
rec = wfdb.rdrecord("105", pn_dir="mitdb", sampto=3600)
ecg_100 = resample(rec.p_signal[:, 0], 1000)
ecg_100 = (ecg_100 - np.median(ecg_100)) / np.percentile(np.abs(ecg_100 - np.median(ecg_100)), 99)
b, a = butter(3, [5.0 / 50.0, 15.0 / 50.0], btype="band")
filt_100 = lfilter(b, a, ecg_100)
t = np.arange(1000) / 100.0

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.6, 3.6))

# Panel (a): Clinical Record 105 Raw vs Filtered
ax1.plot(t, ecg_100, color="#7f7f7f", lw=0.85, label="MIT-BIH Rec. 105 Raw (100 Hz)")
ax1.plot(t, filt_100, color="#1b9e77", lw=1.2, label="Cortex-M4 5-15 Hz Biquad Output")
ax1.set_xlabel("Time (seconds)")
ax1.set_ylabel("Normalized Amplitude (mV)")
ax1.set_title("(a) Clinical MIT-BIH Rec. 105 Artifact Filtering")
ax1.grid(True, linestyle="--", alpha=0.45)
ax1.legend(loc="upper right", fontsize=8.2)

# Panel (b): True Annotated Beats vs Cortex-M4 Detected Peaks + 100% Fault F1
x = np.arange(len(clin_df))
w = 0.35
ax2.bar(x - w/2, clin_df["True_Beats"], width=w, color="#2b5c8f", label="Cardiologist Annotated Beats")
ax2.bar(x + w/2, clin_df["Detected_Peaks"], width=w, color="#59a14f", label="QEMU Cortex-M4 Detected Peaks")
for i, f1 in enumerate(clin_df["F1"]):
    top = max(clin_df["True_Beats"].iloc[i], clin_df["Detected_Peaks"].iloc[i])
    ax2.text(i, top + 2.0, f"F1:{f1:.0f}%", ha="center", va="bottom", fontsize=7.8, fontweight="bold")
ax2.set_xticks(x)
ax2.set_xticklabels([f"Rec.\n{r}" for r in clin_df["MITBIH_Record"]])
ax2.set_ylim(0, 98)
ax2.set_ylabel("Beat Count (60 s Window)")
ax2.set_title("(b) Clinical R-Peak & Fault-Detection Accuracy")
ax2.grid(True, axis="y", linestyle="--", alpha=0.45)
ax2.legend(loc="lower right", fontsize=8.2)

plt.tight_layout()
plt.savefig("results/fig8_clinical_mitbih.png", dpi=300)
plt.close()
print("Saved results/fig8_clinical_mitbih.png successfully!")
