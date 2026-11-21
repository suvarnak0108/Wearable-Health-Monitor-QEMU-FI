
import pandas as pd
import matplotlib.pyplot as plt

# Load generated dataset
data = pd.read_csv("datasets/normal_data.csv")

# Create three plots
fig, axes = plt.subplots(3, 1, figsize=(10, 8))

# ECG waveform
axes[0].plot(data["time_s"], data["ecg_mV"])
axes[0].set_title("Synthetic ECG Signal")
axes[0].set_ylabel("Amplitude (mV)")
axes[0].grid(True)

# Heart rate
axes[1].plot(data["time_s"], data["heart_rate_bpm"])
axes[1].set_title("Heart Rate")
axes[1].set_ylabel("BPM")
axes[1].grid(True)

# SpO2
axes[2].plot(data["time_s"], data["spo2_pct"])
axes[2].set_title("Oxygen Saturation")
axes[2].set_ylabel("SpO2 (%)")
axes[2].set_xlabel("Time (seconds)")
axes[2].grid(True)

plt.tight_layout()

# Save figure
plt.savefig("results/normal_data.png", dpi=300)

plt.show()

print("Plots generated successfully!")

