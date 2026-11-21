import numpy as np
import pandas as pd

# Simulation configuration
duration = 30
sampling_rate = 100

time = np.arange(0, duration, 1 / sampling_rate)

# Simulated physiological parameters
heart_rate = 75
spo2 = 98

# ECG signal generation
frequency = heart_rate / 60
ecg = np.zeros(len(time))

for i, t in enumerate(time):
    phase = (t * frequency) % 1

    p = 0.10 * np.exp(-((phase - 0.20) / 0.04) ** 2)
    q = -0.15 * np.exp(-((phase - 0.38) / 0.015) ** 2)
    r = 1.00 * np.exp(-((phase - 0.40) / 0.012) ** 2)
    s = -0.25 * np.exp(-((phase - 0.43) / 0.02) ** 2)
    tw = 0.30 * np.exp(-((phase - 0.65) / 0.08) ** 2)

    ecg[i] = p + q + r + s + tw

# Create dataset
data = pd.DataFrame({
    "time_s": time,
    "ecg_mV": ecg,
    "heart_rate_bpm": np.full(len(time), heart_rate),
    "spo2_pct": np.full(len(time), spo2)
})

# Save dataset
data.to_csv("datasets/normal_data.csv", index=False)

print("Simulation completed!")
print("Total samples:", len(data))
print(data.head())
