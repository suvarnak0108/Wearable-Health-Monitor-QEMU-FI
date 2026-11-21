import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Simulation settings
fs = 100
duration = 60
time = np.arange(0, duration, 1/fs)

# Variable heart rate in 10-second intervals
hr_values = [70, 85, 75, 95, 80, 70]

true_hr = np.zeros(len(time))

for i, hr in enumerate(hr_values):
    start = i * 10
    end = (i + 1) * 10
    true_hr[(time >= start) & (time < end)] = hr

# Generate continuous heartbeat times
beat_times = []
current_time = 0

while current_time < duration:
    hr = hr_values[min(int(current_time // 10), 5)]
    beat_times.append(current_time)
    current_time += 60 / hr

# Generate synthetic ECG waveform
ecg = np.zeros(len(time))

for beat in beat_times:
    # P wave
    ecg += 0.15 * np.exp(-((time - (beat - 0.20)) / 0.045)**2)

    # Q wave
    ecg -= 0.15 * np.exp(-((time - (beat - 0.035)) / 0.015)**2)

    # R wave
    ecg += 1.0 * np.exp(-((time - beat) / 0.018)**2)

    # S wave
    ecg -= 0.25 * np.exp(-((time - (beat + 0.035)) / 0.018)**2)

    # T wave
    ecg += 0.30 * np.exp(-((time - (beat + 0.25)) / 0.08)**2)

# Add baseline wander and Gaussian noise
np.random.seed(42)

baseline_wander = 0.08 * np.sin(2 * np.pi * 0.3 * time)
noise = np.random.normal(0, 0.08, len(time))

noisy_ecg = ecg + baseline_wander + noise

# Save dataset
data = pd.DataFrame({
    "time_s": time,
    "ecg_mV": noisy_ecg,
    "true_hr_bpm": true_hr
})

data.to_csv("datasets/variable_noisy_ecg.csv", index=False)

# Plot ECG
plt.figure(figsize=(14, 5))
plt.plot(time, noisy_ecg, linewidth=0.7)
plt.xlabel("Time (seconds)")
plt.ylabel("ECG amplitude (mV)")
plt.title("Synthetic ECG with Variable HR and Noise")
plt.grid(True)
plt.savefig("results/variable_noisy_ecg.png", dpi=300)
plt.show()

print("Experiment 8 dataset generated.")
print("Total samples:", len(data))
print("Duration:", duration, "seconds")
print("Sampling frequency:", fs, "Hz")
print("HR values:", hr_values)
print("Noise standard deviation: 0.08 mV")
print("Baseline wander amplitude: 0.08 mV")
