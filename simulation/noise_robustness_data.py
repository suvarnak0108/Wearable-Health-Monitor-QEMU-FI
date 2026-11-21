import numpy as np
import pandas as pd

# Simulation configuration
fs = 100
duration = 60
time = np.arange(0, duration, 1/fs)

hr_values = [70, 85, 75, 95, 80, 70]
noise_levels = [0.08, 0.15, 0.25, 0.35]
trials = 10

# Generate variable HR profile
true_hr = np.zeros(len(time))

for i, hr in enumerate(hr_values):
    mask = (time >= i * 10) & (time < (i + 1) * 10)
    true_hr[mask] = hr

# Generate heartbeat times
beat_times = []
current_time = 0

while current_time < duration:
    hr = hr_values[min(int(current_time // 10), 5)]
    beat_times.append(current_time)
    current_time += 60 / hr

# Generate clean synthetic ECG
clean_ecg = np.zeros(len(time))

for beat in beat_times:
    clean_ecg += 0.15 * np.exp(
        -((time - (beat - 0.20)) / 0.045)**2
    )

    clean_ecg -= 0.15 * np.exp(
        -((time - (beat - 0.035)) / 0.015)**2
    )

    clean_ecg += 1.0 * np.exp(
        -((time - beat) / 0.018)**2
    )

    clean_ecg -= 0.25 * np.exp(
        -((time - (beat + 0.035)) / 0.018)**2
    )

    clean_ecg += 0.30 * np.exp(
        -((time - (beat + 0.25)) / 0.08)**2
    )

# Baseline wander
baseline_wander = 0.08 * np.sin(2 * np.pi * 0.3 * time)

# Keep the same 150 faulty HR positions across trials
np.random.seed(42)

fault_indices = np.random.choice(
    len(time), 150, replace=False
)

# Generate independent noise trials
all_results = []

for level_index, noise_std in enumerate(noise_levels):

    for trial in range(1, trials + 1):

        rng = np.random.default_rng(
            1000 + level_index * 100 + trial
        )

        noise = rng.normal(
            0, noise_std, len(time)
        )

        noisy_ecg = (
            clean_ecg + baseline_wander + noise
        )

        reported_hr = true_hr.copy()
        reported_hr[fault_indices] = 110

        trial_data = pd.DataFrame({
            "noise_std": noise_std,
            "trial": trial,
            "time_s": time,
            "ecg_mV": noisy_ecg,
            "true_hr_bpm": true_hr,
            "heart_rate_bpm": reported_hr,
            "fault": 0
        })

        trial_data.loc[fault_indices, "fault"] = 1

        all_results.append(trial_data)

# Combine all trials
final_data = pd.concat(
    all_results, ignore_index=True
)

final_data.to_csv(
    "datasets/noise_robustness_data.csv",
    index=False
)

# Save ground truth fault positions
pd.DataFrame({
    "sample_index": fault_indices
}).to_csv(
    "datasets/noise_robustness_ground_truth.csv",
    index=False
)

print("Noise robustness datasets generated successfully.")
print("Noise levels:", noise_levels)
print("Trials per level:", trials)
print("Total simulation runs:", len(noise_levels) * trials)
print("Total samples:", len(final_data))
print("Faulty HR samples per run:", len(fault_indices))
print("\nDataset dimensions:", final_data.shape)
