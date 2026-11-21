import numpy as np
import pandas as pd

fs = 100
duration = 60
time = np.arange(0, duration, 1/fs)

hr_values = [70, 85, 75, 95, 80, 70]
noise_levels = [0.08, 0.15, 0.25, 0.35]
trials = 20

true_hr = np.zeros(len(time))

for i, hr in enumerate(hr_values):
    mask = (time >= i*10) & (time < (i+1)*10)
    true_hr[mask] = hr

# Generate clean ECG
beat_times = []
current_time = 0

while current_time < duration:
    hr = hr_values[min(int(current_time // 10), 5)]
    beat_times.append(current_time)
    current_time += 60 / hr

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

baseline = 0.08 * np.sin(2 * np.pi * 0.3 * time)

all_data = []

for level_index, noise_std in enumerate(noise_levels):

    for trial in range(1, trials + 1):

        seed = 9000 + level_index * 100 + trial
        rng = np.random.default_rng(seed)

        noise = rng.normal(0, noise_std, len(time))
        noisy_ecg = clean_ecg + baseline + noise

        # New fault positions for every trial
        fault_rng = np.random.default_rng(seed + 50000)

        fault_indices = fault_rng.choice(
            len(time), 150, replace=False
        )

        reported_hr = true_hr.copy()
        reported_hr[fault_indices] = 110

        fault = np.zeros(len(time), dtype=int)
        fault[fault_indices] = 1

        trial_data = pd.DataFrame({
            "noise_std": noise_std,
            "trial": trial,
            "time_s": time,
            "ecg_mV": noisy_ecg,
            "true_hr_bpm": true_hr,
            "heart_rate_bpm": reported_hr,
            "fault": fault
        })

        all_data.append(trial_data)

validation = pd.concat(all_data, ignore_index=True)

validation.to_csv(
    "datasets/independent_validation_data.csv",
    index=False
)

print("Independent validation dataset generated.")
print("Noise levels:", noise_levels)
print("Trials per level:", trials)
print("Total trials:", len(noise_levels) * trials)
print("Total samples:", len(validation))
