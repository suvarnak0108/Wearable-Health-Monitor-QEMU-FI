import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import butter, sosfiltfilt, find_peaks

# Load dataset
data = pd.read_csv("datasets/variable_noisy_ecg.csv")

fs = 100
time = data["time_s"].to_numpy()
ecg = data["ecg_mV"].to_numpy()

# Step 1: Bandpass filtering (5–15 Hz)
sos = butter(3, [5, 15], btype="bandpass",
             fs=fs, output="sos")

filtered_ecg = sosfiltfilt(sos, ecg)

# Step 2: R-peak detection
peaks, properties = find_peaks(
    filtered_ecg,
    distance=int(0.45 * fs),
    prominence=0.15
)

peak_times = time[peaks]

print("Total detected R-peaks:", len(peaks))

# Step 3: Estimate HR for each 10-second interval
results = []

for start in range(0, 60, 10):
    end = start + 10

    # Select peaks inside this interval
    local_peaks = peak_times[
        (peak_times >= start) & (peak_times < end)
    ]

    if len(local_peaks) >= 2:
        rr = np.diff(local_peaks)
        estimated_hr = 60 / np.median(rr)
    else:
        estimated_hr = np.nan

    true_hr = data.loc[
        (data["time_s"] >= start) &
        (data["time_s"] < end),
        "true_hr_bpm"
    ].iloc[0]

    results.append({
        "interval_s": f"{start}-{end}",
        "true_hr_bpm": true_hr,
        "estimated_hr_bpm": estimated_hr,
        "detected_peaks": len(local_peaks)
    })

# Step 4: Display and save results
result_df = pd.DataFrame(results)

print("\nHR Estimation Results:")
print(result_df.to_string(index=False))

result_df.to_csv(
    "results/ecg_hr_estimation.csv",
    index=False
)

# Step 5: Plot comparison
plt.figure(figsize=(10, 5))

x = np.arange(len(result_df))

plt.plot(x, result_df["true_hr_bpm"],
         marker="o", label="True HR")

plt.plot(x, result_df["estimated_hr_bpm"],
         marker="s", label="Estimated HR")

plt.xticks(x, result_df["interval_s"])
plt.xlabel("Time interval (seconds)")
plt.ylabel("Heart rate (bpm)")
plt.title("True vs Estimated HR under ECG Noise")
plt.legend()
plt.grid(True)

plt.savefig("results/hr_estimation_comparison.png", dpi=300)
plt.show()

print("\nResults saved successfully.")
