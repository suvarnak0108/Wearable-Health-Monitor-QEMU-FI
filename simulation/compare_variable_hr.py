import pandas as pd
import numpy as np
from scipy.signal import butter, sosfiltfilt, find_peaks

# Load data
data = pd.read_csv("datasets/variable_hr_fault_data.csv")
ground_truth = pd.read_csv("datasets/variable_hr_ground_truth.csv")

N = len(data)
fs = 100
tolerance = 10

actual = set(ground_truth["sample_index"].astype(int))

# ---------------- ECG FILTERING ----------------

ecg = data["ecg_mV"].to_numpy()

sos = butter(
    3, [5, 15],
    btype="bandpass",
    fs=fs,
    output="sos"
)

filtered_ecg = sosfiltfilt(sos, ecg)

# ---------------- R-PEAK DETECTION ----------------

peaks, _ = find_peaks(
    filtered_ecg,
    distance=int(0.45 * fs),
    prominence=0.15
)

peak_times = data["time_s"].to_numpy()[peaks]

# ---------------- LOCAL HR ESTIMATION ----------------

ecg_hr_estimates = np.full(N, np.nan)

for start in range(0, 60, 10):
    end = start + 10

    local_peaks = peak_times[
        (peak_times >= start) &
        (peak_times < end)
    ]

    if len(local_peaks) >= 2:
        rr = np.diff(local_peaks)
        estimated_hr = 60 / np.median(rr)

        mask = (
            (data["time_s"] >= start) &
            (data["time_s"] < end)
        )

        ecg_hr_estimates[mask] = estimated_hr

# ---------------- BASELINE DETECTOR ----------------

baseline_detected = set(
    data.index[
        ~data["heart_rate_bpm"].between(30, 200)
    ]
)

# ---------------- PROPOSED DETECTOR ----------------

difference = np.abs(
    data["heart_rate_bpm"].to_numpy() - ecg_hr_estimates
)

proposed_detected = set(
    np.where(difference > tolerance)[0]
)

# ---------------- EVALUATION ----------------

def evaluate(name, detected):
    TP = len(actual & detected)
    FP = len(detected - actual)
    FN = len(actual - detected)
    TN = N - TP - FP - FN

    precision = TP / (TP + FP) if TP + FP else 0
    recall = TP / (TP + FN) if TP + FN else 0

    f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall else 0
    )

    print("\n" + name)
    print("TP:", TP)
    print("FP:", FP)
    print("FN:", FN)
    print("TN:", TN)
    print("Precision:", round(precision * 100, 2), "%")
    print("Recall:", round(recall * 100, 2), "%")
    print("F1-score:", round(f1 * 100, 2), "%")


print("Total ECG R-peaks:", len(peaks))

print("\nECG-derived HR per window:")

for start in range(0, 60, 10):
    mask = (
        (data["time_s"] >= start) &
        (data["time_s"] < start + 10)
    )
    print(
        f"{start}-{start+10}s:",
        round(ecg_hr_estimates[np.where(mask)[0][0]], 2),
        "bpm"
    )

evaluate("BASELINE RANGE DETECTOR", baseline_detected)

evaluate(
    "PROPOSED ECG-HR CONSISTENCY DETECTOR",
    proposed_detected
)

# Save detected indices for later verification
pd.DataFrame({
    "detected_index": sorted(proposed_detected)
}).to_csv(
    "results/variable_hr_detected.csv",
    index=False
)

print("\nDetection results saved.")
