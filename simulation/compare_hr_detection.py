import pandas as pd
import numpy as np
from scipy.signal import find_peaks

# Load corrupted dataset and ground truth
data = pd.read_csv("datasets/subtle_corrupted_data.csv")
ground_truth = pd.read_csv("datasets/subtle_ground_truth.csv")

N = len(data)

# Actual faulty sample indices
actual = set(ground_truth["sample_index"].astype(int))

# ---------------- BASELINE DETECTOR ----------------

baseline_detected = set(
    data.index[
        ~data["heart_rate_bpm"].between(30, 200)
    ]
)

# ---------------- ECG-HR CONSISTENCY DETECTOR ----------------

fs = 100  # Sampling frequency in Hz

ecg = data["ecg_mV"].to_numpy()

# Detect R-peaks in ECG
peaks, properties = find_peaks(
    ecg,
    distance=int(0.6 * fs),
    prominence=0.5
)

# Estimate heart rate from R-R intervals
rr_intervals = np.diff(peaks) / fs

if len(rr_intervals) == 0:
    raise ValueError("No ECG peaks detected. Check ECG signal.")

ecg_hr = 60 / np.mean(rr_intervals)

print("Detected ECG peaks:", len(peaks))
print("ECG-derived HR:", round(ecg_hr, 2), "bpm")

# Compare reported HR against ECG-derived HR
tolerance = 10

proposed_detected = set(
    data.index[
        abs(data["heart_rate_bpm"] - ecg_hr) > tolerance
    ]
)

# ---------------- PERFORMANCE EVALUATION ----------------

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

    print("\n", name)
    print("TP:", TP)
    print("FP:", FP)
    print("FN:", FN)
    print("TN:", TN)
    print("Precision:", round(precision * 100, 2), "%")
    print("Recall:", round(recall * 100, 2), "%")
    print("F1-score:", round(f1 * 100, 2), "%")


evaluate("BASELINE RANGE DETECTOR", baseline_detected)
evaluate("PROPOSED ECG-HR CONSISTENCY DETECTOR", proposed_detected)
