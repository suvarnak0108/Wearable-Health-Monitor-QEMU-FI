
import pandas as pd
import numpy as np

# Sampling configuration
sampling_rate = 100
expected_interval = 1 / sampling_rate

# Load fault-injected data and ground truth
data = pd.read_csv("datasets/missing_data.csv")
ground_truth = pd.read_csv("datasets/missing_ground_truth.csv")

# Convert timestamps into sample indices
received_indices = set(
    np.rint(data["time_s"].to_numpy() * sampling_rate).astype(int)
)

actual_missing_indices = set(
    np.rint(ground_truth["time_s"].to_numpy() * sampling_rate).astype(int)
)

# Detect missing samples from timestamp gaps
detected_missing_indices = set()

timestamps = data["time_s"].to_numpy()

for i in range(1, len(timestamps)):
    interval = timestamps[i] - timestamps[i - 1]

    # Estimate the number of missing samples
    gap_samples = round(interval / expected_interval) - 1

    if gap_samples > 0:
        previous_index = round(timestamps[i - 1] * sampling_rate)

        for j in range(1, gap_samples + 1):
            detected_missing_indices.add(previous_index + j)

# Compare detection with ground truth
TP = len(detected_missing_indices & actual_missing_indices)
FP = len(detected_missing_indices - actual_missing_indices)
FN = len(actual_missing_indices - detected_missing_indices)

precision = TP / (TP + FP) if TP + FP else 0
recall = TP / (TP + FN) if TP + FN else 0
f1 = (
    2 * precision * recall / (precision + recall)
    if precision + recall else 0
)

print("\n--- Missing Sample Detection Results ---")
print("Actual missing samples:", len(actual_missing_indices))
print("Detected missing samples:", len(detected_missing_indices))
print("True positives:", TP)
print("False positives:", FP)
print("False negatives:", FN)

print("Precision: {:.2f}%".format(precision * 100))
print("Detection rate (recall): {:.2f}%".format(recall * 100))
print("F1-score: {:.2f}%".format(f1 * 100))

# Save results
pd.DataFrame({
    "actual_missing_indices": sorted(actual_missing_indices),
}).to_csv("results/actual_missing_indices.csv", index=False)

pd.DataFrame({
    "detected_missing_indices": sorted(detected_missing_indices),
}).to_csv("results/detected_missing_indices.csv", index=False)
